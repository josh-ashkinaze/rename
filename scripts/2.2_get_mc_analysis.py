"""
Author: Joshua Ashkinaze

Description: Gets mediacloud data for unique AI-generated topics.

Inputs:
    - data/clean/ai_generated_topics_dt_unique_topics.txt: A txt file with unique topics after deduplication

Outputs:
    - data/clean/mediacloud_analysis_dt.csv: A CSV file with MediaCloud analysis for each topic containing:
        - total_mentions: Number of times topic mentioned
        - controversy_mentions: Number of mentions with controversy-related terms
        - controversy_ratio: Ratio of controversy mentions to total mentions
        - mention_rank: Percentile rank of total mentions
        - controversy_rank: Percentile rank of controversy ratio
        - composite_score: 0.5 * mention_rank + 0.5 * controversy_rank
        - is_dummy: Boolean indicating if this is a control phrase

Date: 2025-02-12 17:58:55
"""

import os
from mediacloud.api import SearchApi
import pandas as pd
from datetime import datetime, date
from tqdm import tqdm
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
import time
import logging
from dotenv import load_dotenv
import yaml
with open("config.yaml", 'r') as stream: config = yaml.safe_load(stream)

# Global variables
DUMMY_PHRASES = [
    "weather forecast",
    "parking space",
    "bus schedule",
    "post office",
    "gas station",
    "traffic light",
    "phone number",
    "store hours",
    "parking meter",
    "train schedule",
    "movie ticket",
    "lunch menu"
]

START_DATE = "2024-01-01"
END_DATE = "2024-01-02"

# Setup logging
logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log",
                   level=logging.INFO,
                   format='%(asctime)s: %(message)s',
                   filemode='w',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   force=True)

load_dotenv("../src/.env")

logging.info(f"Starting MediaCloud analysis for datetime pull: {config['datetime_pull']}")

def make_controversy_query(phrase):
    """Create a controversy-focused version of a query."""
    return f'"{phrase}" AND (controversial OR controversy OR debate OR debated OR criticize OR criticized)'

@retry(
    retry=retry_if_exception_type((requests.exceptions.RequestException, RuntimeError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def query_mediacloud(query_info, collection_id, start_date, end_date, api_key):
    """Run a single MediaCloud query with retry logic."""
    query = query_info['query']
    mc = SearchApi(api_key)

    start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()

    try:
        results = mc.story_count_over_time(
            query,
            start_dt,
            end_dt,
            [collection_id]
        )

        if not results:
            logging.info(f"No results for query: {query}")
            return None

        df = pd.DataFrame(results)
        if df.empty:
            logging.info(f"Empty results for query: {query}")
            return None

        df['date'] = pd.to_datetime(df['date'])
        df['query'] = query
        df['phrase'] = query_info['phrase']
        df['query_type'] = query_info['type']

        total_count = df['count'].sum()
        logging.info(f"Query: {query}, Total Count: {total_count}")

        return df

    except Exception as e:
        logging.info(f"Error querying '{query}': {e}")
        if "429" in str(e):
            logging.info("Rate limit hit, retrying with backoff...")
            raise requests.exceptions.RequestException("Rate limit")
        return None

def batch_queries(queries, batch_size=5):
    """Split queries into batches."""
    for i in range(0, len(queries), batch_size):
        yield queries[i:i + batch_size]

def analyze_phrases(phrases, api_key, collection_id=34412234,
                   start_date=START_DATE, end_date=END_DATE,
                   max_workers=3, batch_size=5):
    """Analyze phrases using MediaCloud and compute composite scores."""

    # Generate queries
    queries = []
    for phrase in phrases:
        queries.append({
            'phrase': phrase,
            'query': f'"{phrase}"',
            'type': 'base'
        })
        queries.append({
            'phrase': phrase,
            'query': make_controversy_query(phrase),
            'type': 'controversy'
        })

    all_results = []
    # Process in batches using ThreadPoolExecutor
    for batch in tqdm(list(batch_queries(queries, batch_size)), desc="Processing batches"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {
                executor.submit(
                    query_mediacloud,
                    query_info,
                    collection_id,
                    start_date,
                    end_date,
                    api_key
                ): query_info for query_info in batch
            }

            for future in concurrent.futures.as_completed(future_to_query):
                try:
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                except Exception as e:
                    logging.info(f"Query failed: {e}")

        time.sleep(1)

    if not all_results:
        logging.info("No valid results found")
        return pd.DataFrame()

    # Combine results
    df = pd.concat(all_results, ignore_index=True)

    # Calculate metrics
    metrics = []
    for phrase in phrases:
        phrase_data = df[df['phrase'] == phrase]

        base_data = phrase_data[phrase_data['query_type'] == 'base']
        controversy_data = phrase_data[phrase_data['query_type'] == 'controversy']

        base_counts = base_data['count'].sum() if not base_data.empty else 0
        controversy_counts = controversy_data['count'].sum() if not controversy_data.empty else 0

        controversy_ratio = controversy_counts / base_counts if base_counts > 0 else 0

        metrics.append({
            'phrase': phrase,
            'total_mentions': base_counts,
            'controversy_mentions': controversy_counts,
            'controversy_ratio': controversy_ratio,
            'success': not phrase_data.empty,
            'is_dummy': phrase in DUMMY_PHRASES
        })

    metrics_df = pd.DataFrame(metrics)

    metrics_df['mention_rank'] = metrics_df['total_mentions'].rank(pct=True)
    metrics_df['controversy_rank'] = metrics_df['controversy_ratio'].rank(pct=True)
    metrics_df['composite_score'] = 0.5 * metrics_df['mention_rank'] + 0.5 * metrics_df['controversy_rank']
    metrics_df = metrics_df.sort_values('composite_score', ascending=False)

    return metrics_df

if __name__ == "__main__":
    logging.info("Starting MediaCloud analysis")


    # Read unique topics from txt file
    topics_file = f"../data/clean/ai_generated_topics{config['datetime_pull']}_unique_topics.txt"
    logging.info(f"Reading topics from {topics_file}")

    with open(topics_file, 'r') as f:
        phrases = [line.strip() for line in f if line.strip()]

    logging.info(f"Found {len(phrases)} unique topics")

    # Add dummy phrases
    phrases.extend(DUMMY_PHRASES)
    phrases = list(set(phrases))
    logging.info(f"Added {len(DUMMY_PHRASES)} dummy phrases. Total phrases: {len(phrases)}")

    try:
        results = analyze_phrases(
            phrases=phrases,
            api_key=os.environ['MEDIACLOUD_API_KEY']
        )

        if not results.empty:
            logging.info("\nResults summary:")
            logging.info(f"Total phrases processed: {len(results)}")
            logging.info(f"Dummy phrases: {len(results[results['is_dummy']])}")
            logging.info(f"Real topics: {len(results[~results['is_dummy']])}")

            output_file = '../data/clean/mediacloud_analysis.csv'
            results.to_csv(output_file, index=False)
            logging.info(f"\nSaved results to {output_file}")

            # Print success summary
            success_count = results['success'].sum()
            total_count = len(results)
            logging.info(f"Successfully processed {success_count} out of {total_count} phrases")

            # Log dummy phrase statistics
            dummy_results = results[results['is_dummy']]
            logging.info("\nDummy phrase statistics:")
            logging.info(f"Average controversy ratio: {dummy_results['controversy_ratio'].mean():.3f}")
            logging.info(f"Median controversy ratio: {dummy_results['controversy_ratio'].median():.3f}")
        else:
            logging.info("No results to display")

    except Exception as e:
        logging.info(f"Error running analysis: {e}")
        raise e