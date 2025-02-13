"""
Author: Joshua Ashkinaze

Description: Gets mediacloud data for unique AI-generated topics.

Inputs:
    - data/clean/ai_generated_topics_dt_unique_topics.txt: A txt file with unique topics after deduplication

Outputs:
    - data/clean/mediacloud_raw_dt_START_DATE_END_DATE.csv: Raw mediacloud data

        cols:
            - date: date of query
            - total_count: total count of hits
            - count: count of mentions FOR THAT QUERY
            - ratio: count/total_count
            - query: query used
            - phrase: phrase used
            - query_type: base or controversy
            - is_dummy: 1 if phrase is in DUMMY_PHRASES, 0 otherwise


    - data/clean/mediacloud_daily_dt_START_DATE_END_DATE.csv: Daily aggregated mediacloud data
        cols:
            - date: see above
            - phrase: see above
            - total_mentions of phrase on date
            - controversy_mentions of phrase on date
            - is_dummy: see above


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

START_DATE = "2021-01-01"
END_DATE = "2025-02-01"

# Setup logging
logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log",
                    level=logging.INFO,
                    format='%(asctime)s: %(message)s',
                    filemode='w',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)

load_dotenv("../src/.env")

logging.info(f"Starting MediaCloud analysis for datetime pull: {config['datetime_pull']}")
logging.info(f"Start date: {START_DATE}, End date: {END_DATE}")
logging.info(f"Dummy phrases: {DUMMY_PHRASES}")


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
    """Get MediaCloud data and save both raw and daily aggregated metrics."""

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
        return None

    raw_df = pd.concat(all_results, ignore_index=True)

    raw_df['is_dummy'] = raw_df['phrase'].isin(DUMMY_PHRASES).astype(int)

    pivot_df = pd.pivot_table(
        raw_df,
        values='count',
        index=['date', 'phrase'],
        columns='query_type',
        fill_value=0
    ).reset_index()

    pivot_df.columns.name = None
    pivot_df = pivot_df.rename(columns={
        'base': 'total_mentions',
        'controversy': 'controversy_mentions'
    })

    pivot_df['is_dummy'] = pivot_df['phrase'].isin(DUMMY_PHRASES).astype(int)

    return raw_df, pivot_df


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

        if results is not None:
            raw_df, daily_df = results

            logging.info("\nResults summary:")
            logging.info(f"Total phrases processed: {len(raw_df['phrase'].unique())}")
            logging.info(f"Date range: {raw_df['date'].min()} to {raw_df['date'].max()}")

            # Save raw data
            raw_output_file = f'../data/clean/mediacloud_raw_{config['datetime_pull']}_{START_DATE}_{END_DATE}.csv'
            raw_df.to_csv(raw_output_file, index=False)
            logging.info(f"\nSaved raw results to {raw_output_file}")

            # Save daily aggregated data
            daily_output_file = f'../data/clean/mediacloud_daily_{config['datetime_pull']}_{START_DATE}_{END_DATE}.csv'
            daily_df.to_csv(daily_output_file, index=False)
            logging.info(f"Saved daily aggregated results to {daily_output_file}")

            # Print summary statistics
            total_phrases = len(raw_df['phrase'].unique())
            total_mentions = raw_df[raw_df['query_type'] == 'base']['count'].sum()
            total_controversy = raw_df[raw_df['query_type'] == 'controversy']['count'].sum()

            logging.info(f"\nSummary Statistics:")
            logging.info(f"Total unique phrases: {total_phrases}")
            logging.info(f"Total mentions: {total_mentions}")
            logging.info(f"Total controversy mentions: {total_controversy}")

        else:
            logging.info("No results to display")

    except Exception as e:
        logging.info(f"Error running analysis: {e}")
        raise e