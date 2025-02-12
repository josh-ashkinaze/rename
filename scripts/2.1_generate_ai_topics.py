"""
Author: Joshua Ashkinaze

Description: Generates controversial academic topics for a list of domains using AI.

Inputs:
    - data/clean/gallup_top_topics.csv: A CSV file with domains and descriptions

Outputs:
    - data/clean/ai_generated_topics_{DT}.json: A JSON file with generated topics for each domain

Date: 2025-01-29 17:47:50
"""

import os
import litellm
import json
from dotenv import load_dotenv
import pandas as pd
import re
import logging

from  datetime  import datetime


DT = datetime.now().strftime("%Y-%m-%d__%H:%M:%S")





load_dotenv("../src/.env")

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')

MODELS = [
    'claude-3-sonnet-20240229',
    'gpt-4-0125-preview',
    'gpt-3.5-turbo',
]

logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)

logging.info("models")
logging.info(MODELS)


def parse_llm_topics(response_text: str) -> list:
    """
    Parse topics from LLM response text with enhanced robustness.

    Args:
        response_text: Raw text response from LLM containing topics

    Returns:
        list: Cleaned list of topics
    """

    def clean_topic(topic: str) -> str:
        """Clean individual topic strings."""
        topic = re.sub(r'^\d+\.\s*', '', topic)
        topic = re.sub(r'^[-•*]\s*', '', topic)
        topic = re.split(r'[.:]', topic)[0]
        topic = topic.strip(' "\'[](){}<>')
        topic = re.split(r'\s+[-–]\s+', topic)[0]

        return topic.strip().lower()

    def extract_topics_from_text(text: str) -> list:
        """Extract topics from plain text with various formats."""
        
        
        potential_topics = []

        # lets try newline split
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) > 1:
            potential_topics.extend(lines)
        else:
            # try splitting by commas
            potential_topics.extend([t.strip() for t in text.split(',')])

            # try splitting by semicolons
            if len(potential_topics) <= 1:
                potential_topics = [t.strip() for t in text.split(';')]

        return [clean_topic(topic) for topic in potential_topics if clean_topic(topic)]

    response_text = response_text.strip()

    # Try parsing as JSON first
    try:
        data = json.loads(response_text)
        if isinstance(data, dict) and 'topics' in data:
            topics = [clean_topic(t) for t in data['topics']]
            return [t for t in topics if t]
        elif isinstance(data, list):
            topics = [clean_topic(t) for t in data]
            return [t for t in topics if t]
    except json.JSONDecodeError:
        pass

    # Try finding lists in the text using regex
    list_pattern = r'\[(.*?)\]'
    matches = re.findall(list_pattern, response_text)
    if matches:
        for match in matches:
            try:
                topics = json.loads(f"[{match}]")
                topics = [clean_topic(t) for t in topics]
                if len(topics) > 1:
                    return [t for t in topics if t]
            except json.JSONDecodeError:
                continue

    # response contains "TOPICS:" or similar
    if re.search(r'topics:?\s*', response_text, re.IGNORECASE):
        content = re.split(r'topics:?\s*', response_text, flags=re.IGNORECASE)[-1]
        topics = extract_topics_from_text(content)
        if len(topics) > 1:
            return topics

    #  numbered lists (e.g., "1. topic1\n2. topic2")
    numbered_topics = re.findall(r'^\d+\.\s*(.+)$', response_text, re.MULTILINE)
    if len(numbered_topics) > 1:
        topics = [clean_topic(t) for t in numbered_topics]
        return [t for t in topics if t]

    #  bullet points
    bullet_topics = re.findall(r'[-•*]\s*(.+)(?:\n|$)', response_text)
    if len(bullet_topics) > 1:
        topics = [clean_topic(t) for t in bullet_topics]
        return [t for t in topics if t]

    # fallback: try to extract topics from plain text
    topics = extract_topics_from_text(response_text)

    return topics if len(topics) > 1 else []  

def generate_topics():
    """Main function to generate topics using multiple models."""

    input_file = '../data/clean/gallup_top_topics.csv'
    topics_df = pd.read_csv(input_file)

    all_results = []

    for model in MODELS:
        logging.info(f"\nProcessing with model: {model}")

        for x, row in topics_df.iterrows():
            user_prompt = f"""INSTRUCTIONS
Given a domain, return a list of 5 common phrases that describe controversial
topics or academic ideas.

CONSTRAINTS
- Each phrase should be highly popular and has been talked about a lot
- Each phrase should describe a discrete idea and not a general topic
- Each phrase should be currently polarizing
- Each phrase should be 1-4 words
- Each phrase should describe an idea that originated in academic or policy contexts
- We do not want memes or highly 'online' terms
- We want the shortest and simplest phrases that describe the idea. For example, do not say "[thing] policies" when "[thing]", alone, is sufficient. 

# Input 
DOMAIN: The domain
DESCRIPTION: A brief description of the domain
SUBTOPICS: A list of subtopics within the domain

# Expected Output
TOPICS: A list like ["topic1", "topic2", "topic3", "topic4", "topic5"] and nothing else

###
DOMAIN: {row['topic_str']}
DESCRIPTION: {row['description']}
SUBTOPICS: {row['subtopics']}
"""

            try:
                response = litellm.completion(
                    model=model,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                )

                response_text = response.choices[0].message.content
                topics = parse_llm_topics(response_text)

                if topics:
                    result = {
                        "model": model,
                        "domain": row['topic_str'],
                        "raw_response": response_text,
                        "user_prompt": user_prompt,
                        "topic_list": topics
                    }
                    all_results.append(result)

                    logging.info(f"\nTopics for {row['topic_str']} using {model}:")
                    for topic in topics:
                        logging.info(f"- {topic}")

            except Exception as e:
                logging.info(f"Error with {model} for domain {row['topic_str']}: {str(e)}")
                continue

    output_file = f'../data/clean/ai_generated_topics_{DT}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"\nAll results saved to {output_file}")


    logging.info("\nSummary of generated topics:")
    for model in MODELS:
        model_results = [r for r in all_results if r['model'] == model]
        logging.info(f"\n{model}: Generated topics for {len(model_results)} domains")
        for result in model_results:
            logging.info(f"\n{result['domain']}: {len(result['topic_list'])} topics")
            for topic in result['topic_list']:
                logging.info(f"- {topic}")

if __name__ == "__main__":
    generate_topics()