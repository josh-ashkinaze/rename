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

from datetime import datetime


DT = datetime.now().strftime("%Y-%m-%d__%H:%M:%S")





load_dotenv("../src/.env")

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')

MODELS = [
    'gemini/gemini-2.0-flash',
    'gpt-4o-2024-08-06',
    'gpt-4-turbo-2024-04-09',
]

logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)

logging.info("models")
logging.info(MODELS)


def parse_llm_topics(response_text: str) -> list:
    """
    Parse topics from LLM response text with enhanced robustness.
    Handles Gemini's code block format and escaped characters.

    Args:
        response_text: Raw text response from LLM containing topics

    Returns:
        list: Cleaned list of topics
    """

    def clean_topic(topic: str) -> str:
        """Clean individual topic strings."""
        # Remove code block markers
        topic = re.sub(r'```\s*', '', topic)

        # Remove escaped quotes
        topic = re.sub(r'\\"', '"', topic)

        # Basic cleaning as before
        topic = re.sub(r'^\d+\.\s*', '', topic)
        topic = re.sub(r'^[-•*]\s*', '', topic)
        topic = re.split(r'[.:]', topic)[0]
        topic = topic.strip(' "\'[](){}<>')
        topic = re.split(r'\s+[-–]\s+', topic)[0]

        # Remove trailing commas and quotes
        topic = re.sub(r'",?$', '', topic)

        return topic.strip().lower()

    def extract_topics_from_text(text: str) -> list:
        """Extract topics from plain text with various formats."""
        potential_topics = []

        # Handle code blocks first
        code_block_pattern = r'```(?:json)?\s*(.+?)```'
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        if code_blocks:
            for block in code_blocks:
                try:
                    # Try parsing as JSON
                    data = json.loads(block)
                    if isinstance(data, dict) and any(k.lower() == 'topics' for k in data.keys()):
                        topics_key = next(k for k in data.keys() if k.lower() == 'topics')
                        potential_topics.extend(data[topics_key])
                        return [clean_topic(t) for t in potential_topics if clean_topic(t)]
                except json.JSONDecodeError:
                    continue

        # If no code blocks or JSON parsing failed, try regular text parsing
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

    # Try parsing as JSON first (for non-code-block JSON)
    try:
        data = json.loads(response_text)
        if isinstance(data, dict) and any(k.lower() == 'topics' for k in data.keys()):
            topics_key = next(k for k in data.keys() if k.lower() == 'topics')
            topics = [clean_topic(t) for t in data[topics_key]]
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
                # Handle both JSON-style and plain comma-separated lists
                if '"' in match or "'" in match:
                    topics = json.loads(f"[{match}]")
                else:
                    topics = [t.strip() for t in match.split(',')]
                topics = [clean_topic(t) for t in topics]
                if len(topics) > 1:
                    return [t for t in topics if t]
            except json.JSONDecodeError:
                continue

    # Check for "TOPICS:" indicator
    if re.search(r'topics:?\s*', response_text, re.IGNORECASE):
        content = re.split(r'topics:?\s*', response_text, flags=re.IGNORECASE)[-1]
        topics = extract_topics_from_text(content)
        if len(topics) > 1:
            return topics

    # Try numbered lists
    numbered_topics = re.findall(r'^\d+\.\s*(.+)$', response_text, re.MULTILINE)
    if len(numbered_topics) > 1:
        topics = [clean_topic(t) for t in numbered_topics]
        return [t for t in topics if t]

    # Try bullet points
    bullet_topics = re.findall(r'[-•*]\s*(.+)(?:\n|$)', response_text)
    if len(bullet_topics) > 1:
        topics = [clean_topic(t) for t in bullet_topics]
        return [t for t in topics if t]

    # Final fallback
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
Given a domain, return a list of 7 common phrases that describe controversial
topics or ideas. 

CONSTRAINTS
- Each phrase should be highly popular and has been talked about a lot
- Each phrase should be currently polarizing in the sense that: (A) It provokes strong disagreement between political groups; (B) The concept can be described in alternative, less charged language
- Each phrase should describe a discrete idea and not a general topic
- Each phrase should be 1-4 words, with widely-known acronyms allowed 
- Each phrase should originate from academic, policy, or institutional contexts. That is: We do not want memes or highly "online" terms
- Each phrase should be the shortest and simplest phrase that describes the idea. For example, do not say "[thing] policies" when "[thing]", alone, is sufficient

AVOID
- Avoid phrases that include the name of a specific identity. For example, "transgender rights" or "gay marriage" is not allowed since these include "transgender" and "gay". But "critical race theory" is allowed since it does not name a specific race or identity. 
- Avoid phrases related to atrocities (e.g: "genocide"), hate ideologies (e.g: "terrorism"), human rights violations (e.g.: "torture"), public health misinformation (e.g.: "covid denial"), or where renaming could constitute historical revisionism (e.g.: "slavery") 

# Input 
DOMAIN: The domain
DESCRIPTION: A brief description of the domain
SUBTOPICS: A list of subtopics within the domain

# Expected Output
TOPICS: A list like ["topic1", "topic2", "topic3", "topic4", "topic5", "topic6", "topic7"] and nothing else

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