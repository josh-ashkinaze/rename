"""
Author: Joshua Ashkinaze

Description: Generates controversial academic topics for a list of domains using AI.

Inputs:
    - None

Outputs:
    - data/clean/ai_generated_topics.json: A JSON file with generated topics for each domain

Date: 2025-01-29 17:47:50
"""


import os
import litellm
import json
from dotenv import load_dotenv
load_dotenv("../src/.env")


os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def get_topics_for_domain(domain):
    """
    Get topic suggestions for a domain using LiteLLM.

    Args:
        domain: Domain to get topics for

    Returns:
        List of suggested topics
    """

    system_prompt = """You are a helpful AI assistant that suggests controversial academic topics.
    You will return your response as a JSON object with a 'topics' key containing an array of strings.
    Each string should be a controversial academic topic."""

    user_prompt = f"""
    Return 10 common phrases that describe controversial topics or academic ideas for the domain: {domain}

    Requirements for each phrase:
    - Each phrase should be highly popular and has been talked about a lot. 
    - The specific phrase is popular; not just the topic.  
    - Each phrase should describe a discrete idea and not a general topic. 
    - Each phrase should be currently polarizing. 
    - Each phrase should be concise---if X is controversial, do not say X [Y] where [Y] is some other word.
    - Each phrase should be 1-4 words  
    - Each phrase should describe an idea that originated in academic contexts. That is: We do not want memes or highly "online" terms  
    - Each phrase should be highly popular and has been talked about a lot. 
    - The specific phrase is popular; not just the topic.  
    - Each phrase should describe an idea that originated in academic contexts. That is: We do not want memes or highly "online" terms

    Return your response as a JSON object with format: {{"topics": ["topic1", "topic2", "topic3", "topic4", "topic5"]}}
    """

    try:
        response = litellm.completion(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
            seed=42
        )

        response_text = response.choices[0].message.content
        topics_data = json.loads(response_text)

        return topics_data.get('topics', [])

    except Exception as e:
        print(f"Error getting topics for domain {domain}: {str(e)}")
        return []


def get_all_domain_topics(domains):
    """
    Get topics for multiple domains.

    Args:
        domains: List of domains to get topics for

    Returns:
        Dictionary mapping domains to their topics
    """
    all_topics = {}
    for domain in domains:
        topics = get_topics_for_domain(domain)
        if topics:
            all_topics[domain] = topics
            print(f"\nTopics for {domain}:")
            for topic in topics:
                print(f"- {topic}")
        else:
            print(f"\nNo topics generated for {domain}")

    return all_topics



domains = ['Guns', 'Gender Identity And Sexual Orientation', 'Race And Ethnicity', 'Science', 'Education']

# Generate topics
print("Generating topics for domains...")
topics_dict = get_all_domain_topics(domains)
output_file = '../data/clean/ai_generated_topics.json'
with open(output_file, 'w') as f:
    json.dump(topics_dict, f, indent=2)
print(f"\nResults saved to {output_file}")

print("\nSummary of generated topics:")
for domain, topics in topics_dict.items():
    print(f"\n{domain}: {len(topics)} topics generated")
    for topic in topics:
        print(f"- {topic}")