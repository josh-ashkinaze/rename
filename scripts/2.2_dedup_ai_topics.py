"""
Deduplicates and analyzes AI-generated controversial topics across different domains and models.

This script processes AI-generated topics, standardizes similar terms, and generates statistics about
topic frequency across domains and models.


INPUTS:
- data/clean/ai_generated_topics_dt.json: A JSON file with AI-generated topics for each domain

OUTPUTS
- ...{fn}_standardized.json: Original data with standardized topic names
    Format: Same as input but with standardized topic_list. What this means is that if a term maps to multiple others,
    we replace it with a single term. So eg (defund police, defunding the police) -> defund the police

- ...{fn}_unique_topics.txt: Simple list of all unique topics
    Format: One topic per line

- ...{fn}_topic_stats.json: Stats for each unique topic in the standardized json
    {
        "topic": str,           # Topic name
        "domains": list[str],   # List of domains where topic appears
        "models": list[str],    # List of models that generated the topic
        "n_domains": int,       # Number of domains
        "n_models": int         # Number of models
    }


Author: Joshua Ashkinaze
Date: 2025-02-12 17:59:06
"""

import json
import itertools
from difflib import SequenceMatcher
from collections import defaultdict
import os 
import logging
import yaml
with open("config.yaml", 'r') as stream: config = yaml.safe_load(stream)


logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)


def basic_stats(l):
    """Print basic statistics of a list"""
    from collections import Counter
    most_common = Counter(l).most_common(1)[0][0]
    stats_dict = {
        "total": len(l),
        "unique": len(set(l)),
        "duplicates": len(l) - len(set(l)),
        "duplicates_ratio": (len(l) - len(set(l))) / len(l),
        "most_common": most_common
    }
    return stats_dict


def get_similarity(a, b):
    """Calculate string similarity between two strings (0-1 score)"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_similar_topics(topics, threshold=0.8):
    """Find pairs of topics that exceed similarity threshold"""
    similar_pairs = defaultdict(list)
    for i, topic1 in enumerate(topics):
        for topic2 in topics[i + 1:]:
            similarity = get_similarity(topic1, topic2)
            if similarity >= threshold:
                similar_pairs[topic1].append((topic2, similarity))
                similar_pairs[topic2].append((topic1, similarity))
    return dict(similar_pairs)


def print_similar_topics(similar_pairs):
    """Print similar topic pairs"""
    for topic, matches in similar_pairs.items():
        if matches:
            logging.info(f"\n{topic}:")
            for similar_topic, score in matches:
                logging.info(f"  - {similar_topic} (similarity: {score:.2f})")


def dedup2standard(dedup_dict, l):
    """Loops through topic list and replaces similar topics with standard topic"""
    return [dedup_dict.get(x, x) for x in l]


def get_topic_stats(data):
    """Create topic statistics from JSON data"""
    topic_stats = defaultdict(lambda: {'domains': set(), 'models': set()})

    for item in data:
        domain = item['domain']
        model = item['model']
        topics = item['topic_list']

        for topic in topics:
            topic_stats[topic]['domains'].add(domain)
            topic_stats[topic]['models'].add(model)

    results = []
    for topic, stats in topic_stats.items():
        results.append({
            'topic': topic,
            'domains': sorted(list(stats['domains'])),
            'models': sorted(list(stats['models'])),
            'n_domains': len(stats['domains']),
            'n_models': len(stats['models'])
        })

    return results


def verify_standardization(standardized_data, dedup2standard_dict):
    """verify that we have successfully standardized topics"""
    standardized_topics = list(itertools.chain(*[item['topic_list'] for item in standardized_data]))

    # ensure things to be remapped in dedup2standard_dict are remapped in standardized_topics
    original_terms = set(dedup2standard_dict.keys())
    remaining_unmapped = original_terms.intersection(set(standardized_topics))

    if remaining_unmapped:
        logging.error(f"Found terms that should have been mapped but weren't: {remaining_unmapped}")
        return False

    return True

def save_topic_stats(stats, output_path):
    """Save topic statistics to JSON file"""
    if not stats:
        return

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)



def main():


    # 1. Load data
    ######################################
    ######################################
    # Load and process JSON file
    fn = f"../data/clean/ai_generated_topics{config['datetime_pull']}.json"
    logging.info(f"Loading data from {fn}")
    with open(fn) as f:
        data = json.load(f)

    # 2. Get initial stats
    ######################################
    ######################################
    topic_lists = [item['topic_list'] for item in data]
    topic_list = list(itertools.chain(*topic_lists))
    initial_stats = basic_stats(topic_list)
    logging.info(f"Initial stats before de-duplication {str(initial_stats)}")

    # 3. Get similar topics so we can standardize
    ######################################
    ######################################
    unique_topics = list(set(topic_list))
    unique_topics = sorted(unique_topics)
    logging.info("start <LOGGING ALL TOPICS>")
    for topic in unique_topics:
        logging.info(f"{topic}")
    logging.info("end <LOGGING ALL TOPICS>")


    similar_pairs = find_similar_topics(unique_topics)
    print_similar_topics(similar_pairs)

    # Standardize topics
    dedup2standard_dict = {
        "affordable care": "affordable care act",
        "austerity": "austerity measures",
        "campaign finance": "campaign finance reform",
        "defund police": "defund the police",
        "defunding the police": "defund the police",
        "opioid epidemic": "opioid crisis",
        "single-payer": "single-payer system",
        "sustainable logging": "sustainability",
        "three strikes": "three strikes laws",
        "three strikes law": "three strikes laws"
    }

    standardized_data = []
    for item in data:
        new_item = item.copy()
        new_item['topic_list'] = dedup2standard(dedup2standard_dict, item['topic_list'])
        standardized_data.append(new_item)

    # Get stats after standardization
    standardized_topics = list(itertools.chain(*[item['topic_list'] for item in standardized_data]))
    new_stats = basic_stats(standardized_topics)
    logging.info(f"Stats after de-duplication {str(new_stats)}")

    # Verify standardization worked correctly
    assert verify_standardization(standardized_data, dedup2standard_dict), "Standardization verification failed"
    standard_fn = fn.replace(".json", "_standardized.json")
    logging.info(f"Saving standardized topics to {standard_fn}")
    with open(standard_fn, 'w') as f:
        json.dump(standardized_data, f, indent=2)

    # 4. save text file of unique topics
    ######################################
    ######################################
    unique_topics = sorted(list(set(standardized_topics)))
    assert len(unique_topics) == new_stats['unique'], "Unique topics list is not correct"
    txt_fn = fn.replace(".json", "_unique_topics.txt")
    with open(txt_fn, 'w') as f:
        f.write('\n'.join(unique_topics))
    logging.info(f"Saving unique topics list to {txt_fn}")

    # 5. Let's get a json for annotation that is all the unique
    # topics and their domains and models
    ######################################
    ######################################
    logging.info("\nGenerating topic statistics...")
    topic_stats = get_topic_stats(standardized_data)
    topic_stats.sort(key=lambda x: (x['n_domains'] + x['n_models']), reverse=True)

    # Save topic statistics to JSON
    stats_fn = fn.replace(".json", "_topic_stats.json")
    save_topic_stats(topic_stats, stats_fn)
    logging.info(f"Saved topic statistics to {stats_fn}")

    # Print summary of most frequent topics
    logging.info("\nTop 5 most frequent topics:")
    for topic in topic_stats[:5]:
        logging.info(f"\nTopic: {topic['topic']}")
        logging.info(f"Appeared in {topic['n_domains']} domains and used by {topic['n_models']} models")
        logging.info(f"Domains: {', '.join(topic['domains'])}")
        logging.info(f"Models: {', '.join(topic['models'])}")


if __name__ == "__main__":
    main()