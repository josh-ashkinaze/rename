"""
Script to analyze Pew Research survey data comparing Harris and Trump supporter responses.

The goal is to find domains where Harris and Trump supporters have divergent responses. So to do this, we
take the responses for each question and compute JSD between Trump and Harris supporters.

Note:
- A small percent of people say 'refused' and that proportion is removed from the distribution when we do the JSD
stuff

Input:
 - ../data/raw/pew_data/merged_survey_data.json: JSON file containing survey data.

Output:
- ../data/clean/pew_question_Level_analysis.csv: CSV file containing processed survey data with computed metrics at question level
- ../data/clean/pew_domain_level_analysis.csv: CSV file containing summary statistics by domain
- ../data/clean/top3_domains.txt: Text file containing the top three domains with the highest JSD
- ../tables/pew_summary_stats_by_domain.tex: LaTeX table containing summary statistics by domain
"""

import pandas as pd
import json
from scipy.spatial import distance
import numpy as np
import logging
import os

logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)

def open_json(file: str):
    """
    Load JSON data from a file.
    """
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def compute_array_metrics(ar1, ar2):
    """
    Compute Jensen-Shannon divergence between two probability distributions.

    Args:
        ar1: First probability distribution array.
        ar2: Second probability distribution array.

    Returns:
        Float representing Jensen-Shannon divergence, or None if computation fails.
    """
    try:
        return distance.jensenshannon(ar1, ar2)
    except:
        return None


def parse_question(q):
    """
    Parse and analyze a survey question's response data.

    This function processes survey response data for a single question, computing various
    metrics including normalized responses, entropy, and Jensen-Shannon divergence between
    Harris and Trump supporter responses.

    Args:
        q: Dictionary containing question data with the following structure:
           {
               'question_id': str,
               'question': str,
               'responses': {
                   'harris_supporters': Dict[str, int],
                   'trump_supporters': Dict[str, int]
               },
               'domain': str
           }

    Returns:
        Dictionary containing processed data and computed metrics:

    """
    qid = q['question_id']
    q_text = q['question']
    harris_responses = q['responses']['harris_supporters'].copy()
    trump_responses = q['responses']['trump_supporters'].copy()

    # Remove 'refused' responses
    harris_responses.pop('refused', None)
    trump_responses.pop('refused', None)

    harris_responses_values = list(harris_responses.values())
    trump_responses_values = list(trump_responses.values())

    # Normalize responses
    harris_responses_normalized = [i / sum(harris_responses_values) for i in harris_responses_values]
    trump_responses_normalized = [i / sum(trump_responses_values) for i in trump_responses_values]

    # Calculate entropy and other metrics
    data = {
        'qid': qid,
        'question': q_text,
        'harris_responses': harris_responses,
        'trump_responses': trump_responses,
        'harris_responses_values': harris_responses_values,
        'trump_responses_values': trump_responses_values,
        'harris_responses_normalized': harris_responses_normalized,
        'trump_responses_normalized': trump_responses_normalized,
        'harris_responses_normalized_sum': sum(harris_responses_normalized),
        'trump_responses_normalized_sum': sum(trump_responses_normalized),
        'jsd': compute_array_metrics(harris_responses_normalized, trump_responses_normalized),
        'domain': q['domain']
    }

    if data['jsd'] is None:
        logging.info(f"Warning: Could not compute JSD for question {qid}")
        logging.info(str(data))

    return data


def main(input_file):
    """
    Main function to process Pew survey data and create analysis DataFrame.

    Args:
        input_file: Path to the input JSON file containing survey data.

    Returns:
        pandas DataFrame containing processed survey data with computed metrics.
    """
    # Load and process data
    survey_data = open_json(input_file)

    # Process each question
    processed_data = []
    for key in survey_data.keys():
        processed_data.append(parse_question(survey_data[key]))

    # Create and clean DataFrame
    df = pd.DataFrame(processed_data)
    df = df.dropna(subset=['jsd'])

    return df


if __name__ == "__main__":
    input_file = "../data/raw/pew_data/merged_survey_data.json"
    df = main(input_file)
    df.to_csv("../data/clean/pew_question_Level_analysis.csv", index=False)

    # Make table
    aggd = df.groupby('domain')['jsd'].agg(['mean', 'median', 'std']).reset_index()
    aggd = aggd.rename({'std': 'sd'}, axis=1)
    aggd['domain'] = aggd['domain'].replace({"gender_identity_sexor": "Gender Identity and Sexual Orientation"})
    aggd['domain'] = aggd['domain'].replace({"govt_scope_role": "Government Scope and Role"})
    aggd['domain'] = aggd['domain'].replace({"gender_family_reproductive_issues": "Gender, Family, and Reproductive Issues"})

    aggd['snr'] = aggd['mean'] / aggd['sd']
    aggd.columns = [c.title() for c in aggd.columns]
    aggd.columns = [c.upper() if c in ['Sd', 'Snr'] else c for c in aggd.columns]

    for col in ['Mean', 'Median', 'SD', 'SNR']:
        aggd[col] = aggd[col].map('{:.2f}'.format)

    aggd = aggd.sort_values('Mean', ascending=False)

    aggd['Domain'] = aggd['Domain'].str.replace('_', ' ').str.title()
    
    # latex table
    latex_output = aggd.to_latex(index=False,
                                 caption='JSD by Pew domain. Higher values indicate greater divergence between Harris and Trump supporters.',
                                 label='jsd_measures',
                                 escape=False)    
    
    with open("../tables/pew_summary_stats_by_domain.tex", "w") as f:
        f.write(latex_output)

    # write csv
    aggd.to_csv("../data/clean/pew_domain_level_analysis.csv", index=False)
    
    # just write the top three domains 
    top3 = aggd.head(3)['Domain'].to_list()
    logging.info(top3)
    with open("../data/clean/top3_domains.txt", "w") as f:
        for item in top3:
            f.write("%s\n" % item)
    


    logging.info(aggd)