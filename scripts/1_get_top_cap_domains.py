"""
Author: Joshua Ashkinaze

Description: Gets topics and descriptions for the most important CAP problems from Gallup MIP (2013-2023).

Date: 2025-02-07 13:03:44

Inputs:
    - ../data/raw/gallup_data/US-Public-Gallups_Most_Important_Problem-21.3.csv: CAP + MIP data
    - ../data/raw/cap/cap_codebook.jsonl: CAP major topics and subtopics

Outputs:
    - ../data/clean/gallup_top_topics.csv: Top 10 topics from Gallup data by MIP importance in the last 10 years
    - ../tables/gallup_top_topics.tex: LaTeX table of the top 10 topics from Gallup data by MIP importance in the last 10 years
"""

import pandas as pd
import json
import logging
import os

logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)

if __name__ == "__main__":

    N_YEAR_WINDOW = 10

    cap_codebook_path = "../data/raw/cap/cap_codebook.jsonl"

    # Load CAP codebook
    topics = []
    with open(cap_codebook_path, "r") as f:
        for line in f:
            try:
                topics.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON: {e}")

    # Create mapping dictionaries
    major2code = {topic["name"]: topic["major_code"] for topic in topics}
    code2major = {topic["major_code"]: topic["name"] for topic in topics}
    code2desc = {topic["major_code"]: topic["description"] for topic in topics}
    code2subtopics = {}
    for topic in topics:
        subtopics = [sub["name"] for sub in topic.get("subtopics", []) if sub["name"] not in {"General", "Other"}]
        code2subtopics[topic["major_code"]] = ", ".join(subtopics)

    logging.info("Available CAP topics: %s", str(list(major2code.keys())))

    # Load Gallup data
    gallup_data_path = "../data/raw/gallup_data/US-Public-Gallups_Most_Important_Problem-21.3.csv"
    df = pd.read_csv(gallup_data_path)

    # Map topic names and descriptions
    df['topic_str'] = df['majortopic'].map(code2major)
    df['description'] = df['majortopic'].map(code2desc)
    df['major_code'] = df['majortopic']
    df['subtopics'] = df['majortopic'].map(code2subtopics)
    df['year'] = df['year'].astype(int)

    # Drop rows with missing topic names or percentage values
    df = df.dropna(subset=['topic_str', 'percent'])

    logging.info(f"Number of unique topics: {df['topic_str'].nunique()}")

    max_year = df['year'].max()
    min_year = max_year - N_YEAR_WINDOW
    subset = df[df['year'] >= min_year]

    subset = subset.groupby(['major_code', 'topic_str', 'description', 'subtopics']).agg({
        'percent': 'mean'
    }).sort_values('percent', ascending=False).reset_index()

    logging.info("\nTop 10 topics with descriptions:")
    for _, row in subset.head(10).iterrows():
        logging.info(f"\n{row['topic_str']} (Major Code: {row['major_code']})")
        logging.info(f"Description: {row['description']}")
        logging.info(f"Subtopics: {row['subtopics']}")

    # save the top 10 topics to CSV
    top = subset.head(10)
    top.to_csv("../data/clean/gallup_top_topics.csv", index=False)
    top_topics = str(top['topic_str'].tolist())
    logging.info(top_topics)

    # pretty latex table
    top.columns = ['Major Code', 'Topic', 'Description', 'Subtopics', 'Percent']
    top = top.drop(columns=['Percent', 'Major Code'])
    top['Rank'] = range(1, len(top) + 1)


    # Function to escape ampersands in text
    def escape_ampersands(text):
        return text.replace('&', '\\&')


    top['Description'] = top['Description'].apply(escape_ampersands)
    top['Subtopics'] = top['Subtopics'].apply(escape_ampersands)

    latex_output = top[['Rank', 'Topic', 'Description', 'Subtopics']].to_latex(
        "../tables/gallup_top_topics.tex",
        index=False,
        label="gallup_top_topics",
        caption="We included the 10 Comparative Agendas Project (CAP) domains that had the highest average importance from 2013-2023, according to a dataset that coded Gallup Most Important Problem questions into CAP domains. CAP breaks major topics into subtopics. 'Description' comes from the 'general' subtopic.",
        column_format='rlp{4cm}p{7cm}',
        escape=False
    )

    with open("../tables/gallup_top_topics.tex", 'r') as file:
        latex_content = file.read()
    latex_content = latex_content.replace('\\begin{tabular}', '\\begin{tabular}{rlp{4cm}p{7cm}}')
    with open("../tables/gallup_top_topics.tex", 'w') as file:
        file.write(latex_content)