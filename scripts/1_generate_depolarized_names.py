"""
Author: Joshua Ashkinaze


This script generates depolarized names for political topics.

Process:
- Have a series of random ANES personas brainstorm critiques of a phrase then a Moderator come up with new name.

Input:
- data/clean/original_definitions.csv: A CSV file with topics and definitions.

Outputs:
    - raw/depolarized_names: A folder with two things.
        The first thing is a series of json files with the output of the ensemble's deliberation. This is per topic.
        The second thing is a CSV file with the original topics, definitions, and the new definitions.

Date: 2024-10-24 10:21:53
"""

import pandas as pd
from plurals.agent import Agent
from plurals.deliberation import Graph, Debate, Ensemble, Moderator
from pathlib import Path
import logging

import json
from dotenv import load_dotenv
from datetime import datetime
import os

logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO,
                    format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)

load_dotenv(os.path.join(os.path.dirname(__file__), "../src/.env"))

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

dt = datetime.now().strftime("%Y-%m-%d__%H:%M:%S")


def critique_task(phrase):
    return f"""Some commentators say that the phrase  <start>{phrase}</end> is polarizing.
  Offer specific objections from your perspective. Be very specific. You are in a focus group."""


def revise_to_address_concerns(phrase, defintion):
    return f"""
  {phrase} is defined as <start>{defintion}</end>. Given previous critiques, return
  a FinalName for the same concept that will address the critiques, while adhering to the definition.

  Follow the following format.

  Step1: The main points of previous critiques are...
  Step2: Given the main points of previous critiques, and avoiding all polarizing connotations, the new name should be...
  Step3: FinalName
  """


def extract_name_only(final_response):
    s = f"""
    Given a rationale from an Agent for renaming a concept, return the final name that the Agent came up with and NOTHING ELSE. 
    Do not return 'FinalNameOnly', just the actual name. 
    
    Follow the following format:
    
    Rationale: Rationale
    FinalNameOnly: FinalName
    
    Rationale: {final_response}
    FinalNameOnly:
    """
    return s


def run_plurals(phrase, definition, n_agents=5, agent_llm="claude-3-5-sonnet-20240620",
                mod_llm="claude-3-5-sonnet-20240620"):
    """
    Runs a moderated ensemble of nationally representative agents to depolarize a phrase.
    """

    agent_task = critique_task(phrase)
    mod_task = revise_to_address_concerns(phrase, definition)

    agents = [
        Agent(
            model=agent_llm,
            task=agent_task,
            persona="random",
        )
        for i in range(n_agents)
    ]

    moderator = Moderator(
        model=mod_llm,
        combination_instructions="Here are the previous critiques: <start>${previous_responses}</end>",
        task=mod_task,
        system_instructions=None
    )
    ensemble = Ensemble(agents=agents, moderator=moderator)
    ensemble.process()

    final_name = Agent(task=extract_name_only(ensemble.final_response), model=agent_llm).process()

    fn = f"{phrase.replace(' ', '')}_{n_agents}_AgentLLM_{agent_llm}__ModLLM_{mod_llm}{dt}"

    Path("../data/raw/depolarized_names").mkdir(parents=True, exist_ok=True)

    with open(f"../data/raw/depolarized_names/{fn}.json", "w") as f:
        json.dump(ensemble.info, f)

    return final_name


def main():
    df = pd.read_csv('../data/clean/original_definitions.csv')
    pol = df.query("is_political==1")
    logging.info(f"Number of political names: {pol.shape[0]}")
    data = []
    for i, row in pol.iterrows():
        phrase = row['topic']
        definition = row['grammar_fix_definition']
        new_name = run_plurals(phrase, definition)
        logging.info(f"{phrase} is done")
        data.append({
            'original_name': phrase,
            'definition': definition,
            'new_name': new_name
        })
    df = pd.DataFrame(data)
    df.to_csv(f"../data/raw/depolarized_names/depolarized_names_raw{dt}.csv", index=False)
    logging.info("All done")


if __name__ == "__main__":
    main()
