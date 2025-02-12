"""
Author: Joshua Ashkinaze

Description: Makes a CSV to use for Qualtrics loop and merge. This CSV will have columns

[loop_merge_idx, name, definition, new_name]
where for the non-political topics, the name and new_name will be the same.

Inputs:
- data/raw/original_definitions.csv: A CSV file with topics and definitions.
- data/raw/depolarized_names_raw: Depolarized names for political topics.

Outputs:
- data/clean/qualtrics_loop_merge_file.csv

Date: 2024-10-24 10:23:01
"""

import pandas as pd

def main():
    original_definitions = pd.read_csv("../data/clean/original_definitions.csv")
    depolarized_names = pd.read_csv("../data/raw/depolarized_names/depolarized_names_raw2024-10-24__10:34:39.csv")
    print(original_definitions.columns)
    print(depolarized_names.columns)
    data = []
    for idx, row in original_definitions.iterrows():
        topic = row["topic"]
        if row['is_political'] == 1:
            data_pt = {
                "name": row["topic"],
                "definition": row["grammar_fix_definition"],
                "new_name": depolarized_names[depolarized_names["original_name"] == topic]["new_name"].values[0]
            }
        else:
            data_pt = {
                "name": row["topic"],
                "definition": row["grammar_fix_definition"],
                "new_name": row["grammar_fix_definition"]
            }
        data.append(data_pt)
    df = pd.DataFrame(data)
    df['definition'] = df['definition'].str.replace('"', '').str.replace("'", '')
    df['loop_merge_idx'] = [i+1 for i in range(df.shape[0])]
    df.to_csv("../data/clean/qualtrics_loop_merge_file.csv", index=False)



if __name__ == "__main__":
    main()
