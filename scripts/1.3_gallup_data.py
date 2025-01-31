"""
Author: Joshua Ashkinaze

Description: Finds most polarizing instiutions from Gallup data

Inputs:
    - data/raw/gallup_data/gallup_inst.csv: Gallup data on institution trust
Outputs:
    - data/clean/gallup_data.csv: Cleaned Gallup data
    - tables/gallup_table.tex: LaTeX table of Gallup data

Date: 2025-01-31 14:49:06
"""


import pandas as pd
import logging


def main():

    pol_cols = ['The presidency', 'The U.S. Supreme Court', 'Congress'] # delete explicitly political institutions

    df = pd.read_csv("../data/raw/gallup_data/gallup_inst.csv")
    df = df[~df['Issue'].isin(pol_cols)]
    df['Republicans - Democrats'] = df['Republicans'] - df['Democrats']
    df['Absolute Difference'] = abs(df['Republicans - Democrats'])
    df['Absolute Difference'] = df['Absolute Difference']*100
    df = df.sort_values('Absolute Difference', ascending=False)
    df.to_csv("../data/clean/gallup_data.csv", index=False)
    logging.info(str(df))
    df[['Issue', 'Republicans - Democrats']].to_latex("../tables/gallup_table.tex",
                                                      index=False,
                                                      caption="Gallup Poll Results",
                                                      label="gallup")


if __name__ == "__main__":
    main()
