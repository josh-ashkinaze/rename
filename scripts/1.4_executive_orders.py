"""
Author: Joshua Ashkinaze

Description: Makes a table of executive orders by topic from trump first week

Date: 2025-01-31 16:18:13

Inputs:
    - data/raw/executive_orders.csv: CSV file with executive orders and topic annotated by npr

Outputs:
    - data/clean/executive_orders_by_topic.csv: CSV file with executive orders by topic
    - tables/executive_orders_by_topic.tex: LaTeX table of executive orders by topic
"""
import pandas as pd
import logging
import os

logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)

if __name__ == "__main__":

    df = pd.read_csv("../data/raw/executive_orders.csv")
    science_cols = ["Environment", "Health", "Disaster response"]
    df['topic2'] = df['Topic'].apply(lambda x: "Science" if x in science_cols else x)
    df2 = df['topic2'].value_counts().reset_index()
    df2.columns = ['Topic', 'Count']
    df2.to_csv("../data/clean/executive_orders_by_topic.csv", index=False)
    logging.info(str(df2))
    df2.to_latex("../tables/executive_orders_by_topic.tex",
                 index=False,
                 caption="Count of executive orders from Donald Trump's first week by topic.", label="executive_orders")
