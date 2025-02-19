"""
Author: Joshua Ashkinaze

Description: Makes some plots of mediacloud data.

Inputs:
    - data/clean/mediacloud_analysis{...}.csv: A CSV file with MediaCloud data for each topic

Outputs:
    - plots/mc_analysis_control_comparison.pdf: Bar plot comparing control and candidate terms on controversy ratio
    - plots/mc_analysis_composite_score.pdf: Bar plot of highest and lowest composite scores
    - plots/mc_analysis_controversy_rank.pdf: Bar plot of most and least controversial phrases
    - plots/mc_analysis_mention_rank.pdf: Bar plot of most and least mentioned phrases
    - data/clean/mediacloud_aggregated_{cutoff_date}_{end_date}.csv: A CSV file with aggregated MediaCloud data. This looks like the old format
    with cols ['phrase', 'total_mentions', 'controversy_mentions', 'is_dummy', 'controversy_ratio', 'mention_rank', 'controversy_rank', 'composite_score']

Logs:
    Some stats about comparing control and candidate terms on controversy ratio



Date: 2025-02-13 09:29:34
"""


import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import yaml

logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO,
                    format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)



with open("config.yaml", 'r') as stream: config = yaml.safe_load(stream)
datetime_pull = config['datetime_pull']

logging.info(f"Starting mc analysis for datetime pull: {datetime_pull}")

def make_aesthetic(hex_color_list=None,
                   with_gridlines=False,
                   bold_title=False,
                   save_transparent=False,
                   font_scale=2,
                   latex2arial=True
                   ):
    """Make Seaborn look clean and add space between title and plot"""

    # Note: To make some parts of title bold and others not bold, we have to use
    # latex rendering. This should work:
    # plt.title(r'$\mathbf{bolded\ title}$' + '\n' + 'And a non-bold subtitle')

    sns.set(style='white', context='paper', font_scale=font_scale)
    if not hex_color_list:
        # 2024-11-28: Reordered color list
        hex_color_list = [
            "#2C3531",  # Dark charcoal gray with green undertone
            "#D41876",  # Telemagenta
            "#00A896",  # Persian green
            "#826AED",  # Medium slate blue
            "#F45B69",  # Vibrant pinkish-red
            "#E3B505",  # Saffron
            "#89DAFF",  # Pale azure
            "#342E37",  # Dark grayish-purple
            "#7DCD85",  # Emerald
            "#F7B2AD",  # Melon
            "#D4B2D8",  # Pink lavender
            "#020887",  # Phthalo blue
            "#E87461",  # Medium-bright orange
            "#7E6551",  # Coyote
            "#F18805"  # Tangerine
        ]

    sns.set_palette(sns.color_palette(hex_color_list))

    # Enhanced typography settings
    plt.rcParams.update({
        # font settings
        'font.family': 'Arial',
        'font.weight': 'regular',
        'axes.labelsize': 11 * font_scale,
        'axes.titlesize': 14 * font_scale,
        'xtick.labelsize': 10 * font_scale,
        'ytick.labelsize': 10 * font_scale,
        'legend.fontsize': 10 * font_scale,

        # spines/grids
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 0.8,  # Thinner spines
        'axes.grid': with_gridlines,
        'grid.alpha': 0.2,
        'grid.linestyle': ':',
        'grid.linewidth': 0.5,

        # title
        'axes.titlelocation': 'left',
        'axes.titleweight': 'bold' if bold_title else 'regular',
        'axes.titlepad': 15 * (font_scale / 1),

        # fig
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.2,
        'figure.constrained_layout.w_pad': 0.2,

        # legend
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.facecolor': 'white',
        'legend.borderpad': 0.4,
        'legend.borderaxespad': 1.0,
        'legend.handlelength': 1.5,
        'legend.handleheight': 0.7,
        'legend.handletextpad': 0.5,

        # export
        'savefig.dpi': 300,
        'savefig.transparent': save_transparent,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
        'figure.autolayout': False,

        # do this for the bold hack
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold'

    })

    return hex_color_list


mypal = make_aesthetic(font_scale=3)


def get_extremes(df, col, n):
    """Get top and bottom n values for a given column."""
    top = df.sort_values(by=[col], ascending=True).head(n)
    print("Col", col, "Top")
    bottom = df.sort_values(by=[col], ascending=True).tail(n)
    both = pd.concat([top, bottom])
    both = both.sort_values(by=[col], ascending=False)
    both['mean'] = df[col].mean()
    return both


def plot_extremes(df, column, n, title=None):
    """
    Plot top and bottom n values for a given column with mean line.
    Returns the figure for saving.
    """
    top = df.nlargest(n, column)
    print("Col", column, "Top", top['phrase'].to_list())

    bottom = df.nsmallest(n, column)
    both = pd.concat([top, bottom])
    both = both.sort_values(by=column, ascending=True)
    mean_val = df[column].mean()

    fig = plt.figure(figsize=(18, 12))

    bars = plt.barh(range(len(both)), both[column])

    for i, bar in enumerate(bars):
        if i < n:  # Bottom n
            bar.set_color('#ff9999')
        else:  # Top n
            bar.set_color('#66b3ff')

    plt.axvline(x=mean_val, color='gray', linestyle='--', alpha=0.7)
    plt.text(mean_val, len(both), f'Mean: {mean_val:.3f}',
             verticalalignment='bottom', horizontalalignment='right')

    plt.yticks(range(len(both)), both['phrase'])
    if title:
        plt.title(title)
    plt.xlabel(column)

    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    return fig, top


def analyze_controversy_comparison(df, seed=42):
    """
    Analyze and plot control vs candidate terms comparison using permutation test.
    Returns the figure and statistical summary.
    """
    stats_df = df.groupby('control')['controversy_ratio'].agg(['mean', 'std', 'count']).round(3)
    candidate_data = df[df['control'] == 'Candidate Terms']['controversy_ratio']
    control_data = df[df['control'] == 'Control']['controversy_ratio']
    obs_diff = np.mean(candidate_data) - np.mean(control_data)
    np.random.seed(seed)

    def diff_in_means(x, y):
        return np.mean(x) - np.mean(y)

    def cohen_d(x, y):
        # from here
        # https: // stackoverflow.com / questions / 21532471 / how - to - calculate - cohens - d - in -python
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)


    perm_test = stats.permutation_test(
        (candidate_data, control_data),
        diff_in_means,
        n_resamples=10000,
        alternative='two-sided'
    )

    summary = f"""stats:
Candidate Terms (n={int(stats_df.loc['Candidate Terms', 'count'])}):
    M = {stats_df.loc['Candidate Terms', 'mean']:.2f}
    SD = {stats_df.loc['Candidate Terms', 'std']:.2f}

Control Terms (n={int(stats_df.loc['Control', 'count'])}):
    M = {stats_df.loc['Control', 'mean']:.2f}
    SD = {stats_df.loc['Control', 'std']:.2f}

Permutation Test (seed={seed}):
    Observed difference = {obs_diff:.2f}
    p-value = {perm_test.pvalue:.3e}
    
Effect Size:
    Cohen's d = {cohen_d(candidate_data, control_data):.2f}
    
    
    """

    fig, ax = plt.subplots(figsize=(18, 12))
    sns.barplot(data=df, x='control', y='controversy_ratio', ax=ax)
    plt.ylabel("Controversy Ratio")
    plt.xlabel("")

    return fig, summary


def save_plot(fig, name):
    """Save a plot."""
    filename = f"../plots/mc_analysis{config['datetime_pull']}_{name}.pdf"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def process_mediacloud_data(input_file, cutoff_date):
    """
    Process MediaCloud daily data to add aggregate columns.

    Args:
        input_file (str): Path to input CSV file
        cutoff_date (str): Cutoff date in YYYY-MM-DD format

    Returns:
        pd.DataFrame: Processed dataframe with aggregate metrics
    """
    logging.info(f"Reading data from {input_file}")
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])

    # 1. Filter based on cutoff date
    cutoff = pd.to_datetime(cutoff_date)
    df = df[df['date'] >= cutoff].copy()

    # 2. aggregate the mentions for each phrase
    agg_df = df.groupby('phrase').agg({
        'total_mentions': 'sum',
        'controversy_mentions': 'sum',
        'is_dummy': 'first'  # carry forward the dummy indicator
    }).reset_index()

    # 3. calc controversy ratio on aggregated data
    agg_df['controversy_ratio'] = (
        agg_df['controversy_mentions'] /
        agg_df['total_mentions'].where(agg_df['total_mentions'] > 0, 1)
    ).fillna(0)

    # 4. now calculate ranks on the aggregated data
    agg_df['mention_rank'] = agg_df['total_mentions'].rank(pct=True)
    agg_df['controversy_rank'] = agg_df['controversy_ratio'].rank(pct=True)

    # 5. sort by composite
    agg_df['composite_score'] = 0.5*agg_df['mention_rank'] + 0.5*agg_df['controversy_rank']
    agg_df = agg_df.sort_values('composite_score', ascending=False)

    logging.info(f"Processed data summary:")
    logging.info(f"Total unique phrases: {len(agg_df)}")
    logging.info(f"Date range used for aggregation: {df['date'].min().date()} to {df['date'].max().date()}")

    return agg_df

def main():
    # 1. Read in data
    ##################################################
    ##################################################
    daily_fn = "../data/clean/mediacloud_daily__2025-02-12__16:11:44_2021-01-01_2025-02-01.csv".replace("_2025-02-12__16:11:44", f"{datetime_pull}")
    end_date = daily_fn.split("_")[-1].replace(".csv", "")
    cutoff_date = '2021-01-01'
    aggd_fn = f"../data/clean/mediacloud_aggregated_{cutoff_date}_{end_date}.csv"

    logging.info(f"Reading data from {daily_fn}")
    logging.info(f"Using cutoff date: {cutoff_date}")

    df = pd.read_csv(daily_fn)
    df = process_mediacloud_data(daily_fn, cutoff_date) # so this will subset `df` to only the data after the cutoff date
    df.to_csv(aggd_fn, index=False)
    print(df.head(10))
    df['control'] = df['is_dummy'].replace({0: 'Candidate Terms', 1: 'Control'})

    # 2. Different plots
    ##################################################
    ##################################################
    control_fig, summary = analyze_controversy_comparison(df)
    save_plot(control_fig, "control_comparison")
    logging.info(summary)

    composite_fig, composites = plot_extremes(df, 'composite_score', 10, 'Highest and Lowest Composite Score')
    save_plot(composite_fig, "composite_score")

    controversy_fig, contros = plot_extremes(df, 'controversy_rank', 10, 'Most and Least Controversial Phrases')
    save_plot(controversy_fig, "controversy_rank")

    mention_fig, mentions = plot_extremes(df, 'mention_rank', 10, 'Most and Least Mentioned Phrases')
    save_plot(mention_fig, "mention_rank")

    # 3. Save things that are any extreme
    ##################################################
    ##################################################
    N_EXTREMES = 10

    get_top = lambda df, col, n: df.nlargest(n, col)

    # Get tops for each metric
    df_no_dummy = df[df['is_dummy'] == 0]
    composites = get_top(df_no_dummy, 'composite_score', N_EXTREMES)
    contros = get_top(df_no_dummy, 'controversy_rank', N_EXTREMES)
    mentions = get_top(df_no_dummy, 'mention_rank', N_EXTREMES)

    # Combine all tops
    any_extreme = pd.concat([composites, contros, mentions])
    any_extreme = any_extreme.drop_duplicates(subset='phrase')
    any_extreme = any_extreme.sort_values('composite_score', ascending=False)

    # Create new dataframe with required columns
    prompt = """CONSTRAINTS
    - Each phrase should be highly popular and has been talked about a lot
    - Each phrase should be currently polarizing in the sense that: (A) It provokes strong disagreement between political groups; (B) The concept can be described in alternative, less charged language
    - Each phrase should describe a discrete idea and not a general topic
    - Each phrase should be 1-4 words, with widely-known acronyms allowed 
    - Each phrase should originate from academic, policy, or institutional contexts. That is: We do not want memes or highly "online" terms
    - Each phrase should be the shortest and simplest phrase that describes the idea. For example, do not say "[thing] policies" when "[thing]", alone, is sufficient

    AVOID
    - Avoid phrases that include the name of a specific identity. For example, "transgender rights" or "gay marriage" is not allowed since these include "transgender" and "gay". But "critical race theory" is allowed since it does not name a specific race or identity. 
    - Avoid phrases related to atrocities (e.g: "genocide"), hate ideologies (e.g: "terrorism"), human rights violations (e.g.: "torture"), public health misinformation (e.g.: "covid denial"), or where renaming could constitute historical revisionism (e.g.: "slavery")"""

    output_df = pd.DataFrame({
        'prompt': prompt,
        'phrase': any_extreme['phrase'],
        'valid': ''  # Empty valid column
    })

    if not os.path.exists("../data/annot"):
        os.makedirs("../data/annot")

    logging.info(f"Saved {len(output_df)} phrases")
    print(output_df['phrase'].unique())

    annots = ["JA", "JM", "LK", "AP"]
    for annotator in annots:
        any_extreme_csv_fn = f"../data/annot/mediacloud_any_extreme_{cutoff_date}_{end_date}_{annotator}_raw.csv"
        output_df.to_csv(any_extreme_csv_fn, index=False)


if __name__ == "__main__":
    main()
