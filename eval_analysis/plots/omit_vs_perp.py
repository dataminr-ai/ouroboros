import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

fig, ax1 = plt.subplots(figsize=(4, 3))

directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory,  "omit_vs_perp.csv")
agg_df = pd.read_csv(path)
bin_size = 10

fig, ax1 = plt.subplots(figsize=(5, 3))
palette = sns.color_palette("Set2")

# Plot bars first on ax1
bars = ax1.bar(
    agg_df['ref_bin'],
    agg_df['count'],
    width=4.5,
    color='lightgrey',
    label='Number of Records'
)

# Create a secondary axis sharing the same x-axis
ax2 = ax1.twinx()

# Plot line on ax2 — this is now drawn *after* bars, so it appears on top
sns.lineplot(
    x='ref_bin',
    y='avg_omit_rate',
    data=agg_df,
    ax=ax2,
    color=palette[0],
    linewidth=1
)

# Labels
ax1.set_xlabel('Reference Perplexity (binned)')
ax2.set_ylabel('Average Omit Rate', color=palette[0])
ax2.tick_params(axis='y', labelcolor=palette[0])

ax1.set_ylabel('Number of Records', color='grey')
ax1.tick_params(axis='y', labelcolor='grey')

# No legend, clean layout
fig.tight_layout()
path = os.path.join(directory,  "omission_perplexity.pdf")
plt.savefig(path, dpi=200)
plt.show()