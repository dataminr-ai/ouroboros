import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

sns.set_theme(style="white", rc={"axes.grid": False})
sns.set_style("white")
sns.despine(top=True, right=True)

# Use Seaborn's Set2 palette and get a color
palette = sns.color_palette("Set2")
bar_color = palette[1]

means_130m = pd.Series({
    'PUNCT': 26.5, 'DET': 30.5, 'PRON': 32.5, 'ADP': 32.9, 'ADJ': 33.3,
    'NOUN': 34.0, 'VERB': 34.6, 'ADV': 34.8, 'INTJ': 40.0, 'PRT': 40.4,
    'SYM': 40.9, 'CONJ': 41.2, 'X': 47.6, 'NUM': 51.5
})

means_1b = pd.Series({
    'INTJ': 0.0, 'X': 4.8, 'DET': 4.9, 'ADJ': 5.9, 'VERB': 6.1,
    'ADP': 6.1, 'CONJ': 6.2, 'PRON': 6.7, 'SYM': 6.8, 'ADV': 7.1,
    'NOUN': 7.4, 'PRT': 7.8, 'PUNCT': 11.7, 'NUM': 12.6
})

# Sort each Series individually in ascending order
means_130m_sorted = means_130m.sort_values()
means_1b_sorted = means_1b.sort_values()

# Set up Seaborn style
sns.set_theme(style="white")
sns.set_style("white")
sns.despine(top=True, right=True)
palette = sns.color_palette("Set2")

fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# Plot 130M
bars1 = axs[0].bar(means_130m_sorted.index, means_130m_sorted.values, color=palette[0])
axs[0].set_title("130M Model", fontsize=14)
axs[0].set_xlabel("Part of Speech", fontsize=12)
axs[0].set_ylabel("Omission Rate", fontsize=12)
axs[0].tick_params(labelsize=12)
axs[0].set_xticklabels(means_130m_sorted.index, rotation=90)

# Annotate
for bar in bars1:
    height = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}", 
                ha='center', fontsize=10, rotation = 90)

# Plot 1B
bars2 = axs[1].bar(means_1b_sorted.index, means_1b_sorted.values, color=palette[1])
axs[1].set_title("1B Model", fontsize=14)
axs[1].set_xlabel("Part of Speech", fontsize=12)
axs[1].tick_params(labelsize=12)
axs[1].set_xticklabels(means_1b_sorted.index, rotation=90)

# Annotate
for bar in bars2:
    height = bar.get_height()
    axs[1].text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}", 
                ha='center', fontsize=10, rotation=90)

# Set y-limit
max_y = max(means_130m_sorted.max(), means_1b_sorted.max())
margin = 9 
axs[0].set_ylim(0, max_y + margin)
axs[1].set_ylim(0, max_y + margin)

plt.tight_layout()

directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory,  "comparative_pos.pdf")
plt.savefig(path)