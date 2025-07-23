import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="white", rc={"axes.grid": False})
sns.set_style("white")
sns.despine(top=True, right=True)

# Use Seaborn's Set2 palette and get a color
palette = sns.color_palette("Set2")
bar_color = palette[1]

means_1b = pd.Series({
    'Misc': 6.8,
    'Not NE': 7.0,
    'Location': 7.1,
    'Person': 9.4,
    'Organization': 11.3
})

means_130m = pd.Series({
    'Misc': 30.9,
    'Person': 32.6,
    'Location': 33.2,
    'Not NE': 36.2,
    'Organization': 40.4
})

# Sort each Series individually in ascending order
means_130m_sorted = means_130m.sort_values()
means_1b_sorted = means_1b.sort_values()

# Set up Seaborn style
sns.set_theme(style="white")
sns.set_style("white")
sns.despine(top=True, right=True)
palette = sns.color_palette("Set2")

fig, axs = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

# Plot 130M
bars1 = axs[0].bar(means_130m_sorted.index, means_130m_sorted.values, color=palette[0])
axs[0].set_title("130M Model", fontsize=14)
axs[0].set_xlabel("Named Entity", fontsize=12)
axs[0].set_ylabel("Omission Rate", fontsize=12)
axs[0].tick_params(labelsize=12)
axs[0].set_xticklabels(means_130m_sorted.index, rotation=45)

# Annotate
for bar in bars1:
    height = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}", 
                ha='center', fontsize=10)

# Plot 1B
bars2 = axs[1].bar(means_1b_sorted.index, means_1b_sorted.values, color=palette[1])
axs[1].set_title("1B Model", fontsize=14)
axs[1].set_xlabel("Named Entity", fontsize=12)
axs[1].tick_params(labelsize=12)
axs[1].set_xticklabels(means_1b_sorted.index, rotation=45)

# Annotate
for bar in bars2:
    height = bar.get_height()
    axs[1].text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}", 
                ha='center', fontsize=10)

# Set consistent y-limit across plots
max_y = max(means_130m_sorted.max(), means_1b_sorted.max())
axs[0].set_ylim(0, max_y + 5)
axs[1].set_ylim(0, max_y + 5)

plt.tight_layout()
directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory,  "comparative_ner.pdf")
plt.savefig(path)