import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


f1_aave_130m = [99.60809868460998, 99.46776833881368, 98.94276427661896, 96.89204300498547, 94.66439673305983, 82.2479566149738, 59.73628687418131]
f1_sae_130m = [99.7163767788183, 99.65006623131848, 99.33085435652397, 97.74344499847327, 95.82591941018983, 83.90555933573226, 65.27160837039116]
f1_aave_1b = [99.81973611840296, 99.67262309397772, 99.8717452692232, 99.55915353821898, 98.76204148235789, 97.9669569589624, 86.13592011143739]
f1_sae_1b = [99.93702777577892, 99.94645150773712, 99.93376235004044, 99.80662504580188, 99.45549621506056, 98.38283612896592, 89.50576633009611]
seq_labels = ['4', '8', '16', '32', '64', '128', '256']

sns.set_theme(style="white", rc={"axes.grid": False})
sns.set_style("white")
sns.despine(top=True, right=True)

# Use Seaborn Set2 palette
set2_colors = sns.color_palette("Set2")
color_list = set2_colors

x = np.arange(len(seq_labels))
width = 0.35

fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# 130m Model
bars_sae_130m = axs[0].bar(x, f1_sae_130m, width, label="SAE", color=color_list[0], alpha=0.8)
bars_aave_130m = axs[0].bar(x + width, f1_aave_130m, width, label="AAVE", color=color_list[1], alpha=0.8)

for bar in bars_sae_130m:
    height = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height:.1f}", ha="center", va="bottom", fontsize=10, rotation=90)

for bar in bars_aave_130m:
    height = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height:.1f}", ha="center", va="bottom", fontsize=10, rotation=90)

axs[0].set_title("130m Model", fontsize=14, pad=20)
axs[0].set_xlabel("Sequence Length", fontsize=12)
axs[0].set_ylabel("ROUGE-1 F1", fontsize=12)
axs[0].set_xticks(x + width / 2)
axs[0].set_xticklabels(seq_labels, fontsize=12)

# 1b Model Plot
bars_sae_1b = axs[1].bar(x, f1_sae_1b, width, color=color_list[0], alpha=0.8)
bars_aave_1b = axs[1].bar(x + width, f1_aave_1b, width, color=color_list[1], alpha=0.8)

for bar in bars_sae_1b:
    height = bar.get_height()
    axs[1].text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height:.1f}", ha="center", va="bottom", fontsize=10, rotation=90)

for bar in bars_aave_1b:
    height = bar.get_height()
    axs[1].text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height:.1f}", ha="center", va="bottom", fontsize=10, rotation=90)

axs[1].set_title("1b Model", fontsize=14, pad=20)
axs[1].set_xlabel("Sequence Length", fontsize=12)
axs[1].set_xticks(x + width / 2)
axs[1].set_xticklabels(seq_labels, fontsize=12)

handles = [
    plt.Line2D([0], [0], color=color_list[0], lw=10, label='SAE'),
    plt.Line2D([0], [0], color=color_list[1], lw=10, label='AAVE')
]
fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=12, frameon=False)

# Set y-axis limits
max_y = max_value = max(
    max(f1_aave_130m),
    max(f1_sae_130m),
    max(f1_aave_1b),
    max(f1_sae_1b)
)
margin = 10
axs[0].set_ylim(0, max_y + margin)
axs[1].set_ylim(0, max_y + margin)

sns.despine(top=True, right=True)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # leave space for title and legend

directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory,  "comparative_race.pdf")
plt.savefig(path, dpi=200)
plt.show()