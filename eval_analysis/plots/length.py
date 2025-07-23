import matplotlib.pyplot as plt
import seaborn as sns
import os

seq_labels_sorted = ['4', '8', '16', '32', '64', '128', '256']
f1_values_sorted = [98.62480074844548, 99.66299432400399, 99.22969329142292, 97.7982626623034, 95.23683795863502, 84.99724878550478, 66.57035469270886]

plt.figure(figsize=(6, 4))
sns.set_theme(style="white", rc={"axes.grid": False})
sns.set_style('white')
bars = sns.barplot(x=seq_labels_sorted, y=f1_values_sorted, color=sns.color_palette("Set2")[2])
sns.despine(top=True, right=True)
# Annotation loop can remain mostly unchanged:
for bar, score in zip(bars.patches, f1_values_sorted):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        f"{score:.1f}",
        ha="center",
        va="bottom",
        fontsize=12
    )

plt.xlabel("Sequence Length", fontsize=12)
plt.ylabel("ROUGE F1", fontsize=12)
plt.ylim(0, 110)  # Adjust if needed
#plt.grid(axis="y", linestyle=":", alpha=0.7)
plt.tight_layout()


directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory,  "length.pdf")
plt.savefig(path, dpi=200)
plt.show()
