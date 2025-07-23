import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


f1_130m = {'pile/math': [99.8707715315758, 98.56898065482146, 92.89653378215874, 82.06532894260572, 74.6375927699838, 59.92207739798789, 41.63264603075177], 
'pile/arxiv': [98.70969063091965, 99.77117873996079, 99.57097272713649, 98.49384333691896, 96.27352905145983, 86.12373935491031, 70.26335097910294], 
'pile/free_law': [99.22783769501264, 99.6288521763791, 99.31513160784972, 97.64441491242391, 94.25763906858123, 81.16353590800911, 64.68701490849533], 
'pile/stack_exchange': [95.4984320152308, 98.33570838982097, 98.62544178353637, 97.54567032323334, 94.99011614479187, 84.98664202095829, 66.32271778275349], 
'pile/nih': [99.98073568171154, 99.95039955673187, 99.79396649411252, 98.82734988364923, 96.14848808147434, 85.56328424568139, 69.7077027433285], 
'pile/pubmed_central': [98.51170725465957, 99.54920049653887, 99.41917195103764, 98.36544830898578, 96.0690185864562, 85.16112251960956, 67.1138562402175], 
'pile/enron_email': [98.44975600474417, 99.57071844364735, 99.51404397283561, 98.23083212430474, 96.02801524510079, 85.80258867741239, 65.04356658034244], 
'pile/github': [93.09332105160414, 98.56271534575475, 99.10209461415504, 98.25997455741697, 96.47406811903177, 87.82309386959749, 69.28481439118632], 
'pile/cc': [99.75961239442867, 99.88758470401731, 99.6508441573775, 98.52764813077542, 96.30744630664319, 86.96519677672855, 69.72167879838855]}

f1_1b = {'pile/math': [99.53593949260772, 99.13264292325353, 91.99085187846697, 71.01846571982102, 42.71306896851712, 51.84362957404605, 36.70329355813639], 
'pile/arxiv': [98.71659199700858, 99.8266371578188, 99.81893305885211, 99.54497166298823, 98.76466781067684, 97.37656688602642, 88.46574968971002], 
'pile/free_law': [99.23622847797378, 99.71689654689149, 99.72272035156423, 99.2281243021535, 97.54564218047794, 94.92369070978228, 83.41605843258718], 
'pile/stack_exchange': [95.45100524054439, 98.37935013278096, 98.8485667822845, 99.03193518110037, 98.3828733153421, 97.31594977556026, 87.37930253059493], 
'pile/nih': [99.9856328667336, 99.98411839508911, 99.91758602051284, 99.86363393719816, 99.50870313970228, 98.38502198463077, 90.94997908398244], 
'pile/pubmed_central': [98.45311336115175, 99.69006259307618, 99.60185024743747, 99.22613188245337, 98.27111728260489, 96.71396373774083, 87.1376068026357], 
'pile/enron_email': [98.4757333806754, 99.64448255237774, 99.8283572378182, 99.6514230807802, 98.76052661968883, 97.5378446006415, 88.06988403078071], 
'pile/github': [93.11038998619551, 98.62445361781896, 99.34967424961877, 99.33986048278841, 98.20311868896985, 96.71249510546636, 86.83884464555469], 
'pile/cc': [99.7636099525246, 99.94405228319836, 99.88624581368178, 99.7319608608138, 99.34946878099609, 98.40681527451466, 91.70950856588955]}

eval_types = [
    "pile/cc",
    "pile/arxiv",
    "pile/free_law",
    "pile/stack_exchange",
    "pile/nih",
    "pile/pubmed_central",
    "pile/enron_email",
    "pile/github",
    "pile/math",
]
eval_labels = {
    "pile/cc": "Common Crawl",
    "pile/arxiv": "ArXiv",
    "pile/free_law": "Free Law",
    "pile/stack_exchange": "Stack Exchange",
    "pile/nih": "NIH",
    "pile/pubmed_central": "Pubmed Central",
    "pile/enron_email": "Enron Email",
    "pile/github": "Github",
    "pile/math": "DM Math",
}

seq_labels = ['4', '8', '16', '32', '64', '128', '256']

# Seaborn theme settings
sns.set_theme(style="white", rc={"axes.grid": False})
sns.set_style("white")
sns.despine(top=True, right=True)

# Use Seaborn Set2 palette and add an extra color
set2_colors = sns.color_palette("Set2")
extra_color = (0.6, 0.6, 0.8)
color_list = set2_colors + [extra_color]

x = np.arange(len(seq_labels))
group_width = 0.9  # wider bars

fig, axs = plt.subplots(1, 2, figsize=(18, 5), sharey=True)

# 130m Model 
total_bars_130m = len(eval_types)
width_130m = group_width / total_bars_130m

for i, et in enumerate(eval_types):
    label_text = eval_labels.get(et, et)
    bars = axs[0].bar(
        x + i * width_130m,
        f1_130m[et],
        width_130m,
        label=label_text,
        color=color_list[i % len(color_list)],
        alpha=0.8
    )
    for bar in bars:
        height = bar.get_height()
        axs[0].text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90
        )

axs[0].set_title("130m Model", fontsize=14, pad=20)
axs[0].set_xlabel("Sequence Length", fontsize=12)
axs[0].set_ylabel("ROUGE-1 F1", fontsize=12)
axs[0].set_xticks(x + (total_bars_130m - 1) * width_130m / 2)
axs[0].set_xticklabels(seq_labels, fontsize=12)
axs[0].grid(axis="y", linestyle=":", alpha=0.7)

# 1b Model
total_bars_1b = len(eval_types)
width_1b = group_width / total_bars_1b

for i, et in enumerate(eval_types):
    label_text = eval_labels.get(et, et)
    bars = axs[1].bar(
        x + i * width_1b,
        f1_1b[et],
        width_1b,
        color=color_list[i % len(color_list)],
        alpha=0.8
    )
    for bar in bars:
        height = bar.get_height()
        axs[1].text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90
        )

axs[1].set_title("1b Model", fontsize=14, pad=20)
axs[1].set_xlabel("Sequence Length", fontsize=12)
axs[1].set_xticks(x + (total_bars_1b - 1) * width_1b / 2)
axs[1].set_xticklabels(seq_labels, fontsize=12)
axs[1].grid(axis="y", linestyle=":", alpha=0.7)

# Legend
legend_handles = [
    plt.Line2D([0], [0], color=color_list[i % len(color_list)], lw=10, label=eval_labels.get(et, et))
    for i, et in enumerate(eval_types)
]
fig.legend(handles=legend_handles, loc='lower center', ncol=len(eval_types), fontsize=12, frameon=False)

# Set y-limit
max_height = max(
    max(max(v) for v in f1_130m.values() if v),
    max(max(v) for v in f1_1b.values() if v)
)
plt.ylim(0, max_height + 10)

sns.despine(top=True, right=True)
plt.tight_layout(rect=[0, 0.07, 1, 0.95])

directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory,  "comparative_distributions.pdf")
plt.savefig(path, dpi=200)
plt.show()