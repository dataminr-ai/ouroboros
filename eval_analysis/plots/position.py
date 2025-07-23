import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.gridspec as gridspec

sns.set_theme(style="white", rc={"axes.grid": False})
sns.set_style("white")

fig = plt.figure(figsize=(12, 3))
outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
top_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_gs[0, 0])
bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1, 0])

ax1 = fig.add_subplot(top_gs[0, 0])
ax2 = fig.add_subplot(top_gs[0, 1])
ax3 = fig.add_subplot(top_gs[0, 2])
ax4 = fig.add_subplot(top_gs[0, 3])
ax5 = fig.add_subplot(bottom_gs[0, 0])
ax6 = fig.add_subplot(bottom_gs[0, 1])
ax7 = fig.add_subplot(bottom_gs[0, 2])
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

bar_color = sns.color_palette("Set2")[0]

chunks =[4,8,16,32,64,128,256]

for i, chunk in enumerate(chunks):
    bin_edges_file = f'eval_analysis/plots/positions/{chunk}_bin_edges.txt'
    with open(bin_edges_file, "r") as f:
        bin_edges = [float(line.strip()) for line in f]

    # Read err_pos from the file
    err_pos_file = f'eval_analysis/plots/positions/{chunk}_err_position.txt'
    with open(err_pos_file, "r") as f:
        err_pos = [float(line.strip()) for line in f] 

    axes[i].hist(
        err_pos,
        bins=bin_edges,
        color=bar_color,
        alpha=0.7,
        rwidth=0.9,       # Controls width-to-space ratio
        align='mid',      # Ensures bars are centered on each bin
        edgecolor='white' # Helps highlight bar boundaries
    )
    axes[i].set_title(f'Sequence Length {chunk}')
    axes[i].set_xlabel('Position')
    axes[i].set_ylabel('Omit Count')

# Increase space between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.4)
sns.despine(top=True, right=True)
plt.tight_layout()

directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory,  "position.pdf")
plt.savefig(path, dpi=200)
plt.show()
