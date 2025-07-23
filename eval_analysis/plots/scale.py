import matplotlib.pyplot as plt
import seaborn as sns
import os

data = {4: [(130, 0.9862480074844547), (370, 0.9864885847330667), (790, 0.986466927181514), (1000, 0.9861258194099917)], 
        8: [(130, 0.9966299432400398), (370, 0.9973112025843913), (790, 0.997340960497099), (1000, 0.9971100374495926)], 
        16: [(130, 0.9922969329142292), (370, 0.9955270571798874), (790, 0.996897156606976), (1000, 0.994023615918523)], 
        32: [(130, 0.977982626623034), (370, 0.9867635611033821), (790, 0.979463045059355), (1000, 0.9828613252562154)], 
        64: [(130, 0.9523683795863502), (370, 0.9722358557464316), (790, 0.9757652884261957), (1000, 0.967137376087415)], 
        128: [(130, 0.8499724878550479), (370, 0.910480226315231), (790, 0.9562131444468447), (1000, 0.9534399152778771)], 
        256: [(130, 0.6657035469270886), (370, 0.7787970159536167), (790, 0.8738561062483873), (1000, 0.8593911660059561)]}

plt.figure(figsize=(6, 4))
sns.set_theme(style="white", rc={"axes.grid": False})
sns.set_style('white')
sns.despine(top=True, right=True)

categorical_colors = sns.color_palette("Set2", n_colors=len(data))

for i, (chunk_len, points) in enumerate(data.items()):
    points_sorted = sorted(points, key=lambda x: x[0])
    xs = [p[0] for p in points_sorted]
    ys = [p[1] * 100 for p in points_sorted]
    label_str = f"Seq len {chunk_len}"
    sns.lineplot(x=xs, y=ys, marker='o', label=label_str, color=categorical_colors[i])
    plt.xscale('log')

sns.despine(top=True, right=True)
plt.xlabel("# Parameters (log scale)")
plt.ylabel("ROUGE F1")

plt.legend()
plt.tight_layout()

directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory,  "scale.pdf")
plt.savefig(path, dpi=200)
plt.show()
