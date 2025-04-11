import seaborn as sns
import matplotlib.pyplot as plt

def grouped_barplots(data, x, target):
    g = sns.catplot(
    data=data, kind="count",
    x=x, hue=target,
        errorbar="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "Count")
    g.legend.set_title("")
    plt.show()