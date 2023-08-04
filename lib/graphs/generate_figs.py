import collections
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

GRAPH = 'graph_seed-'
SUMMARY = 'summary-'


def get_csv(path, seed):
    global GRAPH
    global SUMMARY
    graphs = pd.read_csv(osp.join(path, GRAPH + seed + '.csv'))
    summary = pd.read_csv(osp.join(path, SUMMARY + seed + '.csv'))

    graphs['ram_crit'] = graphs[['rm', 'sm']].max(axis=1)
    graphs.loc[graphs.sparsity == 1.0, 'ram_crit'] = 1.0
    graphs.loc[graphs.sparsity == 1.0, 'sm'] = 1.0
    graphs.loc[graphs.sparsity == 1.0, 'rm'] = 1.0

    return graphs, summary


def savefig(dst, title):
    os.makedirs(dst, exist_ok=True)
    plt.savefig(osp.join(dst, title + '.png'))


def line_plot(m: pd.DataFrame,
              x: str,
              y: str,
              hue: str,
              dst: str = None,
              filter_by=None,
              title: str = None,
              ylim=(-1, 1)):
    # c = sns.color_palette("flare", as_cmap=True)\
    plt.figure(figsize=(12, 8))
    if filter_by:
        masks = None
        for cond in filter_by:
            mask = m.prune_type.str.contains(cond)
            if masks is None: masks = mask
            else:
                masks |= mask
        m = m[masks]
    g = sns.lineplot(
        data=m,
        x=x,
        y=y,
        hue=hue,
    )
    plt.xticks(rotation=90)
    g.relim()
    g.autoscale_view()
    if title:
        g.set(title=title)
    if dst:
        savefig(dst, title)


def bar_plot(m: pd.DataFrame,
             x: str,
             y: str,
             dst: str = None,
             filter_by=None,
             title: str = None,
             ylim=(-1, 1),
             op='max'):
    # c = sns.color_palette("flare", as_cmap=True)\
    plt.figure(figsize=(12, 8))
    if filter_by:
        masks = None
        for cond in filter_by:
            mask = m.prune_type.str.contains(cond)
            if masks is None: masks = mask
            else:
                masks |= mask
        m = m[masks]
    df = collections.defaultdict(list)
    for cat in m[x].unique():
        df[x].append(cat)
        if op == 'max':
            if y == 'test_acc':
                temp = m.loc[m[x] == cat].copy()
                temp = temp.reset_index()
                idxmax = temp.val_acc.idxmax()
                df[y].append(temp.iloc[idxmax].test_acc)
            else:
                df[y].append(m.loc[m[x] == cat][y].max())
        elif op == 'min':
            df[y].append(m.loc[m[x] == cat][y].min())
        elif op == 'mean':
            df[y].append(m.loc[m[x] == cat][y].mean())
        else:
            raise NotImplementedError

    df = pd.DataFrame.from_dict(df)
    df = df.sort_values(by=[y])
    g = sns.barplot(data=df, x=x, y=y)
    for i in g.containers:
        g.bar_label(i, )
    if title:
        g.set(title=title)
    plt.ylim(ylim)
    if dst:
        savefig(dst, title)


if __name__ == "__main__":
    args = sys.argv
    csv_dir = args[1]
    which_seed = args[2]
    dst_dir = args[3]

    csv_subfolders = csv_dir.split('/')[-5::]
    csv_subfolders.pop(-1)
    # csv_subfolders = "/".join(csv_subfolders)
    dst = osp.join(dst_dir, *csv_subfolders)
    os.makedirs(dst, exist_ok=True)

    graph_df, summary_df = get_csv(csv_dir, which_seed)
    variables = list(graph_df.keys())
    variables.remove("layer")
    variables.remove('prune_type')
    ylim = [(-1., 1), (-1, 1), (-1, 1), (0, 5), (0.0, 1.1), (-1., 150),
            (-1, 10)]
    for v in variables:
        line_plot(graph_df,
                  x='layer',
                  y=v,
                  hue='prune_type',
                  title=f'{csv_subfolders[0]}-{v}',
                  dst=dst)

    bar_plot(graph_df,
             x='prune_type',
             y='ram_crit',
             title=f'{csv_subfolders[0]}-ram_crit-mean',
             op="mean",
             dst=dst)

    variables = ['test_acc']
    bar_plot(summary_df,
             x='prune_type',
             y='test_acc',
             title=f'{csv_subfolders[0]}-acc',
             op="max",
             ylim=(0, 1.0),
             dst=dst)
