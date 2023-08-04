import os
import os.path as osp
import sys

import torch
from scores import iterative_mean_spectral_gap as imsg
from scores import ramanujan_score as ram
from tqdm import tqdm


def process(graph_path: str):
    """ for each graph get ram scores.
    """
    graphs = torch.load(graph_path)
    for (name, info) in tqdm(graphs.items(),
                             desc=graph_path,
                             total=len(graphs.keys())):
        info['ram_scores'] = ram(info)
        info['imsg'] = imsg(info)
    torch.save(graphs, graph_path)


if __name__ == "__main__":
    _, path = sys.argv
    files = os.listdir(path)
    files = [
        osp.join(path, f) for f in list(filter(lambda x: "mask" in x, files))
    ]

    for file in files:
        process(file)
