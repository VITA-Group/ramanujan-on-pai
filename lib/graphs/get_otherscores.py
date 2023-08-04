import os
import os.path as osp
import sys
from multiprocessing import Process

import torch
from scores import channel_overlap_coefs
from scores import compatibility_ratio
from scores import copeland_score
from scores import pair_layers
from tqdm import tqdm


def main(graph_path: str):
    """
    given path to a graph object, generate overlap coefs and copeland score for that graph

    :graph_path: TODO
    :returns: TODO

    """
    graph = torch.load(graph_path)
    pairs = pair_layers(list(graph.keys()))
    for i, (layer1, layer2) in enumerate(pairs):
        if f'{layer2}_copeland_score' in graph[layer1]:
            continue
        clscore = copeland_score(graph[layer1], graph[layer2])
        overlap_coefs = channel_overlap_coefs(graph[layer2],
                                              graph[layer1]['dim_out'])
        graph[layer2]['overlap_coefs'] = overlap_coefs

        if i == 0:
            overlap_coefs = channel_overlap_coefs(graph[layer1], 3)
            graph[layer1]['overlap_coefs'] = overlap_coefs

        graph[layer1][f"{layer2}_copeland_score"] = clscore
        graph[layer1][f"{layer2}_comatibility"] = compatibility_ratio(
            graph[layer1], graph[layer2])
    # print(f"done {graph_path}")
    torch.save(graph, graph_path)


if __name__ == "__main__":
    _, path = sys.argv
    files = os.listdir(path)

    for file in tqdm(os.listdir(path), desc=path, total=len(os.listdir(path))):
        full_path = osp.join(path, file)
        main(full_path)
