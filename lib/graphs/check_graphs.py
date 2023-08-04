import json
import os
import os.path as osp
import sys
from multiprocessing import Process

import torch
from generate_unstructured_bipartie_graphs import process as generate_unstructured_graphs
from get_otherscores import main as generate_scores
from get_ramanujanscores_per_timestep import process as generate_ram_score
from utils import link_latest


def check_file(graph_path):
    """check whether the file exist and readable

    :tgt: TODO
    :returns: TODO

    """
    try:
        graph = torch.load(graph_path)
        print(f"\t {graph_path} is available")
        return True
    except:
        print(f"\t {graph_path} is not available")
        return False


def check_ram(graph_path):
    """check ram score

    :tgt: TODO
    :returns: TODO

    """
    graph = torch.load(graph_path)

    for name, info in graph.items():
        if 'ram_scores' not in info:
            print(f"\t\t missing ram score in {graph_path}")
            return False
        else:
            return True


def check_others(graph_path):
    """check for other scores

    :graph_path: TODO
    :returns: TODO

    """
    graphs = torch.load(graph_path)
    for name, info in graphs.items():
        if 'fc' in name: continue
        if 'overlap_coefs' not in info:
            return False
    return True


def check(file, dst, seed, model, num_classes):

    tgt = file.replace("model.pth", "graph.pth")
    if not check_file(tgt):
        generate_unstructured_graphs(file, model, num_classes, tgt, seed)

    if not check_ram(tgt):
        print("generate ram score")
        generate_ram_score(tgt)

    if not check_others(tgt):
        # print("generate other score")
        generate_scores(tgt)


def process(path):
    prunes = "SNIP GraSP SynFlow ERK Rand iterSNIP lth"  #PHEW"
    prunes = prunes.split(' ')
    # os.makedirs(dst, exist_ok=True)
    ##
    for p in prunes:
        if 'csv' in p:
            continue
        if not osp.isdir(osp.join(path, p)):
            print(f"missing {p}")
            continue
        link_latest(osp.join(path, p))
        seeds = os.listdir(osp.join(path, p))
        for s in seeds:
            file = osp.join(path, p, str(s), 'latest', 'model.pth')

            with open(osp.join(path, p, str(s), 'latest', 'config.txt'),
                      'r') as handle:
                config = json.load(handle)

            dataset = config['data']
            model = config['model']
            sparsity = None

            dataset = config.get('dataset', config['data'])
            if dataset == 'cifar10':
                num_classes = 10
            elif dataset == 'cifar100':
                num_classes = 100

            check(file, osp.join(path, p, str(s), 'latest'), int(s), model,
                  num_classes)


if __name__ == "__main__":
    arg = sys.argv
    multiprocess = 1
    result_dir = arg[1]  # your top-level result dir

    all_paths = []
    for density in os.listdir(result_dir):
        for dataset in os.listdir(osp.join(result_dir, density)):
            for model in os.listdir(osp.join(result_dir, density, dataset)):
                path = osp.join(result_dir, density, dataset, model)
                print(f'found {path}')
                all_paths.append(path)
    for i in range(0, len(all_paths), multiprocess):
        jobs = []
        for j in range(min(multiprocess, len(all_paths) - i)):
            print(f"working on {all_paths[i+j]}")
            # process(all_paths[i + j])
            job = Process(target=process, args=(all_paths[i + j], ))
            job.start()
            jobs.append(job)

        for j in jobs:
            j.join()
