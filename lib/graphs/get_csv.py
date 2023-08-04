import collections
import os
import os.path as osp
import sys
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
import torch
import torch_geometric as pyg
from scores import pair_layers
from scores import ramanujan_score
from tqdm import tqdm


def save(item: pd.DataFrame, dst: Optional[str], name: str):
    """
    save pandas df to csv file under name
    """
    if dst is not None:
        os.makedirs(dst, exist_ok=True)
        item.to_csv(osp.join(dst, name + '.csv'), index=False)


def get_summary_view(layers: pd.DataFrame) -> pd.DataFrame:
    """
    summarize the overall score of the entire masks by avg the values of each layer
    """
    summary = collections.defaultdict(list)
    for mask in layers.mask_no.unique():
        tab = layers.loc[layers.mask_no == mask]
        tab = tab.drop(columns=['layer'])
        summary['mask_no'].append(mask)
        for k in tab.keys():
            if k == 'mask_no':
                continue
            summary[k].append(tab[k].mean())

    return pd.DataFrame.from_dict(summary)


def convert_tensor(data: collections.defaultdict):
    for k, v in data.items():
        _v = []
        for i in v:
            if isinstance(i, torch.Tensor):
                _v.append(i.tolist())
            else:
                _v.append(i)

        data[k] = _v
    return data


def find_related_pair(layer: str, pairs: List[Tuple]):
    """given name of a layer, finds its corresponding pairs

    :layer: TODO
    :pairs: TODO
    :returns: TODO

    """
    ret = []
    for p in pairs:
        if p[0][0] == layer and 'downsample' != p[-1][0]:
            return p
    return None


def generate_graph_csv(files: list, write=False) -> pd.DataFrame:
    """
    generate pandas Dataframe structure.
    params:
        TODO
    """
    global output_dir
    layers_df = collections.defaultdict(list)
    for file in files:

        density, dataset, model, prune_type, seed, directory, _ = file.split(
            '/')[-7::]
        graphs = torch.load(file)
        pairs = pair_layers(list(graphs.items()))
        for i, (layer, info) in enumerate(graphs.items()):
            s_m, sm_ub, r_m, rm_ub, t1m, t2m, ep, random_factor = info[
                'ram_scores']
            imsg = info['imsg']
            if len(imsg) != 7:
                imsg = [None] * 7
            i_sm, i_rm, ism_norm, irm_norm, num_sub_regulars, i_max_ep, i_mean_ep = imsg

            layers_df['prune_type'].append(prune_type)
            layers_df['layer'].append(layer)
            layers_df['sparsity'].append(info['sparsity'])
            layers_df['sm'].append(s_m)
            layers_df['sm_ub'].append(sm_ub)
            layers_df['rm'].append(r_m)
            layers_df['rm_ub'].append(rm_ub)
            layers_df['ism'].append(i_sm)
            layers_df['irm'].append(i_rm)
            layers_df['ism_norm'].append(ism_norm)
            layers_df['irm_norm'].append(irm_norm)
            layers_df['t1m'].append(t1m)
            layers_df['t2m'].append(t2m)
            layers_df['ep'].append(ep)
            layers_df['imax_ep'].append(i_max_ep)
            layers_df['i_mean_ep'].append(i_mean_ep)
            layers_df['num_sub_regulars'].append(i_mean_ep)
            layers_df['random_factor'].append(random_factor)

            related_pairs = find_related_pair(layer, pairs)
            if related_pairs:
                nxt = related_pairs[1][0]
                layers_df['copeland_score'].append(
                    info.get(f"{nxt}_copeland_score", 0))
                layers_df['compatibility'].append(
                    info.get(f"{nxt}_comatibility", 0))
                layers_df['overlap_coefs'].append(info.get('overlap_coefs', 0))
            else:
                layers_df['copeland_score'].append(0)
                layers_df['compatibility'].append(0)
                layers_df['overlap_coefs'].append(0)
    layers_df = convert_tensor(layers_df)
    density, dataset, model, prune_type, seed, directory, _ = file.split(
        '/')[-7::]
    # TODO check this please
    folder = file.split('/')[0:-7]
    folder = '/'.join(folder)
    if output_dir is not None:
        folder = output_dir
    name = f"graph_seed-{seed}"
    layers_df = pd.DataFrame.from_dict(layers_df)
    if write:
        save(layers_df, osp.join(folder, density, dataset, model, 'csv'), name)
    return layers_df


def generate_perf_csv(files: list, write=False) -> pd.DataFrame:
    """
    generate pandas Dataframe structure.
    params:
        TODO
    """
    global output_dir
    summary = collections.defaultdict(list)
    for file in files:
        density, dataset, model, prune_type, seed, directory, _ = file.split(
            '/')[-7::]
        df = pd.read_csv(file)
        df = df.sort_values(by=['epoch'])
        keys = df.keys()
        for k in keys:
            summary[f'{k}'].extend(df[k].tolist())
        summary['prune_type'].extend([prune_type] * len(df.index))

    df = convert_tensor(df)
    density, dataset, model, prune_type, seed, directory, _ = file.split(
        '/')[-7::]
    folder = file.split('/')[0:-7]
    folder = '/'.join(folder)
    if output_dir is not None:
        folder = output_dir
    summary = pd.DataFrame.from_dict(summary)
    name = f"summary-{seed}"
    if write:

        save(summary, osp.join(folder, density, dataset, model, 'csv'), name)
    return summary


def process(path):
    prune_type = os.listdir(path)
    graphs = collections.defaultdict(list)
    results = collections.defaultdict(list)
    for p in prune_type:
        if 'csv' in p:
            continue
        subfolder = osp.join(path, p)
        seeds = os.listdir(subfolder)
        for i, seed in enumerate(seeds):
            gp = osp.join(subfolder, seed, 'latest', 'graph.pth')
            rp = osp.join(subfolder, seed, 'latest', 'results.csv')
            if osp.isfile(gp):
                graphs[seed].append(
                    osp.join(subfolder, seed, 'latest', 'graph.pth'))
            else:
                print(f'missing {gp}')
            if osp.isfile(rp):
                results[seed].append(
                    osp.join(subfolder, seed, 'latest', 'results.csv'))
            else:
                print(f'missing {rp}')
    for seed in graphs.keys():
        generate_graph_csv(graphs[seed], True)
        generate_perf_csv(results[seed], True)


if __name__ == "__main__":
    arg = sys.argv
    multiprocess = 4
    result_dir = arg[1]  # your top-level result dir
    if len(arg) == 3:
        output_dir = arg[2]
    else:
        output_dir = None

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
            process(all_paths[i + j])
            # job = Process(target=process, args=(all_paths[i + j], ))
            # job.start()
            # jobs.append(job)

        # for j in jobs:
        # j.join()
