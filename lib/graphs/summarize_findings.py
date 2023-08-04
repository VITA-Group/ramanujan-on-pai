import collections
import os
import os.path as osp
import statistics
import sys

import numpy as np
import pandas as pd
import torch


def summarize_density(path, global_density, model_type):
    """summarize the finding at for all density across all prune type and seeds

    :path: TODO
    :returns: return scores and std of this density for all prune types 

    """
    summarized_df = collections.defaultdict(list)
    for csv_file in os.listdir(path):
        if "graph" in csv_file:
            df = pd.read_csv(osp.join(path, csv_file))
            df['rm_norm'] = df['rm'] / df['rm_ub']
            df['sm_norm'] = df['sm'] / df['sm_ub']

            prune_types = df.prune_type.unique().tolist()
            metrics = df.columns.tolist()[2::]
            for _type in prune_types:
                subset_df = df.loc[df.prune_type == _type]
                summarized_df['prune_type'].append(_type)
                for m in metrics:
                    series = subset_df[m]
                    null_series = series.isnull()
                    non_null_series = series.notnull()
                    mean_series = series[non_null_series].mean()
                    std_series = series[non_null_series].std()
                    non_null_percentage = non_null_series.sum() / len(
                        non_null_series)
                    summarized_df[m].append(
                        (mean_series, std_series, non_null_series.sum(),
                         len(non_null_series)))
                # count
                rm_count = subset_df['rm'] > 0.0
                irm_count = subset_df['irm'] > 0.0
                rm_perc = rm_count.sum() / len(rm_count)
                irm_perc = irm_count.sum() / len(irm_count)
                summarized_df["perc_ram_by_rm"].append(rm_perc)
                summarized_df["perc_ram_by_irm"].append(irm_perc)

    summarized_df = pd.DataFrame.from_dict(summarized_df)
    results = collections.defaultdict(list)

    prune_type = summarized_df.prune_type.unique().tolist()
    metrics = summarized_df.columns.tolist()[1::]
    for _type in prune_type:
        subset_df = summarized_df.loc[summarized_df.prune_type == _type]
        results['model_type'].append(model_type)
        results['prune_type'].append(_type)
        results['global_density'].append(global_density)
        for m in metrics:
            if 'ub' in m: continue
            series = subset_df[m]
            if 'perc_ram' not in m:
                mean = [s[0] * s[2] for s in series]
                std = [s[1]**2 * s[2] for s in series]
                cnt = [s[2] for s in series]

                mean = sum(mean) / sum(cnt)
                std = np.sqrt(sum(std) / sum(cnt))
                cnt = sum(cnt) / len(cnt)
                cnt /= series.iat[0][-1]

                results[m + '_mean'].append(mean)
                results[m + '_std'].append(std)
                results[m + '_count'].append(cnt)
            else:
                results[m + '_mean'].append(series.mean())

    results = pd.DataFrame.from_dict(results)
    cnt = 0
    test_acc = collections.defaultdict(list)
    for csv_file in os.listdir(path):
        if "summary" in csv_file:
            cnt += 1
            df = pd.read_csv(osp.join(path, csv_file))
            for _type in prune_type:
                subset_df = df.loc[df.prune_type == _type].copy().reset_index()
                maxval_idx = subset_df.val_acc.idxmax()
                acc = subset_df.iloc[maxval_idx].test_acc
                test_acc[_type].append(acc)
    results_test = collections.defaultdict(list)
    for k, v in test_acc.items():
        results_test['acc'].append(sum(v) / len(v))
        results_test['acc_std'].append(statistics.stdev(v))
        results_test['prune_type'].append(k)
    results_test = pd.DataFrame.from_dict(results_test)
    results = results.sort_values(by=['prune_type'])
    results_test = results_test.sort_values(by=['prune_type'])

    results_test = results_test.drop(['prune_type'], axis=1)
    results = results.join(results_test)

    return results


if __name__ == "__main__":
    args = sys.argv

    assert len(args) == 3, "require: [result dir] [save_path]"
    # should be the top dir for our saved diretory
    # above all the density folders
    result_dir = args[1]
    save_path = args[2]

    density_dirs = []

    summaries = collections.defaultdict(list)

    df = None
    for density_ratio in os.listdir(result_dir):
        for dataset in os.listdir(osp.join(result_dir, density_ratio)):
            for model in os.listdir(
                    osp.join(result_dir, density_ratio, dataset)):
                density_dirs.append(
                    osp.join(result_dir, density_ratio, dataset, model, 'csv'))
                print(f"found {density_dirs[-1]}")
                global_density_ratio = float(density_ratio.split("_")[-1])
                _df = summarize_density(density_dirs[-1], global_density_ratio,
                                        model)
                if df is None:
                    df = _df
                else:
                    df = pd.concat([df, _df], ignore_index=True)  #df(_df)

    df.to_csv(save_path, index=False)
