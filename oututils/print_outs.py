
from oututils import reporting, os_utils
from datautils import diag_datasets as datasets
from algorithms import alg_selector
from oututils.query import Q
import numpy as np
import pandas as pd
import os
import math
def group_and_sort(records, selection_method):
    grouped_records = reporting.get_grouped_records(records).map(lambda group:
                                                                 {**group, "sweep_acc": selection_method.sweep_acc(
                                                                     group["records"])}
                                                                 ).filter(lambda g: g["sweep_acc"] is not None)

    # read algorithm names and sort (predefined order given by alg_selector.ALGORITHMS)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = [n for n in alg_selector.ALGORITHMS if n in alg_names] + [n for n in alg_names if
                                                                          n not in alg_selector.ALGORITHMS]

    # read dataset names and sort (predefined order given by datasets.DATASETS)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names] + [d for d in dataset_names if
                                                                            d not in datasets.DATASETS]
    return grouped_records, alg_names, dataset_names

def print_hparams_grid_results(args, selection_method):
    records = reporting.load_records(args.input_dir)
    print("Total records:", len(records))
    records = reporting.get_grouped_records(records, args.sub_algorithm)

    if args.sub_algorithm is not None:
        file_name = args.dataset+'_'+args.algorithm + '_'+ args.sub_algorithm+'_Env'+str(args.test_env)
        records = records.filter(
            lambda r:
            r['dataset'] == args.dataset and
            r['algorithm'] == args.algorithm and
            r['sub_algorithm'] == args.sub_algorithm and
            r['test_env'] == args.test_env
        )
    else:
        file_name = args.dataset + '_' + args.algorithm  + '_Env' + str(args.test_env)
        records = records.filter(
            lambda r:
            r['dataset'] == args.dataset and
            r['algorithm'] == args.algorithm and
            r['test_env'] == args.test_env
        )
    print('\n')
    print(f'Model selection: {selection_method.name}')

    # get standard deviation calculation type from saved records
    avg_std = records[0]['records'][0]['args']['avg_std']

    if avg_std in ['e', 'experiments']:
        # hparams are determined by both hparams_seed and trial_seed.
        # take std and average based on repetitions of the entire experiment
        val_accs = []
        test_accs = []
        for group in records:
            best_hparams = selection_method.hparams_accs(group['records'])
            run_acc, _ = best_hparams[0] # the first one is the best based on sorted results
            val_accs.append(run_acc['val_acc'])
            test_accs.append(run_acc['test_acc'])
            r_len = len(best_hparams) if args.is_print_all else 1
            print('-'*31+f"trial_seed: {group['trial_seed']}")

            for i, (run_acc, hparam_records) in enumerate( best_hparams[:r_len]):
                print(f"The test_acc/hparams given by the val_acc ranked {i}")
                print(f"\t\t{run_acc}")
                print("\thparams:")
                for k, v in sorted(hparam_records[-1]['hparams'].items()):
                    print('\t\t{}: {}'.format(k, v))
                #-- debug: check whether the hparam_records are from the same run/trial
                # for r in hparam_records:
                #     a = r['hparams']
                #     print(f'\t\t{a}')
                    # b = hparam_records[0]['hparams']
                    # print(f'\t{b}')
                    # assert(r['hparams'] == hparam_records[0]['hparams'])
                print("\toutput_dirs:")
                output_dirs = hparam_records.select('args.output_dir').unique()
                for output_dir in output_dirs:
                    print(f"\t\t{output_dir}")
        val_mean, val_err, val_mean_err_str = os_utils.format_mean(val_accs)
        test_mean, test_err, test_mean_err_str = os_utils.format_mean(test_accs)
        print('-'*10+'Avg +/- std'+'-'*10)
        print('val_acc: ' + val_mean_err_str)
        print('test_acc: '+ test_mean_err_str)
        print('-' * 31 )

    elif avg_std in ['t', 'trials']:
        hparams_dict = {}
        # get hparams_grid_bases from saved records
        hparams_names = list(records[0]['records'][0]['hparams'].keys())

        os.makedirs(args.save_dir, exist_ok=True)
        file_name = os.path.join(args.save_dir, file_name)

        for group in records:
            best_hparams = selection_method.hparams_accs(group['records'])
            # let's stack the worse (based on val_acc) into the list first
            for run_acc, hparam_records in best_hparams:
                hp_loc = tuple( (key, hparam_records[-1]['hparams'][key]) for key in hparams_names )
                if hp_loc not in hparams_dict:
                    hparams_dict[hp_loc] = {'val_acc':[ run_acc['val_acc'] ], 'test_acc':[ run_acc['test_acc'] ]}
                else:
                    hparams_dict[hp_loc]['val_acc'].append(run_acc['val_acc'])
                    hparams_dict[hp_loc]['test_acc'].append(run_acc['test_acc'])

        # find the best hparams based on best average val_acc across trials
        print('-' * 31)
        hparams_val_acc_dict = {k:np.mean(hparams_dict[k]['val_acc']) for k in hparams_dict}
        best_val_acc_hp_kv = max(hparams_dict, key=lambda k: hparams_val_acc_dict[k])
        print('The best set of hparams based on val_acc: ')
        for k, v in best_val_acc_hp_kv:
            print('\t{}: {}'.format(k, v))
        val_mean, val_err, val_mean_err_str = os_utils.format_mean(hparams_dict[best_val_acc_hp_kv]['val_acc'])
        test_mean, test_err, test_mean_err_str = os_utils.format_mean(hparams_dict[best_val_acc_hp_kv]['test_acc'])
        print('-'*10+'Avg +/- std'+'-'*10)
        print('val_acc: ' + val_mean_err_str)
        print('test_acc: '+ test_mean_err_str)
        print('-' * 31 )

        print('-' * 31)
        hparams_test_acc_dict = {k: np.mean(hparams_dict[k]['test_acc']) for k in hparams_dict}
        best_test_acc_hp_kv = max(hparams_dict, key=lambda k: hparams_test_acc_dict[k])
        print('The best set of hparams based on test_acc (oracle): ')
        for k, v in best_test_acc_hp_kv:
            print('\t{}: {}'.format(k, v))
        val_mean, val_err, val_mean_err_str = os_utils.format_mean(hparams_dict[best_test_acc_hp_kv]['val_acc'])
        test_mean, test_err, test_mean_err_str = os_utils.format_mean(hparams_dict[best_test_acc_hp_kv]['test_acc'])
        print('-' * 10 + 'Avg +/- std' + '-' * 10)
        print('val_acc (oracle): ' + val_mean_err_str)
        print('test_acc (oracle): ' + test_mean_err_str)
        print('-' * 31)

        hparams_dict_save = {hp_name:[] for hp_name in hparams_names}
        hparams_dict_save['val_acc'] = []
        hparams_dict_save['val_avg'] = []
        hparams_dict_save['val_std'] = []
        hparams_dict_save['val_avg_std'] = []
        hparams_dict_save['test_acc'] = []
        hparams_dict_save['test_avg'] = []
        hparams_dict_save['test_std'] = []
        hparams_dict_save['test_avg_std'] = []
        for hp_kv in hparams_dict:
            for k, v in hp_kv:
                hparams_dict_save[k].append(v)
            hparams_dict_save['val_acc'].append(hparams_dict[hp_kv]['val_acc'])
            val_mean, val_err, val_mean_err_str = os_utils.format_mean(hparams_dict[hp_kv]['val_acc'])
            hparams_dict_save['val_avg'].append(val_mean)
            hparams_dict_save['val_std'].append(val_err)
            hparams_dict_save['val_avg_std'].append(val_mean_err_str)
            hparams_dict_save['test_acc'].append(hparams_dict[hp_kv]['test_acc'])
            test_mean, test_err, test_mean_err_str = os_utils.format_mean(hparams_dict[hp_kv]['test_acc'])
            hparams_dict_save['test_avg'].append(test_mean)
            hparams_dict_save['test_std'].append(test_err)
            hparams_dict_save['test_avg_std'].append(test_mean_err_str)

        df = pd.DataFrame(hparams_dict_save)
        # Sort Descending by val
        df_sort = df.sort_values(by=['val_avg', 'val_std'], ascending=[False, True])
        df_sort.to_csv(file_name+'_sorted_val.csv', index=False)
        print('Search results are saved at: ', str(file_name))

        # # Sort Descending by test
        # df_sort = df.sort_values(by=['test_avg', 'test_std'], ascending=[False, True])
        # df_sort.to_csv(file_name + '_sorted_test.csv', index=False)
        # print('Search results are saved at: ', str(file_name))

        # debug
        # df_read =  pd.read_csv(file_name)
    else:
        raise ValueError('args.avg_std is not matched!')
        # plot results

def print_final_search_results(records, selection_method, latex):
    """Given all records, print a results table for each dataset."""
    grouped_records, alg_names, dataset_names = group_and_sort(records, selection_method)

    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        test_envs = range(datasets.num_environments(dataset))
        # initialize the number of columns by [*test_envs, "Avg"] and the number of rows by alg_names
        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            means = []
            for j, test_env in enumerate(test_envs):
                trial_accs = (grouped_records
                              .filter_equals(
                    "dataset, algorithm, test_env",
                    (dataset, algorithm, test_env)
                ).select("sweep_acc"))
                mean, err, table[i][j] = os_utils.format_mean(trial_accs, latex)
                means.append(mean)
            if None in means:
                table[i][-1] = "X"
            else:
                table[i][-1] = "{:.1f}".format(sum(means) / len(means))

        col_labels = [
            "Algorithm",
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ]
        header_text = (f"Dataset: {dataset}, "
                       f"model selection method: {selection_method.name}")
        os_utils.print_table(table, header_text, alg_names, list(col_labels),
                    colwidth=20, latex=latex)

    # Print an "averages" table
    if latex:
        print()
        print("\\subsubsection{Averages}")

    table = [[None for _ in [*dataset_names, "Avg"]] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        means = []
        for j, dataset in enumerate(dataset_names):
            trial_averages = (grouped_records
                              .filter_equals("algorithm, dataset", (algorithm, dataset))
                              .group("trial_seed")
                              .map(lambda trial_seed, group:
                                   group.select("sweep_acc").mean()
                                   )
                              )
            mean, err, table[i][j] = os_utils.format_mean(trial_averages, latex)
            means.append(mean)
        if None in means:
            table[i][-1] = "X"
        else:
            table[i][-1] = "{:.1f}".format(sum(means) / len(means))

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"
    os_utils.print_table(table, header_text, alg_names, col_labels, colwidth=25,
                latex=latex)