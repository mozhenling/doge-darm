import argparse
import pandas as pd
from visualutils.plots import plot3D, plotTSNE
import numpy as np
import ast
import math
import os
import torch
from algorithms.trainer import train
from scipy.interpolate import griddata

def plot_T_SNE(file, seed = 0, batch_size = 128, device = 'cpu', **kwargs):
    dict_init = torch.load(file)
    algorithm_dict = dict_init["model_dict"]
    hparams = dict_init["model_hparams"]
    args_dict = dict_init["args"]
    args_dict['hparams'] = hparams

    #-- adjust the batch size to have a better plot
    if batch_size is not None:
        args_dict['hparams']['batch_size']=batch_size

    train_xy_dict, test_xy_dict = train(args_dict, algorithm_dict=algorithm_dict, t_sne=True)
    plotTSNE(train_xy_dict, test_xy_dict, **kwargs)

    return

def plot_hparams3D(file, hparam_pair, finer_num=0, interpo_method = 'cubic',z_key = 'test_avg', **kwargs):
    x_key, x_base = hparam_pair[0]['name'], hparam_pair[0]['base']
    y_key, y_base = hparam_pair[1]['name'], hparam_pair[1]['base']

    df = pd.read_csv(file)
    # Sort Descending by test
    df_sort = df.sort_values(by=[x_key,y_key], ascending=[True, True])
    # create meshgrid
    # Restore the n-by-n NumPy array from the DataFrame
    n = int(np.sqrt(len(df_sort)))

    if x_base is not None:
        x =np.array([math.log(x, x_base) for x in df_sort[x_key]])
    else:
        x = df_sort[x_key].to_numpy()
    if y_base is not None:
        y = np.array([math.log(y, y_base) for y in df_sort[y_key]])
    else:
        y = df_sort[y_key].to_numpy()
    z = df_sort[z_key].to_numpy()



    X, Y, Z = x.reshape(n, n), y.reshape(n, n), z.reshape(n, n)

    if finer_num > n:
        # Flatten the data for griddata
        points = np.array([X.flatten(), Y.flatten()]).T
        values = Z.flatten()
        # Define a finer grid
        xi = np.linspace(X.min(), X.max(), finer_num)
        yi = np.linspace(Y.min(), Y.max(), finer_num)
        Xi, Yi = np.meshgrid(xi, yi)
        # Interpolate the Z values on the finer grid
        Zi = griddata(points, values, (Xi, Yi), method=interpo_method)
        X, Y, Z = Xi, Yi, Zi

    plot3D(X, Y, Z, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="plot figures")
    parser.add_argument("--input_dir", type=str, default=r'./outputs/hparams_outs')
    parser.add_argument("--save_dir", type=str, default=r'./outputs/plots_outs')
    parser.add_argument('--hparam_pair', type=ast.literal_eval, default=None)
    parser.add_argument("--plot_type", type=str, choices=['3d','3D', 'tsne','t-sne','t_sne','T-SNE', 'T_SNE'],default='3D')
    parser.add_argument('--read_name', type=str, default="PU_bearing")
    parser.add_argument('--save_name', type=str, default="PU_bearing")
    parser.add_argument('--dataset', type=str, default="PU_bearing")
    parser.add_argument('--test_env', type=int, default=0)

    args = parser.parse_args()


    os.makedirs(args.save_dir, exist_ok=True)
    if args.plot_type in ['3d','3D']:
        plot_hparams3D(file =os.path.join(args.input_dir, args.read_name), hparam_pair=args.hparam_pair, fontsize=12,
                       # elev=elev,azim=azim,xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                       save_dir =os.path.join(args.save_dir, args.save_name) )
    elif args.plot_type in ['tsne','t-sne','t_sne','T-SNE', 'T_SNE']:
        plot_T_SNE(file =os.path.join(args.input_dir, args.read_name),
                   fontsize=14, figsize = (8, 6), marker_size=150,
                   save_dir =os.path.join(args.save_dir, args.save_name))
    else:
        raise NotImplementedError
