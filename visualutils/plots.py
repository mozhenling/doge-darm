"""
Created on Fri Mar 19 14:38:01 2021

@author: mozhenling
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from sklearn.manifold import TSNE
import seaborn as sns

def plotDesity(train_dict, test_dict, perplexity=30, n_iter=1000, random_state=42,
            title='Label', fontsize = 10, xlabel='T-SNE Component 1', ylabel='Density',
           save_dir = None, format = 'png', figsize = (14, 7), dpi = 300, non_text = False):
    """Compare the densities of t-sne component 1 of training and test domains for each label"""
    axis_label_fontsize = fontsize+0.5
    title_fontsize = fontsize+1
    tick_fontsize = fontsize
    legend_fontsize = fontsize

    train_features = train_dict['feature']
    train_labels = train_dict['label']
    test_features = test_dict['feature']
    test_labels = test_dict['label']

    # Combine train and test features for TSNE
    combined_features = np.vstack((train_features, test_features))
    combined_labels = np.hstack((train_labels, test_labels))
    domains = np.array(['train'] * len(train_labels) + ['test'] * len(test_labels))

    # Run TSNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    tsne_results = tsne.fit_transform(combined_features)

    # Get unique labels
    unique_labels = np.unique(combined_labels)

    # Plot distributions of each class of features for both domains
    fig, axs = plt.subplots(len(unique_labels), 1, figsize=(figsize[0], len(unique_labels) * 3), dpi = dpi , sharex=True)
    for i, label in enumerate(unique_labels):
        sns.kdeplot(tsne_results[(combined_labels == label) & (domains == 'train'), 0], ax=axs[i], color='blue', label='Train', shade=True)
        sns.kdeplot(tsne_results[(combined_labels == label) & (domains == 'test'), 0], ax=axs[i], color='red', label='Test', shade=True)
        axs[i].set_title(title+f' {label}', fontsize=title_fontsize)
        axs[i].set_xlabel(xlabel, fontsize=axis_label_fontsize)
        axs[i].set_ylabel(ylabel, fontsize=axis_label_fontsize)
        axs[i].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        axs[i].legend(fontsize=legend_fontsize)

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, format=format)
    plt.show()

def plotTSNE(train_dict, test_dict, perplexity=30, n_iter=1000, random_state=42,
             axis_tight=True, title=None, fontsize = 12, xlabel='T-SNE Component 1', ylabel='T-SNE Component 2',
           save_dir = None, format = 'png', figsize = (8, 6), dpi = 300, non_text = False, marker_size=100, is_grid=False):
    """visualize features from training and test domains on 2d t-sne plots"""
    axis_label_fontsize = fontsize+0.5
    title_fontsize = fontsize+1
    tick_fontsize = fontsize
    legend_fontsize = fontsize

    train_features = train_dict['feature']
    train_labels = train_dict['label']
    test_features = test_dict['feature']
    test_labels = test_dict['label']

    # Combine train and test features for TSNE
    combined_features = np.vstack((train_features, test_features))
    combined_labels = np.hstack((train_labels, test_labels))
    domains = np.array(['train'] * len(train_labels) + ['test'] * len(test_labels))

    # Run TSNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    tsne_results = tsne.fit_transform(combined_features)

    # Get unique labels
    unique_labels = np.unique(combined_labels)
    markers = ['o','*', 's', 'D', '^', 'x', 'p',  '<', '>', 'h']  # A list of markers
    domain_colors = {'train': 'blue', 'test': 'red'}  # Colors for domains

    plt.figure(figsize=figsize, dpi=dpi)

    # Plot TSNE results with different colors for domains and markers for labels
    for i, label in enumerate(unique_labels):
        for domain in ['train', 'test']:
            indices = (combined_labels == label) & (domains == domain)
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                        marker=markers[i % len(markers)],
                        color=domain_colors[domain],
                        label=f'{domain} label {label}' if domain == 'train' else f'{domain} label {label}',
                        alpha=0.7, s=marker_size)

    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=axis_label_fontsize)
    plt.ylabel(ylabel, fontsize=axis_label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best',  fontsize=fontsize+0.5)
    plt.grid(is_grid)
    if save_dir is not None:
        plt.savefig(save_dir, format=format)
    plt.show()


def plot3D(X, Y, Z, markoptimal=True, axis_tight=True, title=None, fontsize = 10, xlabel='x', ylabel='y', zlabel='z',
           save_dir = None, format = 'png', figsize = (8, 6), dpi = 300, non_text = False, elev = 30, azim=60):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.rc('font', size=fontsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=('viridis'))  # 'cool' looks good # 'viridis'

    # Set tick label fontsize
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='z', which='major', labelsize=fontsize)

    if axis_tight:
        ax.set_xlim(min(X.flatten()), max(X.flatten()))
        ax.set_ylim(min(Y.flatten()), max(Y.flatten()))
        ax.set_zlim(min(Z.flatten()), max(Z.flatten()))

    if non_text:
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.axes.zaxis.set_visible(False)
    else:
        ax.set_xlabel(xlabel, fontsize = fontsize + 1)
        ax.set_ylabel(ylabel, fontsize = fontsize + 1)
        ax.set_zlabel(zlabel, fontsize = fontsize + 1)
        ax.set_title(title, fontsize = fontsize + 2)

    if markoptimal:
        # Find the optimal point (max z value in this case)
        optimal_idx = np.argmax(Z)
        X_opt = X.flatten()[optimal_idx]
        Y_opt = Y.flatten()[optimal_idx]
        Z_opt = Z.flatten()[optimal_idx]

        # Annotate the optimal point with its coordinates
        ax.text(X_opt, Y_opt, Z_opt, f'({X_opt:.1f}, {Y_opt:.1f}, {Z_opt:.1f})', color='black',zorder=100, fontsize=fontsize)
        # Mark the optimal point
        ax.scatter(X_opt, Y_opt, Z_opt, color='red', s=50, label='Optimal Point', zorder=100)
        # Use quiver to draw an arrow pointing to the optimal point
        # arrow_length = 10  # Length of the arrow
        # ax.quiver(X_opt, Y_opt, Z_opt + arrow_length,
        #           0, 0, -arrow_length,
        #           color='r', arrow_length_ratio=0.1, linewidth=2, zorder=100)

    #-- save the image
    plt.tight_layout()
    ax.legend(fontsize=fontsize+0.5)

    # Adjust the vantage point
    ax.view_init(elev=elev, azim=azim)

    if save_dir is not None:
        plt.savefig(save_dir, format=format)
    plt.show()



def get_X_AND_Y(X_min, X_max, Y_min, Y_max, reso=0.01, step=None):
    """
    reso: normalized resolution for the image
    num: number of points
    step: un-normalized resolution for the image

    num = 1 / reso
    reso = 1 / num
    step = (x_max-x_min )/ num
    step = reso * (x_max - x_min)

    """
    if reso is not None and step is None:
        step_used = reso * (X_max - X_min)
    elif step is not None and reso is None:
        step_used = step
    elif reso is not None and step is not None:
        raise ValueError("Choose reso (resolution in percentage) or step?")
    else:
        raise ValueError('reso and step should not be both None!')
    X = np.arange(X_min, X_max, step_used)
    Y = np.arange(Y_min, Y_max, step_used)
    X, Y = np.meshgrid(X, Y)
    return (X, Y)


# -- for debugging
def Rastrigin(X=None, Y=None, objMin=True, is2Show=False, X_min=-5.52, X_max=5.12, Y_min=-5.12, Y_max=5.12, **kwargs):
    A = 10
    if is2Show:
        X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max, **kwargs)
        Z = 2 * A + X ** 2 - A * np.cos(2 * np.pi * X) + Y ** 2 - A * np.cos(2 * np.pi * Y)
        return (
            X, Y, Z, 100, 'Rastrigin function-3D')
    Z = 2 * A + X ** 2 - A * np.cos(2 * np.pi * X) + Y ** 2 - A * np.cos(2 * np.pi * Y)
    if objMin:
        return Z
    return -Z


if __name__ == '__main__':
    X_min = -1
    X_max = 3
    Y_min = -1
    Y_max = 3

    num = 50
    reso = 1/num

    X, Y, Z, z_max, title = Rastrigin(X_min=X_min, X_max=X_max, Y_min=Y_min, Y_max=Y_max, reso= reso, is2Show=True)  # Schwefel, Rastrigin
    plot3D(X, Y, Z, z_max, title)


