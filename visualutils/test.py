import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')
def visualize_tsne_domains(train_dict, test_dict, perplexity=30, n_iter=1000, random_state=42):
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
    markers = ['o', 's', 'D', '^', 'x', 'p', '*', '<', '>', 'h']  # A list of markers
    domain_colors = {'train': 'blue', 'test': 'red'}  # Colors for domains

    plt.figure(figsize=(14, 7))

    # Plot TSNE results with different colors for domains and markers for labels
    for i, label in enumerate(unique_labels):
        for domain in ['train', 'test']:
            indices = (combined_labels == label) & (domains == domain)
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                        marker=markers[i % len(markers)],
                        color=domain_colors[domain],
                        label=f'{domain} label {label}' if domain == 'train' else f'{domain} label {label}',
                        alpha=0.7)

    plt.title('TSNE Visualization of Training and Test Domains')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.grid(True)
    plt.show()

    # Plot distributions of each class of features for both domains
    fig, axs = plt.subplots(len(unique_labels), 1, figsize=(14, len(unique_labels) * 3), sharex=True)
    for i, label in enumerate(unique_labels):
        sns.kdeplot(tsne_results[(combined_labels == label) & (domains == 'train'), 0], ax=axs[i], color='blue', label='Train', shade=True)
        sns.kdeplot(tsne_results[(combined_labels == label) & (domains == 'test'), 0], ax=axs[i], color='red', label='Test', shade=True)
        axs[i].set_title(f'Label {label}')
        axs[i].set_xlabel('TSNE Component 1')
        axs[i].set_ylabel('Density')
        axs[i].legend()

    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming you have two dictionaries with 'feature' and 'label' keys for training and test data
train_dict = {
    'feature': np.random.rand(100, 50),  # Replace with your actual training features
    'label': np.random.randint(0, 5, 100)  # Replace with your actual training labels
}

test_dict = {
    'feature': np.random.rand(100, 50),  # Replace with your actual test features
    'label': np.random.randint(0, 5, 100)  # Replace with your actual test labels
}

visualize_tsne_domains(train_dict, test_dict)