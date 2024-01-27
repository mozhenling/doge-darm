import torch
import random

# other ref.: https://github.com/hitcszx/ALFs/blob/master/dataset.py

def sym_label_nosie(noise_rate, labels, num_classes):
    """
    https://github.com/filipe-research/tutorial_noisylabels/blob/main/codes/tutorial_sibgrapi20.ipynb
    """
    noise_label = []
    idx = list(range(len(labels)))
    random.shuffle(idx)
    num_noise = int(noise_rate * len(labels))
    noise_idx = idx[:num_noise]
    for i in range(len(labels)):
        if i in noise_idx:
            # Return a random integer N such that a <= N <= b.
            noiselabel = random.randint(0, num_classes-1)
            noise_label.append(noiselabel)
        else:
            noise_label.append(labels[i])

    return torch.tensor(noise_label)

def asym_label_nosie(noise_rate, labels, transition):
    """
    https://github.com/filipe-research/tutorial_noisylabels/blob/main/codes/tutorial_sibgrapi20.ipynb
    # class transition for asymmetric noise, e.g. for MNIST dataset below
    transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}
    For bearing faults: normal<->roller,  outer<-inner, outer->normal
    """
    noise_label = []
    idx = list(range(len(labels)))
    random.shuffle(idx)
    num_noise = int(noise_rate * len(labels))
    noise_idx = idx[:num_noise]
    for i in range(len(labels)):
        if i in noise_idx:
            noiselabel = transition[labels[i]]
            noise_label.append(noiselabel)
        else:
            noise_label.append(labels[i])

    return torch.tensor(noise_label)

def generate_noise(noise_rate, labels,  num_classes, transition=None, noise_mode='sym'):
    """
    https://github.com/filipe-research/tutorial_noisylabels/blob/main/codes/tutorial_sibgrapi20.ipynb
    # class transition for asymmetric noise
    transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}
    """
    noise_label = []
    idx = list(range(len(labels)))
    random.shuffle(idx)
    num_noise = int(noise_rate*len(labels))
    noise_idx = idx[:num_noise]
    for i in range(len(labels)):
        if i in noise_idx:
            if noise_mode=='sym':
                # Return a random integer N such that a <= N <= b.
                noiselabel = random.randint(0, num_classes)
                noise_label.append(noiselabel)
            elif noise_mode=='asym' and transition is not None:
                noiselabel = transition[labels[i]]
                noise_label.append(noiselabel)
        else:
            noise_label.append(labels[i])
            
    return torch.tensor(noise_label)