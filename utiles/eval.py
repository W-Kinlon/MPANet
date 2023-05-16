import os

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff

# Model tag when training
global_tag = 'ag_mutil2_u2net'


def param_get(path):
    if os.path.exists(path):
        dic = np.load(path, allow_pickle=True).item()
        return dic['train_loss'], dic['hausdorff'], dic['val_loss'], dic['dice_loss'], dic['f1_score']
    return [], [], [], [], []


def get_dice(y_pred, y_true, smooth=0.00001):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)).item()


def hd_95(output, target):
    output = output.cpu().reshape([output.shape[0], -1]).numpy()
    target = target.cpu().reshape([target.shape[0], -1]).numpy()
    return max(directed_hausdorff(output, target)[0], directed_hausdorff(target, output)[0])


def get_f1(y_hat, y_true, epsilon=1e-7):
    y_hat = y_hat.cpu().reshape([y_hat.shape[0], -1]).numpy()
    y_true = y_true.cpu().reshape([y_true.shape[0], -1]).numpy()
    tp = np.sum(y_hat * y_true, axis=0)
    fp = np.sum(y_hat * (1 - y_true), axis=0)
    fn = np.sum((1 - y_hat) * y_true, axis=0)
    # The point of epsilon is to prevent the denominator from being 0, which python would give you an error
    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)
    f1 = 2 * p * r / (p + r + epsilon)
    f1 = np.where(np.isnan(f1), np.zeros_like(f1), f1)
    return np.mean(f1)


if __name__ == '__main__':
    pass
