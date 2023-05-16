import os, pandas

import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from matplotlib import pyplot as plt
from eval import global_tag


def draw_mask(tag, mask_path):
    """Example:
     root = '../self'
    for index, image_name in enumerate(os.listdir(root)):
        if image_name.__contains__('jpg'):
            continue
        draw_mask(image_name.replace('.png', ''), f'{root}/{image_name}')

    """
    mask = cv2.imread(mask_path, 0)
    x, y = mask.shape
    img = np.zeros([x, y, 3]).astype(np.int32)
    img[mask == 1] = (255, 0, 0)
    img[mask == 2] = (0, 255, 0)
    plt.imshow(img)
    # plt.savefig(f"../log/origin_mask/{tag}.png", dpi=2000, bbox_inches='tight')
    plt.show()


def draw_loss_impl(_tuples, label, legend_loc="upper left", mutil_tag=''):
    plt.title(label)
    for _tuple in _tuples:
        list_data, legend, colors = _tuple
        plt.plot([i for i in range(len(list_data))], list_data, c=colors, label=legend)
    plt.legend(loc=legend_loc)
    plt.ylabel("Number")
    plt.xlabel("Epoch")
    plt.savefig(f"../log/{mutil_tag if mutil_tag != '' else global_tag}/{label}.png", dpi=2000, bbox_inches='tight')
    plt.show()


def draw_single():
    path = f'../log/{global_tag}/params.npy'
    dic = np.load(path, allow_pickle=True).item()
    print(len(dic['dice_loss']))
    draw_loss_impl([(dic['train_loss'], 'train_loss', 'blue')], 'train_loss')
    draw_loss_impl([(dic['hausdorff'], 'hausdorff', 'blue')], 'hausdorff')
    draw_loss_impl([(dic['dice_loss'], 'dice', 'blue')], 'dice')
    draw_loss_impl([(dic['f1_score'], 'f1', 'blue')], 'f1')


def draw_mutils():
    tags = [global_tag, 'ag_cbam_second_u2net']
    labels = [global_tag, 'ag_cbam_u2net', 'U2net_adam', 'attn_u2net']
    colors = ['orange', 'blue', 'black', 'red']
    train_loss, hausdorff, dice_loss, f1_score = [], [], [], []
    for i, tag in enumerate(tags):
        path = f'../log/{tag}/params.npy'
        dic = np.load(path, allow_pickle=True).item()
        hausdorff.append((dic['hausdorff'], labels[i], colors[i]))
        dice_loss.append((dic['dice_loss'], labels[i], colors[i]))
        f1_score.append((dic['f1_score'], labels[i], colors[i]))

    mutil_tag = 'other'
    draw_loss_impl(hausdorff, 'hausdorff', legend_loc='upper right', mutil_tag=mutil_tag)
    draw_loss_impl(dice_loss, 'dice', legend_loc='upper left', mutil_tag=mutil_tag)
    draw_loss_impl(f1_score, 'f1', legend_loc='lower right', mutil_tag=mutil_tag)


def script():
    tags = [global_tag, 'ag_cbam_second_u2net', 'ag_cbam_u2net', 'ag_se_u2net',
            'se_u2net_middle', 'simam_u2net_middle'
        , 'cbam_u2net_middle', 'cbam_u2net_second', 'cbam_u2net_after',
            'attn2_u2net', 'attn_u2net', 'U2net_adam',
            'Unet_adam', 'transUnet_adam', 'attentionUnet_adam']

    df_data = {
        'tag': [],
        'hd': [],
        'dice': [],
        'f1': []
    }
    for i, tag in enumerate(tags):
        path = f'../log/{tag}/params.npy'
        dic = np.load(path, allow_pickle=True).item()
        hd = round(min(dic['hausdorff'][:]) / 1, 4)
        dice = round(max(dic['dice_loss'][:]) / 1, 4)
        f1 = round(max(dic['f1_score'][:]) / 1, 8)
        print(f"{tag + ':' + ' ' * (20 - len(tag))}hd:{hd}, dice:{dice}, f1:{f1}")
        df_data['tag'].append(tag)
        df_data['hd'].append(hd)
        df_data['dice'].append(dice)
        df_data['f1'].append(f1)
    pandas.DataFrame(df_data).to_csv('../log/script.csv', index=False)


if __name__ == '__main__':
    script()
