# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ

import numpy as np
from skimage.morphology import disk
from skimage.filters.rank import modal
import cv2
import torch


def img_clip(img):
    return np.clip(img, 0, 255)


def hist_augment(src: np.ndarray, kernel=0, top=3, in_place=False):
    # "top" and "kernel" must be inversely proportional
    if in_place:
        img = np.asarray(src)
    else:
        img = np.asarray(src).copy()
    hist_r, hist_g, hist_b = np.zeros(256), np.zeros(256), np.zeros(256)
    vect_r, vect_g, vect_b = img[..., 0].reshape(-1), img[..., 1].reshape(-1), img[..., 2].reshape(-1)  # Flattening

    for pr, pg, pb in zip(vect_r, vect_g, vect_b):  # pixel red, green, blue
        hist_r[pr] += 1
        hist_g[pg] += 1
        hist_b[pb] += 1

    top_r = np.argsort(hist_r)[-top:]
    top_g = np.argsort(hist_g)[-top:]
    top_b = np.argsort(hist_b)[-top:]

    if top == 1:
        mask_r = img[..., 0] == top_r
        mask_g = img[..., 1] == top_g
        mask_b = img[..., 2] == top_b
        img[..., 0][mask_r + mask_g + mask_b], img[..., 1][mask_r + mask_g + mask_b], img[..., 2][mask_r + mask_g + mask_b] = 0, 0, 0

    else:
        r_max, r_min = top_r.max(), top_r.min()
        g_max, g_min = top_g.max(), top_g.min()
        b_max, b_min = top_b.max(), top_b.min()

        mask_r = (img[..., 0] <= r_max) * (img[..., 0] >= r_min)
        mask_g = (img[..., 1] <= g_max) * (img[..., 1] >= g_min)
        mask_b = (img[..., 2] <= b_max) * (img[..., 2] >= b_min)
        img[mask_r + mask_g + mask_b] = 0

    # Propagate the background removal effect (black pixels):
    if kernel:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        r, g, b = modal(r, disk(kernel)), modal(g, disk(kernel)), modal(b, disk(kernel))
        img = np.moveaxis(np.stack((r, g, b)), 0, 2)  # (row, col, channel)
        return img

    return img


def broken_mir(src: np.ndarray, rand=False):
    img = np.moveaxis(src.copy(), 2, 0)  # Move columns to last dimension
    c = img.shape[-1]
    if rand:
        mid_p = np.random.choice(c)
    else:
        mid_p = c // 2
    lp = img[..., :mid_p]
    lp = lp[..., ::-1]
    rp = img[..., mid_p:]
    rp = rp[..., ::-1]
    img[..., :mid_p] = lp
    img[..., mid_p:] = rp
    return np.moveaxis(img, 0, 2)


def equalize_image(src: np.ndarray, brightness=True, saturation=True):
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if brightness:
        v = cv2.equalizeHist(v)
    if saturation:
        s = cv2.equalizeHist(s)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def stabilize_channel(src: np.ndarray):
    img = src.copy()
    means = img.mean((0, 1))
    g_mean = means.mean()
    g_mean /= means
    img[..., 0], img[..., 1], img[..., 2] = img_clip(img[..., 0] * g_mean[0]), img_clip(img[..., 1] * g_mean[1]), img_clip(img[..., 2] * g_mean[2])
    return img


def crop(img, n_holes=1, size=6, fill_value=0, pad=0, gray=False):
    crops = []
    if isinstance(img, torch.Tensor):
        y = img.size(2)
        x = img.size(3)

        for _ in range(n_holes):
            ry = np.random.randint(size+pad, y-pad)
            rx = np.random.randint(size+pad, x-pad)
            if gray:
                patch = img[..., ry-size: ry, rx-size: rx].clone() / 255
                crops.append(patch.mean(1))
            else:
                crops.append(img[..., ry-size: ry, rx-size: rx].clone() / 255)

            if isinstance(fill_value, str):
                if fill_value == 'random':
                    rand = torch.rand(size, size) * 255
                    img[..., ry-size: ry, rx-size: rx] = rand
                elif fill_value == 'random3d':
                    rand = torch.rand(3, size, size) * 255
                    img[..., ry-size: ry, rx-size: rx] = rand
                else:
                    assert fill_value == 'random' or fill_value == 'random3d', "Can only choose between 'random' or 'random3d'"
            else:
                img[..., ry-size: ry, rx-size: rx] = fill_value * 255

        return crops[0]

    else:  # numpy
        if len(img.shape) != 4:
            img = np.expand_dims(img, 0)

        img = np.moveaxis(img, -1, 1)
        y = img.shape[2]
        x = img.shape[3]

        for _ in range(n_holes):
            ry = np.random.randint(size+pad, y-pad)
            rx = np.random.randint(size+pad, x-pad)
            if gray:
                patch = img[..., ry-size: ry, rx-size: rx].copy() / 255
                crops.append(patch.mean(1))
            else:
                crops.append(img[..., ry-size: ry, rx-size: rx].copy() / 255)

            if isinstance(fill_value, str):
                if fill_value == 'random':
                    rand = np.random.rand(size, size) * 255
                    img[..., ry-size: ry, rx-size: rx] = rand
                elif fill_value == 'random3d':
                    rand = np.random.rand(3, size, size) * 255
                    img[..., ry-size: ry, rx-size: rx] = rand
                else:
                    assert fill_value == 'random' or fill_value == 'random3d', "Can only choose between 'random' or 'random3d'"
            else:
                img[..., ry-size: ry, rx-size: rx] = fill_value * 255

        return torch.from_numpy(crops[0]).float()


'''
def hg(src: np.ndarray, top=30, in_place=False):
    # "top" and "kernel" must be inversely proportional
    if in_place:
        img = np.asarray(src)
    else:
        img = np.asarray(src).copy()
    hist = np.zeros(256)
    vect_r, vect_g, vect_b = img[..., 0].reshape(-1), img[..., 1].reshape(-1), img[..., 2].reshape(-1)  # Flattening

    for pr, pg, pb in zip(vect_r, vect_g, vect_b):  # pixel red, green, blue
        hist[pr] += 1
        hist[pg] += 1
        hist[pb] += 1

    top = np.argsort(hist)[-top:]

    p_max, p_min = top.max(), top.min()

    mask = (img[..., 0] <= p_max) * (img[..., 0] >= p_min)
    img[mask] = 0
    return img
'''