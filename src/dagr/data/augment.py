import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from typing import List

import cv2
import numpy as np
import numba
import torch_geometric.transforms as T


@numba.njit
def _add_event(x, y, xlim, ylim, p, i, count, pos, mask, threshold=1):
    count[ylim, xlim] += float(p * (1 - abs(x - xlim)) * (1 - abs(y - ylim)))
    pol = 1 if count[ylim, xlim] > 0 else -1

    if pol * count[ylim, xlim] > threshold:
        count[ylim, xlim] -= pol * threshold

        mask[i] = True
        pos[i, 0] = xlim
        pos[i, 1] = ylim


@numba.njit
def _subsample(pos: np.ndarray, polarity: np.ndarray, mask: np.ndarray, count: np.ndarray, threshold=1):
    for i in range(len(pos)):
        x, y = pos[i]
        x0, x1 = int(x), int(x+1)
        y0, y1 = int(y), int(y+1)

        _add_event(x, y, x0, y0, polarity[i,0], i=i, count=count, pos=pos, mask=mask, threshold=threshold)
        _add_event(x, y, x1, y0, polarity[i,0], i=i, count=count, pos=pos, mask=mask, threshold=threshold)
        _add_event(x, y, x0, y1, polarity[i,0], i=i, count=count, pos=pos, mask=mask, threshold=threshold)
        _add_event(x, y, x1, y1, polarity[i,0], i=i, count=count, pos=pos, mask=mask, threshold=threshold)


def _crop_events(data, left, right, not_crop_idx=None):
    if not_crop_idx is None:
        not_crop_idx = torch.all((data.pos >= left) & (data.pos <= right), dim=1)

    data.x = data.x[not_crop_idx]
    data.pos = data.pos[not_crop_idx]

    if hasattr(data, "t"):
        data.t = data.t[not_crop_idx]

    return data

def _crop_image(image, left, right):
    xmin, ymin = left
    xmax, ymax = right
    image[:ymin, :] = 0
    image[ymax:, :] = 0
    image[:, :xmin] = 0
    image[:, xmax:] = 0
    return image

def _resize_image(image, height, width, bg=None):

    image = image[0].permute(1, 2, 0).numpy()
    new_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)

    px = (new_image.shape[1] - image.shape[1])//2
    py = (new_image.shape[0] - image.shape[0])//2

    if px >= 0:
        bg = new_image[py:py+image.shape[0], px:px+image.shape[1]]
    else:
        assert bg is not None
        bg[-py:-py+new_image.shape[0], -px:-px+new_image.shape[1]] = new_image

    bg = torch.from_numpy(bg).permute(2, 0, 1)[None]

    return bg

def _crop_bbox(bbox: torch.Tensor, left: torch.Tensor, right: torch.Tensor):
    bbox = bbox.clone()
    bbox[:,2:4] += bbox[:,:2]
    bbox[:,:2] = torch.clamp(bbox[:,:2], min=left, max=right)
    bbox[:,2:4] = torch.clamp(bbox[:,2:4], min=left, max=right)
    bbox[:,2:4] -= bbox[:,:2]
    return bbox

def _scale_and_clip(x, scale):
    return int(torch.clamp(x * scale, min=0, max=scale-1))


class RandomHFlip(BaseTransform):
    def __init__(self, p: float):
        self.p = p

    def forward(self, data: Data):
        if torch.rand(1) > self.p:
            return data

        data.pos[:,0] = data.width - 1 - data.pos[:,0]

        if hasattr(data, "image"):
            image = data.image[0].permute(1,2,0).numpy()
            image = np.ascontiguousarray(image[:,::-1])
            image = torch.from_numpy(image).permute(2, 0, 1)[None]
            data.image = image

        if hasattr(data, "bbox"):
            data.bbox[:, 0] = data.width - 1 - (data.bbox[:, 0] + data.bbox[:, 2])

        if hasattr(data, "bbox0"):
            data.bbox0[:, 0] = data.width - 1 - (data.bbox0[:, 0] + data.bbox0[:, 2])

        return data


class Crop(BaseTransform):
    r"""Crop with max and min values, has to be called before a graph is generated.

    Args:
        min (List[float]): min value per dimension
        max (List[float]): max value per dimension
    """
    def __init__(self, min: List[float], max: List[float]):
        self.min = torch.as_tensor(min)
        self.max = torch.as_tensor(max)

    def init(self, height, width):
        size = [width, height]
        self.max = torch.IntTensor([_scale_and_clip(m, s) for m, s in zip(self.max, size)])
        self.min = torch.IntTensor([_scale_and_clip(m, s) for m, s in zip(self.min, size)])

    def forward(self, data: Data):
        data = _crop_events(data, self.min, self.max)

        if hasattr(data, "image"):
            data.image = _crop_image(data.image, self.min, self.max)

        # crop bbox to dimension
        if hasattr(data, "bbox"):
            data.bbox = _crop_bbox(data.bbox, self.min, self.max)

        if hasattr(data, "bbox0"):
            data.bbox0 = _crop_bbox(data.bbox0, self.min, self.max)

        return data


class RandomZoom(BaseTransform):
    def __init__(self, zoom, subsample=False):
        self.zoom = zoom
        self.subsample = subsample
        self.image = None

        if subsample:
            self._count = None

    def _subsample(self, data, zoom, count):
        pos_zoom = data.pos.numpy()

        mask = np.zeros(len(data.pos), dtype="bool")
        _subsample(pos_zoom, data.x.numpy(), mask, count, threshold=1/(float(zoom)**2))

        data.pos = torch.from_numpy(pos_zoom[mask].astype("int16")) # implicit cast to int
        data.x = data.x[mask]
        if hasattr(data, "t"):
            data.t = data.t[mask]

        return data

    def init(self, height, width):
        self.image = np.zeros((height, width, 3), dtype="uint8")
        self._count = np.zeros((height + 1, width + 1), dtype="float32")

    def forward(self, data):
        zoom = torch.rand(1) * (self.zoom[1] - self.zoom[0]) + self.zoom[0]
        width, height = int(np.ceil(data.width * zoom)), int(np.ceil(data.height * zoom))
        H, W = self.image.shape[:2]

        data.pos[:, 0] = ((data.pos[:, 0] - W // 2) * zoom + W // 2).to(torch.int16)
        data.pos[:, 1] = ((data.pos[:, 1] - H // 2) * zoom + H // 2).to(torch.int16)

        if self.subsample and zoom < 1:
            data = self._subsample(data, float(zoom), count=self._count.copy())

        if hasattr(data, "image"):
            data.image = _resize_image(data.image, width=width, height=height, bg=self.image.copy() if zoom < 1 else None)

        if hasattr(data, "bbox"):
            data.bbox[:,2:4] *= zoom
            data.bbox[:,0] = ((data.bbox[:,0] - W//2) * zoom + W//2)
            data.bbox[:,1] = ((data.bbox[:,1] - H//2) * zoom + H//2)

        if hasattr(data, "bbox0"):
            data.bbox0[:,2:4] *= zoom
            data.bbox0[:,0] = ((data.bbox0[:,0] - W//2) * zoom + W//2)
            data.bbox0[:,1] = ((data.bbox0[:,1] - H//2) * zoom + H//2)

        return data


class RandomCrop(BaseTransform):
    r"""Random crop, assumes all pos values are in [0,1]

    Args:
        size (List[float]): crop size per dimension
        dim (List[int]): dimensions of the crop, default = [0,1]
        p (float): probability of applying the crop
        max_trials (int): maximum number of crop resampling attempts
        min_events (int): minimum number of events kept after cropping
    """

    def __init__(
        self,
        size=[0.75, 0.75],
        dim=[0, 1],
        p=0.5,
        max_trials=10,
        min_events=1,
    ):
        self.size = torch.as_tensor(size)
        self.dim = dim
        self.p = p
        self.max_trials = max_trials
        self.min_events = min_events

    def init(self, height, width):
        size = torch.IntTensor([width, height])
        self.size = torch.IntTensor(
            [_scale_and_clip(s, ss) for s, ss in zip(self.size, size)]
        )
        self.left_max = size - self.size

    def forward(self, data: Data):
        if torch.rand(1) > self.p:
            return data

        original = data

        for _ in range(self.max_trials):
            left = (
                torch.rand(len(self.dim)) * self.left_max
            ).to(torch.int16)

            right = left + self.size

            cropped = _crop_events(
                original.clone(),
                left,
                right,
            )

            # Reject empty / nearly empty crops
            if cropped.num_nodes < self.min_events:
                continue

            if hasattr(cropped, "bbox"):
                cropped.bbox = _crop_bbox(
                    cropped.bbox,
                    left,
                    right,
                )

                # Reject degenerate boxes
                valid_bbox = (
                    (cropped.bbox[:, 2] > 0)
                    & (cropped.bbox[:, 3] > 0)
                )

                if not valid_bbox.all():
                    continue

            if hasattr(cropped, "bbox0"):
                cropped.bbox0 = _crop_bbox(
                    cropped.bbox0,
                    left,
                    right,
                )

            if hasattr(cropped, "image"):
                cropped.image = _crop_image(
                    cropped.image,
                    left,
                    right,
                )

            return cropped

        # No valid crop found
        return original


class RandomTranslate(BaseTransform):
    r"""Random crop, assumes all pos values are in [0,1]

    Args:
        size (float): crop size per dimension
        dim (int): dimension of the crop, default = [0,1]
    """
    def __init__(self, size: List[float]):
        self.size = torch.as_tensor(size).float()
        self.image = None

    def init(self, height, width):
        size = [width, height]
        self.size = torch.IntTensor([_scale_and_clip(s, ss) for s, ss in zip(self.size, size)])
        self.image = np.zeros((height + 2 * self.size[1], width + 2 * self.size[0], 3), dtype="uint8")

    def pad(self, image, bg):
        px = (bg.shape[1] - image.shape[1])//2
        py = (bg.shape[0] - image.shape[0])//2
        bg[py:py + image.shape[0], px:px + image.shape[1]] = image
        return bg

    def forward(self, data: Data):
        move_px = (self.size * (torch.rand(len(self.size)) * 2 - 1)).to(torch.int16)
        data.pos = data.pos + move_px

        if hasattr(data, "image"):
            image = data.image[0].permute(1, 2, 0).numpy()
            image = self.pad(image, self.image.copy())
            image = image[self.size[1]-move_px[1]:self.size[1]-move_px[1]+data.height, \
                          self.size[0]-move_px[0]:self.size[0]-move_px[0]+data.width]
            data.image = torch.from_numpy(image).permute(2, 0, 1)[None]

        if hasattr(data, "bbox"):
            data.bbox[:,:2] += move_px

        if hasattr(data, "bbox0"):
            data.bbox0[:,:2] += move_px

        return data


class Augmentations:
    transform_testing = T.Compose([
        Crop([0, 0], [1, 1]),
    ])

    def __init__(self, args):
        self.transform_training = T.Compose([
            RandomHFlip(p=args.aug_p_flip),
            RandomCrop([0.75, 0.75], p=0.2,max_trials=10, min_events=10000),
            RandomZoom(zoom=[1, args.aug_zoom], subsample=True),
            RandomTranslate([args.aug_trans, args.aug_trans, 0]),
            Crop([0, 0], [1, 1]),
        ])

def init_transforms(transforms, height, width):
    for t in transforms:
        if hasattr(t, "init"):
            t.init(height=height, width=width)