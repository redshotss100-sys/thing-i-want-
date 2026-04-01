import datetime
from collections import defaultdict

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import label as ndimage_label

from config import (
    GH,
    GW,
    AIR,
    TUNNEL,
    BORDER,
    TUNNEL_MAX_DIM,
    TUNNEL_MIN_LEN,
    NEST_MIN_AREA,
    FARM_MIN_AREA,
    FARM_MAX_AREA,
    DIRT,
    ROCK,
    GRASS,
)


_DIRT_NOISE = None


def sky_colour():
    now = datetime.datetime.now()
    t = (now.hour * 60 + now.minute + now.second / 60.0) / 1440.0
    kf = [
        (0.00, (5, 5, 20)), (0.08, (15, 10, 40)), (0.25, (30, 20, 60)),
        (0.29, (255, 120, 40)), (0.33, (135, 180, 255)), (0.50, (80, 150, 255)),
        (0.67, (100, 170, 255)), (0.75, (255, 140, 60)), (0.79, (50, 20, 60)),
        (0.83, (15, 10, 35)), (1.00, (5, 5, 20)),
    ]
    for i in range(len(kf) - 1):
        t0, c0 = kf[i]
        t1, c1 = kf[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(int(c0[j] + (c1[j] - c0[j]) * f) for j in range(3))
    return kf[0][1]


def get_dirt_noise():
    global _DIRT_NOISE
    if _DIRT_NOISE is None:
        rng = np.random.default_rng(999)
        _DIRT_NOISE = rng.integers(-8, 9, size=(GH, GW, 3), dtype=np.int16)
    return _DIRT_NOISE


class Pheromones:
    def __init__(self):
        self.food = np.zeros((GH, GW), dtype=np.float32)
        self.danger = np.zeros((GH, GW), dtype=np.float32)
        self.home = np.zeros((GH, GW), dtype=np.float32)

    def deposit(self, x, y, kind, amount):
        ix, iy = int(x), int(y)
        if 0 <= ix < GW and 0 <= iy < GH:
            if kind == 'food':
                self.food[iy, ix] = min(1.0, self.food[iy, ix] + amount)
            elif kind == 'home':
                self.home[iy, ix] = min(1.0, self.home[iy, ix] + amount)
            else:
                self.danger[iy, ix] = min(1.0, self.danger[iy, ix] + amount)

    def decay(self):
        self.food *= 0.996
        self.danger *= 0.985
        self.home *= 0.997

    def diffuse(self):
        k = np.array([[0.05, 0.1, 0.05], [0.10, 0.4, 0.10], [0.05, 0.1, 0.05]], dtype=np.float32)
        self.food = convolve(self.food, k, mode='constant')
        self.danger = convolve(self.danger, k, mode='constant')
        self.home = convolve(self.home, k, mode='constant')


class RoomDetector:
    R_TUNNEL = 1
    R_NEST = 2
    R_FARM = 3
    R_STORAGE = 4
    R_CHAMBER = 5
    NAMES = {1: 'tunnel', 2: 'nest', 3: 'farm', 4: 'storage', 5: 'chamber'}

    def __init__(self):
        self.grid = np.zeros((GH, GW), dtype=np.uint8)
        self.counts = defaultdict(int)

    def detect(self, world, surface_y):
        self.grid[:] = 0
        self.counts = defaultdict(int)
        mask = np.zeros((GH, GW), dtype=bool)
        mask[surface_y:, :] = (world[surface_y:, :] == TUNNEL) | (world[surface_y:, :] == AIR)
        labeled, n = ndimage_label(mask)
        for cid in range(1, n + 1):
            cmask = labeled == cid
            ys, xs = np.where(cmask)
            area = len(xs)
            if area < 4:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            w = x1 - x0 + 1
            h = y1 - y0 + 1
            aspect = max(w, h) / max(1, min(w, h))
            fill = area / max(1, w * h)
            rtype = self._classify(w, h, area, aspect, fill, cmask, xs, ys, x0, y0, world)
            if rtype:
                self.grid[cmask] = rtype
                self.counts[rtype] += 1

    def _classify(self, w, h, area, aspect, fill, cmask, xs, ys, x0, y0, world):
        if min(w, h) <= TUNNEL_MAX_DIM and max(w, h) >= TUNNEL_MIN_LEN:
            return RoomDetector.R_TUNNEL
        if area >= NEST_MIN_AREA and aspect <= 2.2 and fill >= 0.35:
            return RoomDetector.R_NEST
        if FARM_MIN_AREA <= area <= FARM_MAX_AREA:
            if 1.2 <= aspect <= 3.5 and 0.25 <= fill <= 0.65:
                if self._check_semicircle(cmask, xs, ys, x0, y0, w, h):
                    return RoomDetector.R_FARM
        if area >= 15 and fill >= 0.45:
            return RoomDetector.R_CHAMBER
        return 0

    def _check_semicircle(self, cmask, xs, ys, x0, y0, w, h):
        if w < 5 or h < 3:
            return False
        row_counts = np.zeros(h, dtype=int)
        for row in range(h):
            row_counts[row] = np.sum(cmask[y0 + row, x0:x0 + w])
        mid = h // 2
        if mid == 0:
            return False
        top_avg = row_counts[:mid].mean()
        bot_avg = row_counts[mid:].mean()
        if row_counts[0] < row_counts[-1] or top_avg <= bot_avg * 1.1:
            return False
        diffs = np.diff(row_counts)
        increases = int((diffs > 0).sum())
        decreases = int((diffs < 0).sum())
        return decreases > increases and top_avg > bot_avg * 1.15

    def room_at(self, x, y):
        ix, iy = int(x), int(y)
        if 0 <= ix < GW and 0 <= iy < GH:
            return int(self.grid[iy, ix])
        return 0
