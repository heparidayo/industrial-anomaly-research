"""Perlin 노이즈 (원본 DRAEM perlin.py의 rand_perlin_2d_np).

임계값을 넘긴 영역이 '가짜 결함'의 모양(마스크)이 됩니다.
"""

import math

import numpy as np


def _lerp(x, y, w):
    return (y - x) * w + x


def _fade(t):
    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3


def rand_perlin_2d_np(shape, res, rng: np.random.Generator | None = None) -> np.ndarray:
    """shape(H, W) 크기의 2D Perlin 노이즈. res는 각 축의 격자 수 (H, W를 나누어떨어지게)."""
    rng = np.random.default_rng() if rng is None else rng

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * rng.random((res[0] + 1, res[1] + 1))
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    def tile_grads(s1, s2):
        g = gradients[s1[0]:s1[1], s2[0]:s2[1]]
        return np.repeat(np.repeat(g, d[0], axis=0), d[1], axis=1)

    def dot(grad, shift):
        g = grid[:shape[0], :shape[1]]
        shifted = np.stack((g[..., 0] + shift[0], g[..., 1] + shift[1]), axis=-1)
        return (shifted * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])

    t = _fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * _lerp(_lerp(n00, n10, t[..., 0]), _lerp(n01, n11, t[..., 0]), t[..., 1])
