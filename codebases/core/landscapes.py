"""River-valley loss landscapes and gradient-noise models for the toy experiments E1-E4.

Conventions match `discussions/idea_v1.md`:

    straight valley   L(x, y) = (mu/2)(x - x*)^2 + (lam/2) y^2
    curved valley     L(x, y) = (mu/2)(x - x*)^2 + (lam/2)(y - f(x))^2

with 0 < mu << lam. The first coordinate x is the slow "river" direction; the second coordinate
y is the steep "hill" direction. For the curved valley the river floor is y = f(x), the local
river tangent is r(x) = (1, f'(x))/||.|| and the hill normal is n(x) = (-f'(x), 1)/||.||.

These objects are task-agnostic toy infrastructure shared by E1-E4; the closed-loop optimizer
that drives w_t lives in the experiment scripts, not here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "StraightValley",
    "CurvedValley",
    "gaussian_isotropic",
    "anisotropic_hill",
    "heavy_tailed",
]


@dataclass(frozen=True)
class StraightValley:
    """L(x, y) = (mu/2)(x - x_star)^2 + (lam/2) y^2 with mu << lam.

    The river direction is the constant unit vector r = (1, 0); the hill normal is n = (0, 1).
    """

    mu: float = 0.1
    lam: float = 10.0
    x_star: float = 5.0

    def loss(self, w: np.ndarray) -> float:
        x, y = w[..., 0], w[..., 1]
        return 0.5 * self.mu * (x - self.x_star) ** 2 + 0.5 * self.lam * y**2

    def grad(self, w: np.ndarray) -> np.ndarray:
        """grad L = [mu (x - x_star), lam y]."""
        x, y = w[..., 0], w[..., 1]
        return np.stack([self.mu * (x - self.x_star), self.lam * y], axis=-1)

    def river_tangent(self, w: np.ndarray) -> np.ndarray:
        out = np.zeros_like(np.asarray(w, dtype=float))
        out[..., 0] = 1.0
        return out

    def hill_normal(self, w: np.ndarray) -> np.ndarray:
        out = np.zeros_like(np.asarray(w, dtype=float))
        out[..., 1] = 1.0
        return out

    def hill_coordinate(self, w: np.ndarray) -> np.ndarray:
        """Signed distance to the river floor along the hill direction: d = y."""
        return np.asarray(w, dtype=float)[..., 1]


@dataclass(frozen=True)
class CurvedValley:
    """L(x, y) = (mu/2)(x - x_star)^2 + (lam/2)(y - f(x))^2, river floor y = f(x).

    Default river floor f(x) = a sin(k x). The tangent/normal are evaluated at the current x.
    """

    mu: float = 0.1
    lam: float = 10.0
    x_star: float = 5.0
    a: float = 1.0
    k: float = 0.5

    def f(self, x: np.ndarray) -> np.ndarray:
        return self.a * np.sin(self.k * x)

    def fp(self, x: np.ndarray) -> np.ndarray:
        """f'(x) = a k cos(k x)."""
        return self.a * self.k * np.cos(self.k * x)

    def loss(self, w: np.ndarray) -> float:
        x, y = w[..., 0], w[..., 1]
        return 0.5 * self.mu * (x - self.x_star) ** 2 + 0.5 * self.lam * (y - self.f(x)) ** 2

    def grad(self, w: np.ndarray) -> np.ndarray:
        """grad L = [mu (x - x*) - lam (y - f) f', lam (y - f)]."""
        x, y = w[..., 0], w[..., 1]
        resid = y - self.f(x)
        gx = self.mu * (x - self.x_star) - self.lam * resid * self.fp(x)
        gy = self.lam * resid
        return np.stack([gx, gy], axis=-1)

    def river_tangent(self, w: np.ndarray) -> np.ndarray:
        """r(x) = (1, f'(x)) / sqrt(1 + f'(x)^2)."""
        x = np.asarray(w, dtype=float)[..., 0]
        fp = self.fp(x)
        r = np.stack([np.ones_like(fp), fp], axis=-1)
        return r / np.linalg.norm(r, axis=-1, keepdims=True)

    def hill_normal(self, w: np.ndarray) -> np.ndarray:
        """n(x) = (-f'(x), 1) / sqrt(1 + f'(x)^2)."""
        x = np.asarray(w, dtype=float)[..., 0]
        fp = self.fp(x)
        n = np.stack([-fp, np.ones_like(fp)], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def hill_coordinate(self, w: np.ndarray) -> np.ndarray:
        """Signed offset from the river floor: d = y - f(x)."""
        w = np.asarray(w, dtype=float)
        return w[..., 1] - self.f(w[..., 0])


# --- gradient-noise models (Experiment E3) ---------------------------------
def gaussian_isotropic(rng: np.random.Generator, shape, sigma: float) -> np.ndarray:
    """Isotropic Gaussian noise xi ~ N(0, sigma^2 I)."""
    return sigma * rng.standard_normal(shape)


def anisotropic_hill(
    rng: np.random.Generator, shape, sigma: float, hill_scale: float = 4.0
) -> np.ndarray:
    """Anisotropic Gaussian noise, stronger in the hill (last) coordinate by `hill_scale`."""
    xi = sigma * rng.standard_normal(shape)
    xi[..., -1] *= hill_scale
    return xi


def heavy_tailed(
    rng: np.random.Generator, shape, sigma: float, df: float = 3.0
) -> np.ndarray:
    """Heavy-tailed finite-variance noise: Student-t with df>2, scaled to std `sigma`.

    Student-t with `df` degrees of freedom has variance df/(df-2); we rescale so the marginal
    standard deviation equals `sigma`.
    """
    if df <= 2.0:
        raise ValueError(f"df must exceed 2 for finite variance; got {df}")
    t = rng.standard_t(df, size=shape)
    return sigma * t / np.sqrt(df / (df - 2.0))
