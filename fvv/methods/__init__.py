from .nearest_view import NearestView


def make_method(name: str, **kwargs):
    """Construct a method by name; kwargs are passed to methods that accept them."""
    if name == "nearest_view":
        return NearestView()
    if name in ("3dgs", "3dgs_noaffine"):
        from .gaussian_splat import GaussianSplat   # imports gsplat (CUDA) only when needed
        return GaussianSplat(color_affine=name == "3dgs", **kwargs)
    raise ValueError(f"unknown method {name!r}")


METHODS = ["nearest_view", "3dgs", "3dgs_noaffine"]
