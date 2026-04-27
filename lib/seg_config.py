"""Single source of truth for the segmentation contract.

The pipeline assumes one segmentation model. Several values
(num_classes, nonskin label ids, tag->label map, label remap) are tightly
coupled to the model's label space — swap the model and they all have to
change together. Keeping them centralised here prevents the literals from
drifting apart across modules.

Defaults match `mattmdjaga/segformer_b2_clothes` (the model the pipeline was
developed against). Override at runtime by calling `set_active()` with a
dict / OmegaConf node / yaml path before any GaussianModel is constructed.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List


@dataclass(frozen=True)
class SegConfig:
    model_name: str = "mattmdjaga/segformer_b2_clothes"
    num_classes: int = 16
    nonskin_categories: List[int] = field(
        default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9, 10]
    )
    cat_map: Dict[int, int] = field(default_factory=lambda: {16: 0, 17: 0})
    tag_categories: Dict[str, List[int]] = field(default_factory=lambda: {
        'hair':  [2],
        'shirt': [4],
        'pants': [5, 6],
        'shoes': [9, 10],
        'dress': [7],
    })
    render_categories: List[int] = field(
        default_factory=lambda: [2, 4, 5, 6, 7, 9, 10]
    )


_active = SegConfig()


def get() -> SegConfig:
    return _active


def set_active(cfg) -> SegConfig:
    """Override the active config from a dict, OmegaConf node, SegConfig, or yaml path."""
    global _active
    if isinstance(cfg, SegConfig):
        _active = cfg
        return _active
    if isinstance(cfg, str):
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
    else:
        try:
            from omegaconf import OmegaConf, DictConfig
            if isinstance(cfg, DictConfig):
                cfg = OmegaConf.to_container(cfg, resolve=True)
        except ImportError:
            pass
    if not isinstance(cfg, dict):
        raise TypeError(
            f"set_active expected dict / DictConfig / SegConfig / path, got {type(cfg)}"
        )
    overrides = {k: v for k, v in cfg.items() if k in SegConfig.__dataclass_fields__}
    _active = replace(_active, **overrides)
    return _active
