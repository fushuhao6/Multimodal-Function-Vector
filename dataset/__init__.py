from typing import Any, Dict, Union
from pathlib import Path
from os import PathLike
from torch.utils.data import Dataset
import inspect

from .gqa_dataset import ICLRelationDataset
from .synthetic_dataset import SyntheticRelationDataset


def _filter_kwargs_for_constructor(ctor, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs that the constructor accepts (by signature)."""
    params = set(inspect.signature(ctor).parameters.keys())
    # don't ever forward 'self'
    params.discard("self")
    return {k: v for k, v in kwargs.items() if k in params}

def build_relation_dataset(
    kind: str,
    *,
    data_root: Union[str, Path, PathLike],
    model_config: Dict[str, Any],
    num_context: int,
    **kwargs: Any,
) -> Dataset:
    """
    Build a relation dataset by kind.

    Aliases:
      - SyntheticRelationDataset: {"synthetic", "syn", "sr"}
      - ICLRelationDataset:       {"icl", "incontext", "gqa"}
    """
    k = kind.strip().lower()
    data_root = str(Path(data_root))
    num_context = int(num_context)

    if k in {"synthetic", "synthetic_novel"}:
        # pull + normalize known args with defaults
        seed = int(kwargs.pop("seed", 1234))
        use_shuffled = bool(kwargs.pop("use_shuffled", False))
        diagonal = bool(kwargs.pop("diagonal", False))

        # filter remaining kwargs to only what SyntheticRelationDataset accepts
        extra = _filter_kwargs_for_constructor(SyntheticRelationDataset.__init__, kwargs)
        # avoid duplicating explicitly provided args
        for dup in ("data_root", "num_context", "model_config", "seed", "use_shuffled"):
            extra.pop(dup, None)

        return SyntheticRelationDataset(
            data_root=data_root,
            num_context=num_context,
            model_config=model_config,
            seed=seed,
            use_shuffled=use_shuffled,
            diagonal=diagonal,
            **extra,
        )

    if k in {"gqa", "gqa_large"}:
        # pull + normalize known args with defaults
        json_file = kwargs.pop("json_file", None)
        image_ext = str(kwargs.pop("image_ext", ".jpg"))
        seed = int(kwargs.pop("seed", 1234))
        use_shuffled = bool(kwargs.pop("use_shuffled", False))
        output_zero_shot = bool(kwargs.pop("output_zero_shot", False))
        tasks_per_relation = int(kwargs.pop("tasks_per_relation", 200))

        # filter remaining kwargs to only what ICLRelationDataset accepts
        extra = _filter_kwargs_for_constructor(ICLRelationDataset.__init__, kwargs)
        for dup in (
            "data_root",
            "model_config",
            "num_context",
            "json_file",
            "image_ext",
            "seed",
            "use_shuffled",
            "output_zero_shot",
            "tasks_per_relation",
        ):
            extra.pop(dup, None)

        return ICLRelationDataset(
            data_root=data_root,
            model_config=model_config,
            json_file=json_file,
            num_context=num_context,
            image_ext=image_ext,
            seed=seed,
            use_shuffled=use_shuffled,
            tasks_per_relation=tasks_per_relation,
            output_zero_shot=output_zero_shot,
            **extra,
        )

    raise ValueError("Unknown dataset kind '%s'. Use synthetic|syn|sr or icl|incontext|gqa." % kind)