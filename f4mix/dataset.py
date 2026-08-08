from __future__ import annotations

"""Small metadata-first dataset wrapper for one-file F4Mix runs."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import pandas as pd

from .core._admixpy.genotypes import read_ind


def _as_list(value: str | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


@dataclass(frozen=True)
class F4Dataset:
    prefix: Path
    individuals: pd.DataFrame
    selected_iids: tuple[str, ...]

    @property
    def sample_metadata(self) -> pd.DataFrame:
        return self.individuals[
            self.individuals["iid"].astype(str).isin(self.selected_iids)
        ].reset_index(drop=True).copy()

    @property
    def n_samples(self) -> int:
        return len(self.selected_iids)

    def select(
        self,
        *,
        populations: str | Sequence[str] | None = None,
        samples: str | Sequence[str] | None = None,
        mode: str = "union",
    ) -> "F4Dataset":
        populations = _as_list(populations)
        samples = _as_list(samples)
        if populations is None and samples is None:
            return self
        if mode not in {"union", "intersection"}:
            raise ValueError("mode must be 'union' or 'intersection'")
        current = self.sample_metadata
        if populations is not None:
            missing = sorted(set(populations) - set(current["population"].astype(str)))
            if missing:
                raise ValueError(f"Unknown populations: {missing}")
        if samples is not None:
            missing = sorted(set(samples) - set(current["iid"].astype(str)))
            if missing:
                raise ValueError(f"Unknown samples: {missing}")
        pop_mask = (
            current["population"].astype(str).isin(populations).to_numpy()
            if populations is not None
            else None
        )
        sample_mask = (
            current["iid"].astype(str).isin(samples).to_numpy()
            if samples is not None
            else None
        )
        if pop_mask is None:
            keep = sample_mask
        elif sample_mask is None:
            keep = pop_mask
        elif mode == "union":
            keep = pop_mask | sample_mask
        else:
            keep = pop_mask & sample_mask
        assert keep is not None
        selected = tuple(current.loc[keep, "iid"].astype(str))
        if not selected:
            raise ValueError("Sample selection is empty")
        return replace(self, selected_iids=selected)

    def exclude(
        self,
        *,
        populations: str | Sequence[str] | None = None,
        samples: str | Sequence[str] | None = None,
    ) -> "F4Dataset":
        populations = _as_list(populations)
        samples = _as_list(samples)
        if populations is None and samples is None:
            return self
        current = self.sample_metadata
        if populations is not None:
            missing = sorted(set(populations) - set(current["population"].astype(str)))
            if missing:
                raise ValueError(f"Unknown populations: {missing}")
        if samples is not None:
            missing = sorted(set(samples) - set(current["iid"].astype(str)))
            if missing:
                raise ValueError(f"Unknown samples: {missing}")
        remove = pd.Series(False, index=current.index)
        if populations is not None:
            remove |= current["population"].astype(str).isin(populations)
        if samples is not None:
            remove |= current["iid"].astype(str).isin(samples)
        selected = tuple(current.loc[~remove, "iid"].astype(str))
        if not selected:
            raise ValueError("Exclusion removed every sample")
        return replace(self, selected_iids=selected)


def open_genotypes(prefix: str | Path) -> F4Dataset:
    prefix = Path(prefix).expanduser()
    individuals = read_ind(Path(str(prefix) + ".ind"))
    iids = tuple(individuals["iid"].astype(str))
    if len(iids) != len(set(iids)):
        raise ValueError(f"Duplicate individual IDs found in {prefix}.ind")
    return F4Dataset(prefix, individuals, iids)
