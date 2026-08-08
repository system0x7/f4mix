from __future__ import annotations

"""Public one-script API for sample-wise ancient-reference f4 fits."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence
import warnings

import numpy as np
import pandas as pd

from .core.profiles import (
    F4ProfileFit,
    build_f4_profiles_for_targets,
    fit_f4_profiles,
)
from .core._admixpy.genotypes import anygeno_to_afs
from .dataset import F4Dataset


@dataclass(frozen=True)
class F4FitConfig:
    dataset_prefix: Path
    targets: dict[str, tuple[str, ...]]
    component_sources: dict[str, tuple[str, ...]]
    right: tuple[str, ...]
    outgroup: str = "Chimp"
    blgsize: float = 0.05
    chunk_size: int = 250_000
    covariance_ridge: float = 1e-5
    min_effective_f4_snps_warning: int | None = 50_000
    verbose: bool = True

    def as_jsonable(self) -> dict[str, object]:
        result = asdict(self)
        result["dataset_prefix"] = str(self.dataset_prefix)
        result["targets"] = {key: list(value) for key, value in self.targets.items()}
        result["component_sources"] = {
            key: list(value) for key, value in self.component_sources.items()
        }
        result["right"] = list(self.right)
        return result


@dataclass(frozen=True)
class F4FitResult:
    config: F4FitConfig
    fits: dict[str, F4ProfileFit]
    source_sample_counts: dict[str, int]
    features: tuple[str, ...]

    def weights_frame(self) -> pd.DataFrame:
        """Return fitted weights in one row per target and one source per column."""

        return self._wide_metric_frame("weights")

    def weights_se_frame(self) -> pd.DataFrame:
        """Return jackknife standard errors in the weights-table layout."""

        return self._wide_metric_frame("standard_errors")

    def weights_z_frame(self) -> pd.DataFrame:
        """Return weight z-scores in the weights-table layout."""

        rows: list[dict[str, object]] = []
        for target, fit in self.fits.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                values = fit.weights / fit.standard_errors
            rows.append(
                {
                    "target": target,
                    **dict(zip(fit.sources, values, strict=True)),
                }
            )
        return pd.DataFrame(
            rows,
            columns=["target", *self.config.component_sources],
        )

    def _wide_metric_frame(self, metric: str) -> pd.DataFrame:
        sources = list(self.config.component_sources)
        rows: list[dict[str, object]] = []
        for target, fit in self.fits.items():
            values = getattr(fit, metric)
            rows.append(
                {
                    "target": target,
                    **dict(zip(fit.sources, values, strict=True)),
                }
            )
        return pd.DataFrame(rows, columns=["target", *sources])

    def weights_long_frame(self) -> pd.DataFrame:
        """Return weights and jackknife diagnostics in long form."""

        rows: list[dict[str, object]] = []
        for target, fit in self.fits.items():
            values = fit.to_frame()
            for row in values.to_dict(orient="records"):
                rows.append(
                    {
                        "target": target,
                        **row,
                        "chi_square": fit.chi_square,
                    }
                )
        return pd.DataFrame(rows)

    def summary_frame(self) -> pd.DataFrame:
        rows = []
        for target, fit in self.fits.items():
            effective = fit.effective_f4_snps
            effective_blocks = fit.effective_f4_blocks
            rows.append(
                {
                    "target": target,
                    "chi_square": fit.chi_square,
                    "residual_norm": float(np.linalg.norm(fit.residual)),
                    "success": fit.success,
                    "iterations": fit.iterations,
                    "target_callable_snps": fit.target_callable_snps,
                    "min_effective_f4_snps": (
                        int(np.min(effective)) if effective is not None else None
                    ),
                    "median_effective_f4_snps": (
                        float(np.median(effective)) if effective is not None else None
                    ),
                    "min_effective_f4_blocks": (
                        int(np.min(effective_blocks))
                        if effective_blocks is not None
                        else None
                    ),
                }
            )
        return pd.DataFrame(rows)

    def save(self, output_directory: str | Path, *, overwrite: bool = False) -> Path:
        """Write wide weights, uncertainty tables, summaries, and metadata."""

        output = Path(output_directory)
        if output.exists() and not overwrite:
            raise FileExistsError(f"Output directory already exists: {output}")
        output.mkdir(parents=True, exist_ok=True)
        self.weights_frame().to_csv(output / "weights.tsv", sep="\t", index=False)
        self.weights_se_frame().to_csv(output / "weights_se.tsv", sep="\t", index=False)
        self.weights_z_frame().to_csv(output / "weights_z.tsv", sep="\t", index=False)
        self.summary_frame().to_csv(output / "targets.tsv", sep="\t", index=False)
        metadata = {
            "config": self.config.as_jsonable(),
            "source_sample_counts": self.source_sample_counts,
            "features": list(self.features),
            "targets": len(self.fits),
            "sources": list(self.config.component_sources),
        }
        (output / "run.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        return output


def _source_map(
    sources: Mapping[str, Sequence[str]] | Sequence[str],
) -> dict[str, tuple[str, ...]]:
    if isinstance(sources, Mapping):
        return {
            str(name): tuple(str(population) for population in populations)
            for name, populations in sources.items()
        }
    return {str(source): (str(source),) for source in sources}


def _target_map(
    data: F4Dataset,
    *,
    target_populations: Sequence[str] | None,
    targets: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    if target_populations is not None and targets is not None:
        raise ValueError("Provide target_populations or targets, not both")
    if targets is not None:
        output = {
            str(label): tuple(str(iid) for iid in iids)
            for label, iids in targets.items()
        }
        if not output or any(not label or not iids for label, iids in output.items()):
            raise ValueError("targets must contain non-empty labels and sample IDs")
        return output
    if target_populations is None:
        raise ValueError("Provide target_populations or targets")
    wanted = set(str(population) for population in target_populations)
    selected = data.sample_metadata
    missing = sorted(wanted - set(selected["population"].astype(str)))
    if missing:
        raise ValueError(f"Target populations are not selected: {missing}")
    return {
        f"{population}:{iid}": (str(iid),)
        for iid, population in zip(selected["iid"], selected["population"])
        if str(population) in wanted
    }


class F4Model:
    """Ancient-reference f4 model used directly from a run script."""

    def __init__(
        self,
        *,
        sources: Mapping[str, Sequence[str]] | Sequence[str],
        right: Sequence[str],
        outgroup: str = "Chimp",
        blgsize: float = 0.05,
        chunk_size: int = 250_000,
        covariance_ridge: float = 1e-5,
        min_effective_f4_snps_warning: int | None = 50_000,
        verbose: bool = True,
    ) -> None:
        self.sources = _source_map(sources)
        self.right = tuple(str(value) for value in right)
        self.outgroup = str(outgroup)
        self.blgsize = float(blgsize)
        self.chunk_size = int(chunk_size)
        self.covariance_ridge = float(covariance_ridge)
        if (
            min_effective_f4_snps_warning is not None
            and min_effective_f4_snps_warning <= 0
        ):
            raise ValueError("min_effective_f4_snps_warning must be positive or None")
        self.min_effective_f4_snps_warning = min_effective_f4_snps_warning
        self.verbose = bool(verbose)

    def fit(
        self,
        data: F4Dataset,
        *,
        target_populations: Sequence[str] | None = None,
        targets: Mapping[str, Sequence[str]] | None = None,
    ) -> F4FitResult:
        target_map = _target_map(
            data,
            target_populations=target_populations,
            targets=targets,
        )
        selected = data.sample_metadata
        population_to_iids = {
            population: selected.loc[
                selected["population"].astype(str) == population, "iid"
            ].astype(str).tolist()
            for population in set(self.right) | {self.outgroup}
        }
        source_iids: list[str] = []
        source_labels: list[str] = []
        source_counts: dict[str, int] = {}
        for component, populations in self.sources.items():
            iids = [
                iid
                for population in populations
                for iid in selected.loc[
                    selected["population"].astype(str) == population, "iid"
                ].astype(str)
            ]
            if not iids:
                raise ValueError(f"No selected source samples for component {component!r}")
            source_iids.extend(iids)
            source_labels.extend([component] * len(iids))
            source_counts[component] = len(iids)

        target_iids = [iid for iids in target_map.values() for iid in iids]
        all_iids = source_iids + target_iids
        if len(all_iids) != len(set(all_iids)):
            raise ValueError("Source and target samples overlap")
        target_label_vector = [
            label for label, iids in target_map.items() for _ in iids
        ]
        for label, iids in target_map.items():
            missing = sorted(set(iids) - set(selected["iid"].astype(str)))
            if missing:
                raise ValueError(f"Target samples are not selected for {label!r}: {missing}")

        anchor_iids: list[str] = []
        anchor_labels: list[str] = []
        for population in (self.outgroup, *self.right):
            iids = population_to_iids.get(population, [])
            if not iids:
                raise ValueError(f"No selected anchor samples for population {population!r}")
            anchor_iids.extend(iids)
            anchor_labels.extend([population] * len(iids))

        duplicate_iids = set(source_iids) & (set(target_iids) | set(anchor_iids))
        if duplicate_iids:
            raise ValueError(f"Source, target and anchor samples overlap: {sorted(duplicate_iids)}")

        iids = source_iids + target_iids + anchor_iids
        labels = source_labels + target_label_vector + anchor_labels
        af_data = anygeno_to_afs(
            data.prefix,
            inds=iids,
            pops=labels,
            adjust_pseudohaploid=1_000,
            chunk_size=self.chunk_size,
            verbose=self.verbose,
        )
        source_names = tuple(self.sources)
        profiles = build_f4_profiles_for_targets(
            af_data,
            tuple(target_map),
            source_names,
            self.right,
            outgroup=self.outgroup,
            blgsize=self.blgsize,
            verbose=self.verbose,
        )
        if self.min_effective_f4_snps_warning is not None:
            threshold = self.min_effective_f4_snps_warning
            for target, profile in profiles.items():
                if profile.effective_f4_snps is None:
                    continue
                minimum = int(np.min(profile.effective_f4_snps))
                if minimum < threshold:
                    warnings.warn(
                        f"{target}: only {minimum:,} effective f4 SNPs "
                        f"(minimum across {len(profile.features)} f4 features; "
                        f"warning threshold {threshold:,})",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        fits: dict[str, F4ProfileFit] = {
            target: fit_f4_profiles(
                profile,
                covariance_ridge=self.covariance_ridge,
                jackknife=True,
            )
            for target, profile in profiles.items()
        }
        config = F4FitConfig(
            dataset_prefix=data.prefix,
            targets=target_map,
            component_sources=self.sources,
            right=self.right,
            outgroup=self.outgroup,
            blgsize=self.blgsize,
            chunk_size=self.chunk_size,
            covariance_ridge=self.covariance_ridge,
            min_effective_f4_snps_warning=self.min_effective_f4_snps_warning,
            verbose=self.verbose,
        )
        return F4FitResult(
            config=config,
            fits=fits,
            source_sample_counts=source_counts,
            features=next(iter(profiles.values())).features,
        )
