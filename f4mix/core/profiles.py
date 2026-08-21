"""Covariance-aware f4-profile fitting

The fitted values are descriptive reference-similarity proportions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ._admixpy.genotypes import AfData, anygeno_to_afs, get_block_lengths, read_ind


DEFAULT_F4_OUTGROUP = "Chimp"


@dataclass(frozen=True)
class OptimizerDiagnostics:
    """Serializable diagnostics from one constrained optimizer call."""

    success: bool
    status: int | None
    message: str
    nit: int | None
    nfev: int | None

    @classmethod
    def from_result(cls, result: object) -> "OptimizerDiagnostics":
        return cls(
            success=bool(getattr(result, "success", False)),
            status=(
                int(result.status)
                if getattr(result, "status", None) is not None
                else None
            ),
            message=str(getattr(result, "message", "")),
            nit=(int(result.nit) if getattr(result, "nit", None) is not None else None),
            nfev=(
                int(result.nfev)
                if getattr(result, "nfev", None) is not None
                else None
            ),
        )


@dataclass(frozen=True)
class F4ProfileData:
    """Source-minus-target f4 profiles and their joint uncertainty."""

    sources: tuple[str, ...]
    features: tuple[str, ...]
    matrix: np.ndarray
    covariance: np.ndarray
    loo: np.ndarray | None = None
    fit_covariance: np.ndarray | None = None
    target_callable_snps: int | None = None
    effective_f4_snps: np.ndarray | None = None
    effective_f4_blocks: np.ndarray | None = None

    def validate(self) -> None:
        matrix = np.asarray(self.matrix, float)
        covariance = np.asarray(self.covariance, float)
        expected = (len(self.sources), len(self.features))
        if matrix.shape != expected:
            raise ValueError(
                f"f4 profile matrix has shape {matrix.shape}, expected {expected}"
            )
        flat = matrix.size
        if covariance.shape != (flat, flat):
            raise ValueError(
                "f4 profile covariance must describe the flattened source-by-feature matrix"
            )
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("f4 profile source labels must be unique")
        if len(set(self.features)) != len(self.features):
            raise ValueError("f4 profile feature labels must be unique")
        if not np.isfinite(matrix).all() or not np.isfinite(covariance).all():
            raise ValueError("f4 profiles and covariance must be finite")
        if self.loo is not None:
            loo = np.asarray(self.loo, float)
            if loo.ndim != 3 or loo.shape[:2] != expected:
                raise ValueError(
                    "leave-one-block-out profiles must be source-by-feature-by-block"
                )
        if self.fit_covariance is not None:
            fit_covariance = np.asarray(self.fit_covariance, float)
            if fit_covariance.shape != (len(self.features), len(self.features)):
                raise ValueError(
                    "fit covariance must be a feature-by-feature target covariance"
                )
            if not np.isfinite(fit_covariance).all():
                raise ValueError("fit covariance must be finite")
        if self.target_callable_snps is not None and self.target_callable_snps < 0:
            raise ValueError("target callable SNP count must be non-negative")
        for name, values in (
            ("effective f4 SNP counts", self.effective_f4_snps),
            ("effective f4 block counts", self.effective_f4_blocks),
        ):
            if values is None:
                continue
            values = np.asarray(values, float)
            if values.shape != (len(self.features),):
                raise ValueError(f"{name} must have one value per f4 feature")
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class F4ProfileFit:
    """One constrained profile fit."""

    sources: tuple[str, ...]
    weights: np.ndarray
    standard_errors: np.ndarray
    residual: np.ndarray
    chi_square: float
    iterations: int
    success: bool
    initial_optimizer: OptimizerDiagnostics | None = None
    refinement_optimizer: OptimizerDiagnostics | None = None
    target_callable_snps: int | None = None
    effective_f4_snps: np.ndarray | None = None
    effective_f4_blocks: np.ndarray | None = None

    def to_frame(self) -> pd.DataFrame:
        with np.errstate(divide="ignore", invalid="ignore"):
            z = self.weights / self.standard_errors
        return pd.DataFrame(
            {
                "component": self.sources,
                "weight": self.weights,
                "se": self.standard_errors,
                "z": z,
            }
        )


@dataclass(frozen=True)
class _PairwiseBlockStats:
    """Vectorized all-SNP f4 estimates and block-jackknife intermediates."""

    est: np.ndarray
    loo: np.ndarray
    snp_counts: np.ndarray
    influence: np.ndarray
    contributes: np.ndarray


def build_f4_profiles_for_targets(
    data: AfData,
    targets: Sequence[str],
    sources: Sequence[str],
    right: Sequence[str],
    *,
    outgroup: str = DEFAULT_F4_OUTGROUP,
    blgsize: float = 0.05,
    covariance_ridge: float = 1e-5,
    verbose: bool = True,
) -> dict[str, F4ProfileData]:
    """Build base-free, covariance-aware f4 profiles for held-out targets.

    Every source-target comparison is first measured against every unordered
    pair of Right populations.  Those redundant pairwise contrasts are then
    projected by generalized least squares into an orthonormal ``R - 1``
    dimensional Helmert basis.  No Right population is privileged as a base.

    ``outgroup`` remains in the signature for compatibility with the original
    public API, but it is not part of the base-free feature construction.
    """

    targets = tuple(str(value) for value in targets)
    sources = tuple(str(value) for value in sources)
    right = tuple(str(value) for value in right)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("At least one unique f4 profile target is required")
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("At least one unique f4 profile source is required")
    overlap = sorted(set(targets) & set(sources))
    if overlap:
        raise ValueError(f"Targets must not occur among f4 profile sources: {overlap}")
    if len(set(right)) != len(right):
        raise ValueError("Right populations must be unique")
    right_overlap = sorted((set(targets) | set(sources)) & set(right))
    if right_overlap:
        raise ValueError(
            f"Right populations must be distinct from targets and sources: {right_overlap}"
        )
    if len(right) < 2:
        raise ValueError("At least two right populations are required")
    if covariance_ridge <= 0:
        raise ValueError("covariance_ridge must be positive")

    # Retain the compatibility argument deliberately rather than allowing it to
    # influence the feature matrix.  Direct source-target f4 statistics avoid
    # the unequal-SNP-set subtraction used by the original outgroup profiles.
    _ = outgroup
    right_pairs = tuple(combinations(range(len(right)), 2))
    basis = _helmert_basis(len(right))
    edge_incidence = _pairwise_incidence(len(right), right_pairs)
    edge_design = edge_incidence @ basis
    nfeatures = basis.shape[1]
    features = tuple(f"right_contrast_{i + 1}" for i in range(nfeatures))
    target_callable = {
        target: int(
            (
                np.isfinite(data.afs[target].to_numpy(float))
                & (data.counts[target].to_numpy(float) > 0)
            ).sum()
        )
        for target in targets
    }
    # Calculate block estimates for every target in one traversal.  Building a
    # global covariance here would scale quadratically in the number of targets,
    # even though target fits never use cross-target terms.  Retain jackknife
    # influences instead and form only each target's covariance below.
    stats = _pairwise_f4_block_stats(
        data,
        targets,
        sources,
        right,
        right_pairs,
        blgsize=blgsize,
        verbose=verbose,
    )
    pairwise_shape = (len(targets), len(sources), len(right_pairs))
    all_pairwise = np.asarray(stats.est, float).reshape(pairwise_shape)
    all_pairwise_loo = (
        None
        if stats.loo is None
        else np.asarray(stats.loo, float).reshape(*pairwise_shape, -1)
    )
    all_counts = np.asarray(stats.snp_counts, float).reshape(*pairwise_shape, -1)
    per_target_stats = len(sources) * len(right_pairs)
    all_influence = np.asarray(stats.influence, float).reshape(
        len(targets), per_target_stats, -1
    )
    all_contributes = np.asarray(stats.contributes, bool).reshape(
        len(targets), per_target_stats, -1
    )

    outputs: dict[str, F4ProfileData] = {}
    for target_i, target in enumerate(targets):
        pairwise = all_pairwise[target_i]
        pairwise_covariance = _vectorized_influence_covariance(
            all_influence[target_i], all_contributes[target_i]
        )
        pairwise_loo = (
            None
            if all_pairwise_loo is None
            else all_pairwise_loo[target_i]
        )
        matrix, covariance, loo = _project_pairwise_profiles(
            pairwise,
            pairwise_covariance,
            pairwise_loo,
            edge_design,
            covariance_ridge=covariance_ridge,
        )
        counts = all_counts[target_i]
        # Every projected coordinate uses the complete pairwise system.  A
        # conservative shared coverage value is therefore more honest than
        # attributing one raw edge's SNP count to a Helmert coordinate.
        minimum_snps = float(np.min(counts.sum(axis=2)))
        minimum_blocks = float(np.min((counts > 0).sum(axis=2)))
        uniform_weights = np.full(len(sources), 1.0 / len(sources))
        collapse = np.kron(uniform_weights[None, :], np.eye(nfeatures))
        initial_fit_covariance = collapse @ covariance @ collapse.T
        out = F4ProfileData(
            sources=sources,
            features=features,
            matrix=matrix,
            covariance=covariance,
            loo=loo,
            fit_covariance=initial_fit_covariance,
            target_callable_snps=target_callable[target],
            effective_f4_snps=np.full(nfeatures, minimum_snps),
            effective_f4_blocks=np.full(nfeatures, minimum_blocks),
        )
        out.validate()
        outputs[target] = out
    return outputs


def _pairwise_f4_block_stats(
    data: AfData,
    targets: Sequence[str],
    sources: Sequence[str],
    right: Sequence[str],
    right_pairs: Sequence[tuple[int, int]],
    *,
    blgsize: float,
    verbose: bool,
) -> _PairwiseBlockStats:
    """Calculate all direct source-target/Right-pair f4s with batched products.

    The feature builder requires all population roles to be distinct. Therefore
    the finite-sample correction terms in the generic f4 engine are identically
    zero, and this kernel can evaluate the direct allele-frequency products.
    """

    targets = tuple(targets)
    sources = tuple(sources)
    right = tuple(right)
    right_pairs = tuple(right_pairs)
    required = [*targets, *sources, *right]
    if len(set(required)) != len(required):
        raise ValueError(
            "The vectorized f4 kernel requires distinct targets, sources, and Rights"
        )
    missing = [population for population in required if population not in data.afs]
    if missing:
        raise ValueError(f"Populations missing from allele-frequency table: {missing}")

    target_af = data.afs.loc[:, targets].to_numpy(float)
    source_af = data.afs.loc[:, sources].to_numpy(float)
    right_af = data.afs.loc[:, right].to_numpy(float)
    if not (len(target_af) == len(source_af) == len(right_af)):
        raise ValueError("Allele-frequency tables have inconsistent SNP counts")

    block_lengths = get_block_lengths(data.snpfile, blgsize)
    if int(block_lengths.sum()) != len(target_af):
        raise ValueError("Block lengths do not cover every SNP")
    pair_first = np.asarray([first for first, _ in right_pairs], dtype=int)
    pair_second = np.asarray([second for _, second in right_pairs], dtype=int)
    nleft = len(targets) * len(sources)
    nstats = nleft * len(right_pairs)
    block_estimates = np.full((nstats, len(block_lengths)), np.nan, dtype=float)
    snp_counts = np.zeros_like(block_estimates)

    start = 0
    progress_step = max(1, len(block_lengths) // 20)
    for block_i, block_size in enumerate(block_lengths):
        stop = start + int(block_size)
        if verbose and (
            block_i == 0
            or block_i + 1 == len(block_lengths)
            or (block_i + 1) % progress_step == 0
        ):
            end = "\n" if block_i + 1 == len(block_lengths) else ""
            print(
                f"\rComputing base-free f4 block {block_i + 1}/{len(block_lengths)}: "
                f"SNP rows {start + 1}-{stop}      ",
                end=end,
                flush=True,
            )

        # Flatten target-major/source-minor differences so the resulting row
        # order is target, source, Right pair.  Missing values are zeroed only
        # for the matrix product; callable counts use the matching finite masks.
        left = (
            source_af[start:stop, None, :]
            - target_af[start:stop, :, None]
        ).reshape(stop - start, nleft)
        right_block = right_af[start:stop]
        right_difference = (
            right_block[:, pair_first] - right_block[:, pair_second]
        )
        left_valid = np.isfinite(left)
        right_valid = np.isfinite(right_difference)
        left_values = np.where(left_valid, left, 0.0)
        right_values = np.where(right_valid, right_difference, 0.0)

        numerator = left_values.T @ right_values
        counts = left_valid.astype(float).T @ right_valid.astype(float)
        means = np.full_like(numerator, np.nan)
        np.divide(numerator, counts, out=means, where=counts > 0)
        block_estimates[:, block_i] = means.reshape(-1)
        snp_counts[:, block_i] = counts.reshape(-1)
        start = stop

    est, loo, influence, contributes = _vectorized_count_jackknife(
        block_estimates, snp_counts
    )
    return _PairwiseBlockStats(est, loo, snp_counts, influence, contributes)


def _vectorized_count_jackknife(
    block_estimates: np.ndarray,
    snp_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Count-weighted jackknife for many statistics simultaneously."""

    estimates = np.asarray(block_estimates, float)
    counts = np.asarray(snp_counts, float)
    if estimates.shape != counts.shape or estimates.ndim != 2:
        raise ValueError("Block estimates and SNP counts must be matching matrices")
    contributes = np.isfinite(estimates) & np.isfinite(counts) & (counts > 0)
    weighted = np.where(contributes, estimates * counts, 0.0)
    effective_counts = np.where(contributes, counts, 0.0)
    total_counts = effective_counts.sum(axis=1)
    totals = np.full(estimates.shape[0], np.nan, dtype=float)
    np.divide(weighted.sum(axis=1), total_counts, out=totals, where=total_counts > 0)

    loo = np.broadcast_to(totals[:, None], estimates.shape).copy()
    remaining = total_counts[:, None] - effective_counts
    can_delete = contributes & (remaining > 0)
    deleted_totals = totals[:, None] * total_counts[:, None] - weighted
    np.divide(deleted_totals, remaining, out=loo, where=can_delete)
    loo[contributes & ~can_delete] = np.nan

    influence = np.zeros_like(estimates)
    enough_blocks = can_delete.sum(axis=1) >= 2
    eligible = can_delete & enough_blocks[:, None]
    h = np.zeros_like(counts)
    np.divide(
        total_counts[:, None],
        effective_counts,
        out=h,
        where=eligible,
    )
    tau = h * totals[:, None] - (h - 1.0) * loo
    scaled_influence = np.zeros_like(influence)
    np.divide(
        tau - totals[:, None],
        np.sqrt(np.maximum(h - 1.0, 0.0)),
        out=scaled_influence,
        where=eligible,
    )
    influence[eligible] = scaled_influence[eligible]
    influence[contributes & ~enough_blocks[:, None]] = np.nan
    influence[total_counts <= 0] = np.nan
    return totals, loo, influence, contributes


def _vectorized_influence_covariance(
    influence: np.ndarray,
    contributes: np.ndarray,
) -> np.ndarray:
    """Pairwise-overlap covariance for a matrix of jackknife influences."""

    influence = np.asarray(influence, float)
    contributes = np.asarray(contributes, bool)
    if influence.shape != contributes.shape or influence.ndim != 2:
        raise ValueError("Influences and contribution masks must be matching matrices")
    keep = contributes & np.isfinite(influence)
    values = np.where(keep, influence, 0.0)
    overlaps = keep.astype(float) @ keep.astype(float).T
    products = values @ values.T
    covariance = np.full(products.shape, np.nan, dtype=float)
    np.divide(products, overlaps, out=covariance, where=overlaps >= 2)
    return (covariance + covariance.T) / 2.0


def _helmert_basis(size: int) -> np.ndarray:
    """Return orthonormal columns spanning vectors whose entries sum to zero."""

    if size < 2:
        raise ValueError("A Helmert basis requires at least two populations")
    basis = np.zeros((size, size - 1), dtype=float)
    for column in range(size - 1):
        denominator = np.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    return basis


def _pairwise_incidence(
    size: int,
    pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Return oriented complete-graph edges matching f4(...; Ri, Rj)."""

    incidence = np.zeros((len(pairs), size), dtype=float)
    for edge, (first, second) in enumerate(pairs):
        if first < 0 or second >= size or first >= second:
            raise ValueError("Right-population pairs must satisfy 0 <= first < second")
        incidence[edge, first] = 1.0
        incidence[edge, second] = -1.0
    return incidence


def _project_pairwise_profiles(
    matrix: np.ndarray,
    covariance: np.ndarray,
    loo: np.ndarray | None,
    edge_design: np.ndarray,
    *,
    covariance_ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """GLS-project redundant pairwise Right contrasts into ``R - 1`` axes."""

    matrix = np.asarray(matrix, float)
    covariance = np.asarray(covariance, float)
    edge_design = np.asarray(edge_design, float)
    if matrix.ndim != 2 or edge_design.ndim != 2:
        raise ValueError("Pairwise profiles and their edge design must be matrices")
    nsources, nedges = matrix.shape
    if edge_design.shape[0] != nedges:
        raise ValueError("The edge design must have one row per pairwise contrast")
    raw_size = matrix.size
    if covariance.shape != (raw_size, raw_size):
        raise ValueError("Pairwise covariance has an unexpected shape")
    if not np.isfinite(matrix).all() or not np.isfinite(covariance).all():
        raise ValueError("Pairwise profiles and covariance must be finite")

    design = np.kron(np.eye(nsources), edge_design)
    precision = _regularized_inverse(covariance, covariance_ridge)
    information = design.T @ precision @ design
    information = (information + information.T) / 2.0
    projection = np.linalg.pinv(information, rcond=1e-12) @ design.T @ precision

    projected = (projection @ matrix.reshape(-1)).reshape(
        nsources, edge_design.shape[1]
    )
    projected_covariance = projection @ covariance @ projection.T
    projected_covariance = (projected_covariance + projected_covariance.T) / 2.0
    projected_loo = None
    if loo is not None:
        loo = np.asarray(loo, float)
        if loo.ndim != 3 or loo.shape[:2] != matrix.shape:
            raise ValueError("Pairwise leave-one-block-out profiles have an unexpected shape")
        projected_loo = (projection @ loo.reshape(raw_size, -1)).reshape(
            nsources, edge_design.shape[1], -1
        )
    return projected, projected_covariance, projected_loo


def build_sample_mapping(
    dataset_prefix: str | Path,
    component_sources: Mapping[str, Sequence[str]],
) -> tuple[list[str], list[str], dict[str, int]]:
    """Map exact .ind source populations to pooled component labels."""

    ind_path = Path(str(dataset_prefix) + ".ind")
    if not ind_path.exists():
        raise FileNotFoundError(f"Individual file not found: {ind_path}")
    individuals = read_ind(ind_path)
    available = set(individuals["population"].astype(str))

    source_owner: dict[str, str] = {}
    for component, sources in component_sources.items():
        if not sources:
            raise ValueError(f"Component {component!r} has no source populations")
        for source in sources:
            if source in source_owner:
                raise ValueError(
                    f"Source population {source!r} is assigned to both "
                    f"{source_owner[source]!r} and {component!r}"
                )
            source_owner[source] = component

    missing = sorted(set(source_owner) - available)
    if missing:
        raise ValueError(f"Population labels missing from {ind_path}: {missing}")

    iids: list[str] = []
    labels: list[str] = []
    sample_counts: dict[str, int] = {}
    for component, sources in component_sources.items():
        selected = individuals[individuals["population"].isin(sources)]
        if selected.empty:
            raise ValueError(f"No individuals selected for component {component!r}")
        component_iids = selected["iid"].astype(str).tolist()
        iids.extend(component_iids)
        labels.extend([component] * len(component_iids))
        sample_counts[component] = len(component_iids)
    return iids, labels, sample_counts


def load_f4_batch_analysis_data(
    dataset_prefix: str | Path,
    targets: Mapping[str, Sequence[str]],
    *,
    component_sources: Mapping[str, Sequence[str]],
    right: Sequence[str],
    outgroup: str = DEFAULT_F4_OUTGROUP,
    chunk_size: int = 10_000,
    verbose: bool = True,
) -> tuple[AfData, dict[str, int]]:
    """Load pooled sources, multiple held-out targets and shared f4 anchors."""

    if not targets or len(set(targets)) != len(targets):
        raise ValueError("At least one unique target label is required")
    reserved = set(component_sources) | set(right)
    bad_labels = sorted(set(targets) & reserved)
    if bad_labels:
        raise ValueError(f"Target labels overlap source or anchor labels: {bad_labels}")
    reference_iids, reference_labels, sample_counts = build_sample_mapping(
        dataset_prefix, component_sources
    )
    ind_path = Path(str(dataset_prefix) + ".ind")
    individuals = read_ind(ind_path)
    iid_to_population = dict(
        zip(individuals["iid"].astype(str), individuals["population"].astype(str))
    )
    target_iids = [
        str(iid) for values in targets.values() for iid in values
    ]
    if len(target_iids) != len(set(target_iids)):
        raise ValueError("Target individual IDs must be unique across target labels")
    missing_targets = sorted(set(target_iids) - set(iid_to_population))
    if missing_targets:
        raise ValueError(f"Target individuals missing from {ind_path}: {missing_targets}")
    overlap = sorted(set(target_iids) & set(reference_iids))
    if overlap:
        raise ValueError(f"Held-out target individuals occur in the source panel: {overlap}")

    right_iids: list[str] = []
    right_labels: list[str] = []
    available = set(individuals["population"].astype(str))
    analysis_populations = list(right)
    missing_right = [population for population in analysis_populations if population not in available]
    if missing_right:
        raise ValueError(f"Right populations missing from {ind_path}: {missing_right}")
    for population in analysis_populations:
        selected = individuals[individuals["population"] == population]
        right_iids.extend(selected["iid"].astype(str).tolist())
        right_labels.extend([str(population)] * len(selected))

    iid_counts = Counter(reference_iids + target_iids + right_iids)
    duplicate_iids = sorted(iid for iid, count in iid_counts.items() if count > 1)
    if duplicate_iids:
        raise ValueError(f"Source, target and right samples overlap: {duplicate_iids}")

    target_labels = [
        label for label, values in targets.items() for _ in values
    ]
    iids = reference_iids + target_iids + right_iids
    labels = reference_labels + target_labels + right_labels
    data = anygeno_to_afs(
        dataset_prefix,
        inds=iids,
        pops=labels,
        # Mixed modern and ancient panels require per-sample ploidy detection.
        adjust_pseudohaploid=1_000,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    wanted = [*component_sources, *targets, *right]
    missing_columns = [population for population in wanted if population not in data.afs]
    if missing_columns:
        raise ValueError(f"Loaded f4 data are missing populations: {missing_columns}")
    return (
        AfData(
            data.afs.loc[:, wanted],
            data.counts.loc[:, wanted],
            data.snpfile,
        ),
        sample_counts,
    )


def _covariance_scale(covariance: np.ndarray) -> float:
    covariance = np.asarray(covariance, float)
    scale = float(np.trace(covariance) / len(covariance))
    if not np.isfinite(scale) or scale <= 0:
        scale = max(float(np.max(np.abs(covariance))), 1.0)
    return scale


def _regularized_covariance(
    covariance: np.ndarray,
    ridge: float,
    *,
    scale: float | None = None,
) -> np.ndarray:
    covariance = np.asarray(covariance, float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Residual covariance must be square")
    covariance = (covariance + covariance.T) / 2.0
    scale = _covariance_scale(covariance) if scale is None else float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Covariance regularization scale must be positive")
    adjusted = covariance.copy()
    adjusted[np.diag_indices_from(adjusted)] += ridge * scale
    # Numerical covariance estimates can have tiny negative eigenvalues. Make
    # the matrix positive definite so the Gaussian objective has a valid
    # inverse and log-determinant.
    eigenvalues, eigenvectors = np.linalg.eigh(adjusted)
    floor = max(ridge * scale, np.finfo(float).eps)
    if eigenvalues.min() < floor:
        adjusted = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    return adjusted


def _regularized_inverse(
    covariance: np.ndarray,
    ridge: float,
    *,
    scale: float | None = None,
) -> np.ndarray:
    adjusted = _regularized_covariance(covariance, ridge, scale=scale)
    return np.linalg.pinv(adjusted, rcond=1e-12)


def _residual_covariance(
    full_covariance: np.ndarray,
    weights: np.ndarray,
    nfeatures: int,
) -> np.ndarray:
    transform = np.kron(np.asarray(weights, float)[None, :], np.eye(nfeatures))
    return transform @ full_covariance @ transform.T


def _validate_group_sums(
    sources: Sequence[str],
    group_sums: Mapping[str, tuple[Sequence[str], float]] | None,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    if group_sums is None:
        return None
    source_index = {source: i for i, source in enumerate(sources)}
    groups: list[np.ndarray] = []
    values: list[float] = []
    used: set[str] = set()
    for group, (members, value) in group_sums.items():
        members = tuple(members)
        if not members:
            raise ValueError(f"Constraint group {group!r} is empty")
        missing = sorted(set(members) - set(source_index))
        if missing:
            raise ValueError(f"Constraint group {group!r} has missing members: {missing}")
        overlap = sorted(set(members) & used)
        if overlap:
            raise ValueError(f"Constraint groups overlap at sources: {overlap}")
        if not np.isfinite(value) or value < 0 or value > 1:
            raise ValueError(f"Constraint group {group!r} has an invalid mass")
        used.update(members)
        groups.append(np.asarray([source_index[source] for source in members], int))
        values.append(float(value))
    if used != set(sources):
        missing = sorted(set(sources) - used)
        raise ValueError(f"Constraint groups do not cover sources: {missing}")
    values_array = np.asarray(values, float)
    if not np.isclose(values_array.sum(), 1.0, atol=1e-8):
        raise ValueError("Constraint group masses must sum to one")
    return groups, values_array


def _solve_weights(
    matrix: np.ndarray,
    precision: np.ndarray,
    *,
    group_sums: Mapping[str, tuple[Sequence[str], float]] | None,
    sources: Sequence[str],
) -> tuple[np.ndarray, OptimizerDiagnostics]:
    matrix = np.asarray(matrix, float)
    hessian = matrix @ precision @ matrix.T
    hessian = (hessian + hessian.T) / 2.0
    # Multiplying a quadratic objective by a positive constant does not alter
    # its optimum.  Global panels can nevertheless produce Hessians many
    # orders of magnitude larger than the regional model, which makes
    # SLSQP's absolute stopping tests report false failures and can discard
    # otherwise valid jackknife replicates.  Normalize the objective to an
    # order-one scale before solving.
    objective_scale = float(np.max(np.abs(hessian)))
    if np.isfinite(objective_scale) and objective_scale > 0.0:
        hessian = hessian / objective_scale
    groups_and_values = _validate_group_sums(sources, group_sums)
    constraints: list[dict[str, object]] = []
    if groups_and_values is None:
        initial = np.full(len(sources), 1.0 / len(sources))
        constraints.append(
            {
                "type": "eq",
                "fun": lambda w: float(w.sum() - 1.0),
                "jac": lambda w: np.ones_like(w),
            }
        )
    else:
        groups, values = groups_and_values
        initial = np.zeros(len(sources), dtype=float)
        for indices, value in zip(groups, values):
            initial[indices] = value / len(indices)
        # The final group equality follows from the simplex and preceding
        # groups, so omit it to avoid a redundant equality Jacobian.
        constraints.append(
            {
                "type": "eq",
                "fun": lambda w: float(w.sum() - 1.0),
                "jac": lambda w: np.ones_like(w),
            }
        )
        for indices, value in zip(groups[:-1], values[:-1]):
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda w, idx=indices, target=value: float(
                        w[idx].sum() - target
                    ),
                    "jac": lambda w, idx=indices: np.isin(
                        np.arange(len(w)), idx
                    ).astype(float),
                }
            )

    result = minimize(
        lambda w: 0.5 * float(w @ hessian @ w),
        initial,
        jac=lambda w: hessian @ w,
        bounds=[(0.0, 1.0)] * len(sources),
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    weights = np.clip(np.asarray(result.x, float), 0.0, 1.0)
    # SLSQP can leave tiny equality drift after clipping. Renormalizing within
    # constrained groups preserves the hierarchy exactly.
    if groups_and_values is None:
        weights /= weights.sum()
    else:
        groups, values = groups_and_values
        for indices, value in zip(groups, values):
            total = weights[indices].sum()
            weights[indices] = (
                value / len(indices) if total <= 0 else weights[indices] * value / total
            )
    return weights, OptimizerDiagnostics.from_result(result)


def _solve_full_gaussian_weights(
    matrix: np.ndarray,
    full_covariance: np.ndarray,
    nfeatures: int,
    *,
    initial: np.ndarray,
    covariance_ridge: float,
    covariance_scale: float,
    group_sums: Mapping[str, tuple[Sequence[str], float]] | None,
    sources: Sequence[str],
) -> tuple[np.ndarray, OptimizerDiagnostics, bool]:
    """Optimize the Gaussian residual likelihood on the source simplex."""

    matrix = np.asarray(matrix, float)
    initial = np.asarray(initial, float)
    groups_and_values = _validate_group_sums(sources, group_sums)
    constraints: list[dict[str, object]] = [
        {
            "type": "eq",
            "fun": lambda w: float(w.sum() - 1.0),
            "jac": lambda w: np.ones_like(w),
        }
    ]
    if groups_and_values is not None:
        groups, values = groups_and_values
        for indices, value in zip(groups[:-1], values[:-1]):
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda w, idx=indices, target=value: float(
                        w[idx].sum() - target
                    ),
                    "jac": lambda w, idx=indices: np.isin(
                        np.arange(len(w)), idx
                    ).astype(float),
                }
            )

    def objective(weights: np.ndarray) -> float:
        residual = weights @ matrix
        covariance = _regularized_covariance(
            _residual_covariance(full_covariance, weights, nfeatures),
            covariance_ridge,
            scale=covariance_scale,
        )
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0 or not np.isfinite(logdet):
            return float("inf")
        try:
            whitened = np.linalg.solve(covariance, residual)
        except np.linalg.LinAlgError:
            return float("inf")
        value = 0.5 * (float(residual @ whitened) + float(logdet))
        return value if np.isfinite(value) else float("inf")

    result = minimize(
        objective,
        initial,
        bounds=[(0.0, 1.0)] * len(sources),
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    weights = np.clip(np.asarray(result.x, float), 0.0, 1.0)
    if groups_and_values is None:
        weights /= weights.sum()
    else:
        groups, values = groups_and_values
        for indices, value in zip(groups, values):
            total = weights[indices].sum()
            weights[indices] = (
                value / len(indices) if total <= 0 else weights[indices] * value / total
            )
    diagnostics = OptimizerDiagnostics.from_result(result)
    return weights, diagnostics, bool(diagnostics.success and np.isfinite(objective(weights)))


def fit_f4_profiles(
    profile: F4ProfileData,
    *,
    group_sums: Mapping[str, tuple[Sequence[str], float]] | None = None,
    covariance_ridge: float = 1e-5,
    max_gls_iterations: int = 20,
    tolerance: float = 1e-8,
    jackknife: bool = True,
) -> F4ProfileFit:
    """Fit simplex weights with a full covariance Gaussian refinement."""

    profile.validate()
    if covariance_ridge <= 0:
        raise ValueError("covariance_ridge must be positive")
    if max_gls_iterations <= 0:
        raise ValueError("max_gls_iterations must be positive")
    nfeatures = len(profile.features)
    precision = (
        np.eye(nfeatures)
        if profile.fit_covariance is None
        else _regularized_inverse(profile.fit_covariance, covariance_ridge)
    )
    weights, initial_optimizer = _solve_weights(
        profile.matrix,
        precision,
        group_sums=group_sums,
        sources=profile.sources,
    )
    success = initial_optimizer.success
    refinement_optimizer = None
    iterations = 1
    if profile.fit_covariance is not None:
        # Full Gaussian refinement: account for both the residual covariance
        # and its determinant, so noisy source combinations are not rewarded.
        covariance_scale = _covariance_scale(profile.fit_covariance)
        revised, refinement_optimizer, solved = _solve_full_gaussian_weights(
            profile.matrix,
            profile.covariance,
            nfeatures,
            initial=weights,
            covariance_ridge=covariance_ridge,
            covariance_scale=covariance_scale,
            group_sums=group_sums,
            sources=profile.sources,
        )
        success &= solved
        weights = revised
        precision = _regularized_inverse(
            _residual_covariance(profile.covariance, weights, nfeatures),
            covariance_ridge,
            scale=covariance_scale,
        )
        iterations = 2
    else:
        iteration_limit = max_gls_iterations
        for iterations in range(2, iteration_limit + 1):
            residual_covariance = _residual_covariance(
                profile.covariance, weights, nfeatures
            )
            precision = _regularized_inverse(residual_covariance, covariance_ridge)
            revised, refinement_optimizer = _solve_weights(
                profile.matrix,
                precision,
                group_sums=group_sums,
                sources=profile.sources,
            )
            success &= refinement_optimizer.success
            change = float(np.max(np.abs(revised - weights)))
            weights = revised
            if change <= tolerance:
                break

    residual = weights @ profile.matrix
    chi_square = float(residual @ precision @ residual)
    standard_errors = np.full(len(weights), np.nan)
    if jackknife and profile.loo is not None and profile.loo.shape[2] >= 2:
        jackknife_weights: list[np.ndarray] = []
        for block in range(profile.loo.shape[2]):
            matrix = profile.loo[:, :, block]
            if not np.isfinite(matrix).all():
                continue
            fitted, jackknife_optimizer = _solve_weights(
                matrix,
                precision,
                group_sums=group_sums,
                sources=profile.sources,
            )
            if jackknife_optimizer.success and np.isfinite(fitted).all():
                jackknife_weights.append(fitted)
        if len(jackknife_weights) >= 2:
            values = np.asarray(jackknife_weights, float)
            mean = values.mean(axis=0)
            variance = (len(values) - 1) / len(values) * np.sum(
                (values - mean) ** 2, axis=0
            )
            standard_errors = np.sqrt(np.maximum(variance, 0.0))

    return F4ProfileFit(
        sources=profile.sources,
        weights=weights,
        standard_errors=standard_errors,
        residual=residual,
        chi_square=chi_square,
        iterations=iterations,
        success=success,
        initial_optimizer=initial_optimizer,
        refinement_optimizer=refinement_optimizer,
        target_callable_snps=profile.target_callable_snps,
        effective_f4_snps=profile.effective_f4_snps,
        effective_f4_blocks=profile.effective_f4_blocks,
    )
