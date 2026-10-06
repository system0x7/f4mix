"""Direct block-jackknifed f4 statistics on allele-frequency tables.

Reused from the admixpy package.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from .genotypes import AfData, get_block_lengths


def _log(message: str, verbose: bool):
    if verbose:
        print(message, flush=True)


def _log_block(stat: str, b: int, total: int, start: int, stop: int, verbose: bool) -> None:
    # Single carriage-returned status line, updated ~20 times across the run.
    if not verbose or total <= 0:
        return
    step = max(1, total // 20)
    is_last = b == total - 1
    if b == 0 or is_last or (b + 1) % step == 0:
        end = "\n" if is_last else ""
        print(
            f"\rComputing {stat} block {b + 1}/{total}: SNP rows {start + 1}-{stop}      ",
            end=end,
            flush=True,
        )


def _format_number(x, decimals: int) -> str:
    if not np.isfinite(x):
        return "NaN"
    rounded = float(np.round(x, decimals))
    return np.format_float_positional(rounded, precision=decimals, fractional=True, trim="-")


def _format_significant(x, digits: int = 6) -> str:
    if not np.isfinite(x):
        return "NaN"
    return f"{x:.{digits}g}"


def _format_pvalue(x) -> str:
    if not np.isfinite(x):
        return "NaN"
    if x == 0:
        return "0"
    if abs(x) < 0.001:
        return f"{x:.3g}"
    return _format_number(x, 3)


def format_fstats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-formatted copy of an f-statistics result frame."""
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        if col == "p":
            out[col] = out[col].map(_format_pvalue)
        elif col == "z":
            out[col] = out[col].map(lambda x: _format_number(x, 2))
        elif col in {"est", "se"}:
            out[col] = out[col].map(_format_significant)
    return out


class FStatsFrame(pd.DataFrame):
    """DataFrame that keeps raw numeric values but displays f-stats compactly."""

    @property
    def _constructor(self):
        return FStatsFrame

    def __repr__(self) -> str:
        return format_fstats(pd.DataFrame(self)).to_string()

    def _repr_html_(self):
        return None


@dataclass
class BlockStats:
    rows: pd.DataFrame
    blocks: np.ndarray | None
    block_lengths: np.ndarray
    stat: str
    loo: np.ndarray | None = None
    est: np.ndarray | None = None
    cov: np.ndarray | None = None
    variances: np.ndarray | None = None

    @property
    def se(self) -> np.ndarray:
        if self.cov is None:
            if self.variances is None:
                return np.full(len(self.rows), np.nan)
            return np.sqrt(np.asarray(self.variances, float))
        return np.sqrt(np.diag(self.cov))

    @property
    def z(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.est / self.se

    @property
    def p(self) -> np.ndarray:
        return np.array([math.erfc(abs(z) / math.sqrt(2)) if np.isfinite(z) else float("nan") for z in self.z])

    def to_frame(self, round_z: int | None = None, round_p: int | None = None) -> pd.DataFrame:
        out = self.rows.copy()
        ratio_num = getattr(self, "ratio_num", None)
        # In allsnps mode the per-stat per-block SNP counts are attached as
        # `snp_counts`; in that case report each stat with its own block sizes
        # (matches admixtools::jack_dat_stats, the formula used by qpdstat-allsnps).
        snp_counts = getattr(self, "snp_counts", None)
        if ratio_num is not None:
            est_vec = np.asarray(self.est, float)
            se_vec = self.se
            with np.errstate(invalid="ignore", divide="ignore"):
                z = est_vec / se_vec
        elif snp_counts is not None and self.blocks is not None:
            n = len(self.rows)
            est_vec = np.empty(n, dtype=float)
            se_vec = np.empty(n, dtype=float)
            for i in range(n):
                e, v = _jack_stats_per_stat(self.blocks[i], snp_counts[i])
                est_vec[i] = e
                se_vec[i] = math.sqrt(v) if np.isfinite(v) else float("nan")
            with np.errstate(invalid="ignore", divide="ignore"):
                z = est_vec / se_vec
        else:
            est_vec = np.asarray(self.est, float) if self.est is not None else np.full(len(self.rows), np.nan)
            se_vec = self.se
            z = self.z
        out["est"] = est_vec
        out["se"] = se_vec
        out["z"] = np.round(z, round_z) if round_z is not None else z
        p = np.array([math.erfc(abs(zz) / math.sqrt(2)) if np.isfinite(zz) else float("nan") for zz in z])
        out["p"] = np.round(p, round_p) if round_p is not None else p
        return FStatsFrame(out)


@dataclass
class QpWaveStats:
    f4: BlockStats
    left: list[str]
    right: list[str]
    left_base: str
    right_base: str
    row_pops: list[str]
    col_pops: list[str]

    @property
    def matrix(self) -> np.ndarray:
        return self.f4.est.reshape(len(self.row_pops), len(self.col_pops))

    @property
    def cov(self) -> np.ndarray:
        return self.f4.cov

    @property
    def blocks(self) -> np.ndarray | None:
        if self.f4.blocks is None:
            return None
        return self.f4.blocks.reshape(len(self.row_pops), len(self.col_pops), -1)

    @property
    def loo(self) -> np.ndarray | None:
        if self.f4.loo is None:
            return None
        return self.f4.loo.reshape(len(self.row_pops), len(self.col_pops), -1)

    @property
    def snp_counts(self) -> np.ndarray:
        """Usable SNP counts by left row, right feature and block."""

        counts = getattr(self.f4, "snp_counts", None)
        if counts is None:
            raise ValueError("SNP counts were not retained for this qpWave result")
        return np.asarray(counts, float).reshape(
            len(self.row_pops), len(self.col_pops), -1
        )


def _singleton_observation_rows(*pairs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    affected = None
    for afs, counts in pairs:
        values = np.isfinite(afs) & np.isfinite(counts) & (counts < 2)
        rows = values if values.ndim == 1 else np.any(values, axis=tuple(range(1, values.ndim)))
        affected = rows.copy() if affected is None else affected | rows
    return np.asarray([], dtype=bool) if affected is None else affected


def _count_affected_blocks(affected_rows: np.ndarray, block_lengths: Sequence[int]) -> int:
    affected_rows = np.asarray(affected_rows, bool)
    start = 0
    affected_blocks = 0
    for n in block_lengths:
        stop = start + int(n)
        affected_blocks += int(np.any(affected_rows[start:stop]))
        start = stop
    if start != len(affected_rows):
        raise ValueError("Block lengths must sum to the number of SNP rows")
    return affected_blocks


def _singleton_warning_message(
    stat: str,
    apply_corr: bool,
    affected_blocks: int,
    total_blocks: int,
) -> str:
    impact = "in 1 block" if total_blocks == 1 else f"in {affected_blocks} of {total_blocks} blocks"
    if apply_corr:
        return (
            f"{stat} bias correction requires at least two independent allele observations; "
            f"excluding affected SNP values with count < 2 {impact}"
        )
    return (
        f"{stat} includes affected SNP values with count < 2 because apply_corr=False; "
        f"those values cannot be estimated without sampling bias; affected values occurred {impact}"
    )


@dataclass
class _SingletonWarningSummary:
    stat: str
    apply_corr: bool
    affected_blocks: int = 0

    def observe(
        self,
        block_lengths: Sequence[int],
        *pairs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        affected_rows = _singleton_observation_rows(*pairs)
        self.affected_blocks += _count_affected_blocks(affected_rows, block_lengths)

    def warn(self, total_blocks: int, *, stacklevel: int) -> None:
        if not self.affected_blocks:
            return
        warnings.warn(
            _singleton_warning_message(
                self.stat,
                self.apply_corr,
                self.affected_blocks,
                total_blocks,
            ),
            RuntimeWarning,
            stacklevel=stacklevel,
        )


def _warn_singleton_observations(
    stat: str,
    apply_corr: bool,
    *pairs: tuple[np.ndarray, np.ndarray],
    block_lengths: Sequence[int] | None = None,
) -> None:
    affected_rows = _singleton_observation_rows(*pairs)
    if not np.any(affected_rows):
        return
    if block_lengths is None:
        block_lengths = [len(affected_rows)]
    total_blocks = len(block_lengths)
    affected_blocks = _count_affected_blocks(affected_rows, block_lengths)
    message = _singleton_warning_message(stat, apply_corr, affected_blocks, total_blocks)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _sample_bias_correction(afs: np.ndarray, counts: np.ndarray) -> np.ndarray:
    correction = np.full_like(afs, np.nan, dtype=float)
    valid = np.isfinite(afs) & np.isfinite(counts) & (counts > 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(afs * (1 - afs), counts - 1, out=correction, where=valid)
    return correction


@dataclass
class _CountJackknife:
    total: float
    loo: np.ndarray
    influence: np.ndarray
    contributes: np.ndarray
    n: float


def _validate_resampling(resampling: str) -> str:
    if resampling not in {"pairwise_counts", "nominal_blocks"}:
        raise ValueError("resampling must be 'pairwise_counts' or 'nominal_blocks'")
    return resampling


def _count_jackknife(block_ests: np.ndarray, n_per_block: np.ndarray) -> _CountJackknife:
    """Count-weighted total, physical-block LOO values, and jackknife influence."""
    ests = np.asarray(block_ests, float)
    counts = np.asarray(n_per_block, float)
    if ests.shape != counts.shape:
        raise ValueError("block estimates and per-block counts must have the same shape")
    valid = np.isfinite(ests) & np.isfinite(counts) & (counts > 0)
    loo = np.full(ests.shape, np.nan, dtype=float)
    influence = np.zeros(ests.shape, dtype=float)
    if not np.any(valid):
        return _CountJackknife(float("nan"), loo, np.full_like(influence, np.nan), valid, 0.0)

    total_n = float(np.sum(counts[valid]))
    total = float(np.sum(ests[valid] * counts[valid]) / total_n)
    # A block with n=0 deletes no observations, so its LOO equals the full
    # estimate and its influence is exactly zero.
    loo[~valid] = total
    remaining = total_n - counts[valid]
    can_delete = remaining > 0
    valid_i = np.flatnonzero(valid)
    delete_i = valid_i[can_delete]
    loo[delete_i] = (
        total * total_n - ests[delete_i] * counts[delete_i]
    ) / remaining[can_delete]
    if len(delete_i) >= 2:
        h = total_n / counts[delete_i]
        tau = h * total - (h - 1.0) * loo[delete_i]
        influence[delete_i] = (tau - total) / np.sqrt(h - 1.0)
    else:
        influence[valid] = np.nan
    return _CountJackknife(total, loo, influence, valid, total_n)


def _jack_stats_per_stat(block_ests: np.ndarray, n_per_block: np.ndarray) -> tuple[float, float]:
    # Per-stat block jackknife where each stat has its own per-block SNP count
    # (allsnps mode). Mirrors admixtools::est_to_loo_dat + jack_dat_stats, which
    # use the 'tot' form of cpp_jack_vec_stats. Blocks with n=0 or non-finite
    # block estimate are dropped.
    jack = _count_jackknife(block_ests, n_per_block)
    keep = jack.contributes & np.isfinite(jack.influence)
    if int(np.sum(keep)) < 2:
        return jack.total, float("nan")
    return jack.total, float(np.mean(jack.influence[keep] ** 2))


def _influence_covariance(influence: np.ndarray, contributes: np.ndarray) -> np.ndarray:
    influence = np.asarray(influence, float)
    contributes = np.asarray(contributes, bool)
    if influence.ndim != 2 or influence.shape != contributes.shape:
        raise ValueError("Influences and contribution masks must be matching matrices")
    keep = contributes & np.isfinite(influence)
    blocks = keep.sum(axis=1)
    valid = (blocks >= 2) & np.all(~contributes | np.isfinite(influence), axis=1)
    values = np.where(keep, influence, 0.0) / np.sqrt(np.maximum(blocks, 1))[:, None]
    covariance = values @ values.T
    covariance[~valid, :] = np.nan
    covariance[:, ~valid] = np.nan
    return covariance


def _influence_variances(influence: np.ndarray, contributes: np.ndarray) -> np.ndarray:
    # Return only covariance diagonal entries (for custom usage)
    influence = np.asarray(influence, float)
    contributes = np.asarray(contributes, bool)
    variances = np.full(influence.shape[0], np.nan, dtype=float)
    for i in range(influence.shape[0]):
        keep = contributes[i] & np.isfinite(influence[i])
        if int(np.sum(keep)) >= 2:
            variances[i] = float(np.mean(influence[i, keep] ** 2))
    return variances


def _ratio_block_jackknife(
    block_num_means: np.ndarray,
    block_den_means: np.ndarray,
    block_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Jackknife ratios of pooled block components.

    This mirrors admixtools' ``est_to_loo_dat`` followed by
    ``jack_dat_stats`` for the numerator/denominator direct-f3 path. Inputs
    are statistic-by-block matrices of component means and their effective
    block weights (normally contributing SNP counts).
    """
    num = np.asarray(block_num_means, float)
    den = np.asarray(block_den_means, float)
    weights = np.asarray(block_weights, float)
    if num.ndim == 1:
        num, den, weights = num[None, :], den[None, :], weights[None, :]
    if num.shape != den.shape or num.shape != weights.shape:
        raise ValueError("ratio numerator, denominator, and weights must have matching shapes")

    nstats, nblocks = num.shape
    estimates = np.full(nstats, np.nan, dtype=float)
    loo = np.full((nstats, nblocks), np.nan, dtype=float)
    influence = np.full((nstats, nblocks), np.nan, dtype=float)
    contributes = np.zeros((nstats, nblocks), dtype=bool)

    for stat_i in range(nstats):
        valid = (
            np.isfinite(num[stat_i])
            & np.isfinite(den[stat_i])
            & np.isfinite(weights[stat_i])
            & (weights[stat_i] > 0)
        )
        contributes[stat_i] = valid
        if not np.any(valid):
            continue
        w = weights[stat_i, valid]
        nsum = float(w.sum())
        num_sums = num[stat_i, valid] * w
        den_sums = den[stat_i, valid] * w
        total_num = float(num_sums.sum())
        total_den = float(den_sums.sum())
        full = float(np.divide(total_num, total_den)) if total_den != 0 else float("nan")

        valid_idx = np.flatnonzero(valid)
        remaining_w = nsum - w
        remaining_den = total_den - den_sums
        can_delete = (remaining_w > 0) & (remaining_den != 0)
        delete_idx = valid_idx[can_delete]
        loo[stat_i, delete_idx] = (
            total_num - num_sums[can_delete]
        ) / remaining_den[can_delete]

        finite = np.isfinite(loo[stat_i])
        if int(finite.sum()) < 2:
            estimates[stat_i] = full
            continue
        lw = weights[stat_i, finite]
        lv = loo[stat_i, finite]
        finite_n = float(lw.sum())
        delete_weight = 1.0 - lw / finite_n
        jack_total = float(np.average(lv, weights=delete_weight))
        jack_est = float(np.sum(jack_total - lv) + np.average(lv, weights=lw))
        estimates[stat_i] = jack_est
        h = finite_n / lw
        tau = h * jack_total - (h - 1.0) * lv
        influence[stat_i, finite] = (tau - jack_est) / np.sqrt(h - 1.0)

    cov = _influence_covariance(influence, contributes)
    return estimates, loo, influence, cov


def _as_list(x):
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


def _f4_combinations(pop1, pop2, pop3, pop4, comb: bool) -> pd.DataFrame:
    if isinstance(pop1, pd.DataFrame):
        cols = ["pop1", "pop2", "pop3", "pop4"]
        if not all(c in pop1.columns for c in cols):
            raise ValueError("Population-combination data frame must have pop1, pop2, pop3, pop4 columns")
        return pop1[cols].copy()
    p1, p2, p3, p4 = map(_as_list, (pop1, pop2, pop3, pop4))
    if p1 is None or p2 is None or p3 is None or p4 is None:
        raise ValueError("pop1, pop2, pop3, and pop4 are required for f4")
    if comb:
        return pd.DataFrame(product(p1, p2, p3, p4), columns=["pop1", "pop2", "pop3", "pop4"])
    lengths = {len(p1), len(p2), len(p3), len(p4)}
    if len(lengths) != 1:
        raise ValueError("With comb=False, pop1/pop2/pop3/pop4 must have equal lengths")
    return pd.DataFrame({"pop1": p1, "pop2": p2, "pop3": p3, "pop4": p4})


def _f4_direct_blocks_from_afs(
    afdat: AfData,
    combos: pd.DataFrame,
    blgsize: float = 0.05,
    allsnps: bool = False,
    poly_only: bool = False,
    snpwt: Sequence[float] | None = None,
    apply_corr: bool = True,
    stat_name: str = "f4",
    normalize_by_target_het: bool = False,
    covariance: bool = True,
    verbose: bool = True,
    singleton_warning: _SingletonWarningSummary | None = None,
) -> BlockStats:
    cols = ["pop1", "pop2", "pop3", "pop4"]
    combos = combos.reset_index(drop=True).copy()
    missing_cols = [c for c in cols if c not in combos.columns]
    if missing_cols:
        raise ValueError(f"f4 combinations are missing columns: {missing_cols}")
    if "model" not in combos.columns:
        combos["model"] = 1

    pops = list(afdat.afs.columns)
    missing = sorted(set(combos[cols].to_numpy().reshape(-1)) - set(pops))
    if missing:
        raise ValueError(f"Populations missing from allele-frequency table: {missing}")

    arr = afdat.afs.to_numpy(float)
    count_arr = afdat.counts.to_numpy(float)
    pop_i = {p: i for i, p in enumerate(pops)}
    idx = np.asarray([[pop_i[getattr(row, c)] for c in cols] for row in combos.itertuples(index=False)], dtype=int)
    correction_coefficients = np.zeros((len(combos), len(pops)), dtype=float)
    for stat_i, row in enumerate(combos.itertuples(index=False)):
        first: dict[str, float] = {}
        second: dict[str, float] = {}
        first[row.pop1] = first.get(row.pop1, 0.0) + 1.0
        first[row.pop2] = first.get(row.pop2, 0.0) - 1.0
        second[row.pop3] = second.get(row.pop3, 0.0) + 1.0
        second[row.pop4] = second.get(row.pop4, 0.0) - 1.0
        for pop in set(first) | set(second):
            correction_coefficients[stat_i, pop_i[pop]] = first.get(pop, 0.0) * second.get(pop, 0.0)
    correction_pops = np.flatnonzero(np.any(correction_coefficients != 0, axis=0))
    block_lengths = get_block_lengths(afdat.snpfile, blgsize)
    if correction_pops.size:
        pairs = ((arr[:, correction_pops], count_arr[:, correction_pops]),)
        if singleton_warning is None:
            _warn_singleton_observations(
                stat_name,
                apply_corr,
                *pairs,
                block_lengths=block_lengths,
            )
        else:
            singleton_warning.observe(block_lengths, *pairs)
    nstats = len(combos)
    out = np.full((nstats, len(block_lengths)), np.nan, dtype=float)
    denominator = np.full_like(out, np.nan) if normalize_by_target_het else None
    snp_counts = np.zeros((nstats, len(block_lengths)), dtype=float)
    snpwt = None if snpwt is None else np.asarray(snpwt, float)
    if snpwt is not None and len(snpwt) != arr.shape[0]:
        raise ValueError("snpwt must have one value per retained SNP")

    use_by_model: dict[object, np.ndarray] = {}
    if not allsnps:
        for model, sub in combos.groupby("model", sort=False):
            model_pops = sorted(set(sub[cols].to_numpy().reshape(-1)))
            model_idx = [pop_i[p] for p in model_pops]
            use = np.isfinite(arr[:, model_idx]).all(axis=1)
            if poly_only:
                vals = arr[:, model_idx]
                # Direct ADMIXTOOLS statistics retain segregating sites even
                # when every population has the same non-boundary frequency.
                # Only sites fixed at 0 or fixed at 1 are monomorphic here.
                use &= (np.nanmax(vals, axis=1) > 0) & (np.nanmin(vals, axis=1) < 1)
            use_by_model[model] = use

    start = 0
    for b, n in enumerate(block_lengths):
        stop = start + int(n)
        _log_block(f"direct {stat_name}", b, len(block_lengths), start, stop, verbose)
        block = arr[start:stop]
        count_block = count_arr[start:stop]
        for stat_i, row in enumerate(combos.itertuples(index=False)):
            p = idx[stat_i]
            vals = block[:, p]
            use = (
                np.isfinite(vals).all(axis=1)
                if allsnps
                else use_by_model[getattr(row, "model")][start:stop].copy()
            )
            if allsnps and poly_only:
                finite_vals = vals[use]
                if finite_vals.size:
                    use_idx = np.where(use)[0]
                    use[use_idx] &= (
                        (np.max(finite_vals, axis=1) > 0)
                        & (np.min(finite_vals, axis=1) < 1)
                    )
            correction = correction_coefficients[stat_i]
            required = np.flatnonzero(correction != 0)
            if apply_corr and required.size:
                use &= np.isfinite(count_block[:, required]).all(axis=1)
                use &= (count_block[:, required] > 1).all(axis=1)
            if normalize_by_target_het:
                target_idx = p[0]
                use &= np.isfinite(count_block[:, target_idx])
                use &= count_block[:, target_idx] > 1
            if not np.any(use):
                continue
            f4vals = (vals[use, 0] - vals[use, 1]) * (vals[use, 2] - vals[use, 3])
            if apply_corr:
                for pop_idx in required:
                    corr = _sample_bias_correction(
                        block[use, pop_idx],
                        count_block[use, pop_idx],
                    )
                    f4vals = f4vals - correction[pop_idx] * corr
            if snpwt is not None:
                f4vals = f4vals * snpwt[start:stop][use]
            out[stat_i, b] = float(np.mean(f4vals))
            if normalize_by_target_het:
                target_p = block[use, target_idx]
                target_n = count_block[use, target_idx]
                target_het = 2.0 * target_p * (1.0 - target_p) * target_n / (target_n - 1.0)
                if snpwt is not None:
                    # Normalized f3 is a ratio of two SNP-weighted sums.  Apply
                    # the same optional outgroup weight to both components.
                    target_het = target_het * snpwt[start:stop][use]
                denominator[stat_i, b] = float(np.mean(target_het))
            snp_counts[stat_i, b] = int(use.sum())
        start = stop

    effective_lengths = np.nanmax(snp_counts, axis=0)
    effective_lengths = np.where(effective_lengths > 0, effective_lengths, block_lengths).astype(float)
    if normalize_by_target_het:
        ratio_blocks = np.full_like(out, np.nan)
        np.divide(out, denominator, out=ratio_blocks, where=denominator != 0)
        est, loo, _, cov = _ratio_block_jackknife(out, denominator, snp_counts)
        result_blocks = ratio_blocks
    else:
        jacks = [_count_jackknife(out[i], snp_counts[i]) for i in range(nstats)]
        est = np.asarray([jack.total for jack in jacks], float)
        loo = np.asarray([jack.loo for jack in jacks], float)
        influence = np.asarray([jack.influence for jack in jacks], float)
        contributes = np.asarray([jack.contributes for jack in jacks], bool)
        cov = _influence_covariance(influence, contributes) if covariance else None
        result_blocks = out
    rows = combos.drop(columns=["model"]) if set(combos["model"]) == {1} else combos
    stats = BlockStats(rows=rows.reset_index(drop=True), blocks=result_blocks, block_lengths=effective_lengths, stat=stat_name, loo=loo, est=est, cov=cov)
    if not covariance and not normalize_by_target_het:
        stats.variances = _influence_variances(influence, contributes)
    if not normalize_by_target_het:
        stats.influence = influence
        stats.contributes = contributes
    stats.snp_counts = snp_counts
    stats.nominal_block_lengths = np.asarray(block_lengths, float)
    return stats


def f4_stats(
    data,
    pop1,
    pop2=None,
    pop3=None,
    pop4=None,
    comb: bool = True,
    unique_only: bool = True,
    afprod: bool = False,
    keep_blocks: bool = True,
    keep_loo: bool = True,
    covariance: bool = True,
    allsnps: bool = False,
    resampling: str = "pairwise_counts",
    verbose: bool = True,
    **kwargs,
) -> BlockStats:
    _validate_resampling(resampling)
    if afprod and allsnps:
        raise ValueError("afprod=True and allsnps=True together are not supported")
    combos = _f4_combinations(pop1, pop2, pop3, pop4, comb)
    if unique_only:
        combos = combos.drop_duplicates().reset_index(drop=True)
    # This copy carries only the direct allsnps path over an in-memory
    # allele-frequency table; the f2-cache and genotype-streaming paths of the
    # original module are not included.
    if not isinstance(data, AfData):
        raise ValueError("This f4_stats copy requires an AfData allele-frequency table")
    if not allsnps:
        raise ValueError("This f4_stats copy requires allsnps=True")
    stats = _f4_direct_blocks_from_afs(
        data,
        combos,
        allsnps=True,
        verbose=verbose,
        covariance=covariance,
        **kwargs,
    )
    if not keep_blocks:
        stats.blocks = None
    if not keep_loo:
        stats.loo = None
    if not covariance:
        stats.cov = None
    return stats


def _contrast_pops(pops: Sequence[str], base: str | None, name: str) -> tuple[str, list[str]]:
    pops = list(pops)
    if len(pops) < 2:
        raise ValueError(f"{name} must contain at least two populations")
    base = pops[0] if base is None else base
    if base not in pops:
        raise ValueError(f"{name}_base must be included in {name}")
    return base, [p for p in pops if p != base]


def qpwave_f4stats(
    data,
    left: Sequence[str],
    right: Sequence[str],
    left_base: str | None = None,
    right_base: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> QpWaveStats:
    if not kwargs.get("covariance", True):
        raise ValueError(
            "qpWave and qpAdm require covariance=True; covariance=False is only "
            "supported for standalone f4 statistics"
        )
    kwargs.setdefault("allsnps", True)
    left = list(left)
    right = list(right)
    left_base, row_pops = _contrast_pops(left, left_base, "left")
    right_base, col_pops = _contrast_pops(right, right_base, "right")
    combos = pd.DataFrame(
        [
            {
                "pop1": row_pop,
                "pop2": left_base,
                "pop3": col_pop,
                "pop4": right_base,
                "left": row_pop,
                "right": col_pop,
            }
            for row_pop in row_pops
            for col_pop in col_pops
        ]
    )
    stats = f4_stats(data, combos[["pop1", "pop2", "pop3", "pop4"]], unique_only=False, verbose=verbose, **kwargs)
    stats.rows = combos
    return QpWaveStats(stats, left, right, left_base, right_base, row_pops, col_pops)
