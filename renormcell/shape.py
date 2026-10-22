"""Count shaping from the frozen K448 recipe; statistics and both moments are retained."""
import numpy as np
from numba import njit
from .atlas import repair_bulk_totals

def balance_rest(rest, row_sum, column_sum, iterations=60):
    """Rake uncalled genes to fixed cell totals and gene means."""
    x = rest.astype(np.float64, copy=True) + 1e-12 * column_sum
    for _ in range(iterations):
        columns = x.sum(0)
        x *= np.divide(column_sum, columns, out=np.zeros_like(column_sum), where=columns > 0)
        rows = x.sum(1)
        x *= np.divide(row_sum, rows, out=np.zeros_like(row_sum), where=rows > 0)[:, None]
    return x


def balance_two_moments(rest, row_sum, column_sum, raw_sum, depth, iterations=100):
    """Rake uncalled genes to row, normalized-mean, and raw-bulk constraints."""
    x = rest.astype(np.float64, copy=True) + 1e-12 * column_sum
    target_depth = np.divide(raw_sum, column_sum, out=np.zeros_like(raw_sum), where=column_sum > 0)
    for _ in range(iterations):
        columns = x.sum(0)
        x *= np.divide(column_sum, columns, out=np.zeros_like(column_sum), where=columns > 0)
        current = np.divide(depth @ x, column_sum, out=np.zeros_like(column_sum), where=column_sum > 0)
        second = np.divide(depth * depth @ x, column_sum, out=np.zeros_like(column_sum), where=column_sum > 0)
        variance = np.maximum(second - current * current, 0)
        tilt = np.divide(target_depth - current, variance, out=np.zeros_like(current), where=variance > 1e-12)
        lower = -.95 / np.maximum(depth.max() - current, 1e-12)
        upper = .95 / np.maximum(current - depth.min(), 1e-12)
        x *= 1 + np.clip(tilt, lower, upper)[None, :] * (depth[:, None] - current[None, :])
        rows = x.sum(1)
        x *= np.divide(row_sum, rows, out=np.zeros_like(row_sum), where=rows > 0)[:, None]
    return x


def shaped_column(total, raw_total, depth, decoder, local, direction, strength, rng):
    """Meet both moments while moving the Wilcoxon statistic toward the chosen direction."""
    order = rng.permutation(len(depth))
    target = raw_total / total
    best = None
    low, high = 1, len(depth)
    for _ in range(12):
        n = (low + high) // 2
        q = np.zeros(len(depth))
        q[order[:n]] = total / n
        average = depth[order[:n]].mean()
        high = target >= average
        eligible = np.flatnonzero(depth >= target) if high else np.flatnonzero(depth <= target)
        endpoint = int(rng.choice(eligible)) if len(eligible) else int(
            np.argmax(depth) if high else np.argmin(depth)
        )
        extreme = depth[endpoint]
        alpha = np.clip((target - average) / (extreme - average + 1e-30), 0, 1)
        q *= 1 - alpha
        q[endpoint] += alpha * total
        statistic = decoder.psi(local, q * 1e6).mean() / decoder.x.shape[0]
        error = abs(statistic - (.5 + direction * strength))
        if best is None or error < best[0]:
            best = error, q
        if statistic < .5 + direction * strength:
            low = n + 1
        else:
            high = n - 1
        if low > high:
            break
    return best[1]


def round_to_depth(row, depth):
    integer = np.floor(row)
    residual = int(depth - integer.sum())
    if residual:
        integer[np.argpartition(-(row - integer), residual - 1)[:residual]] += 1
    return integer


def repair_exact_columns(integer, expected, target):
    """Move integer counts within rows until every raw gene total is exact."""
    difference = integer.sum(0, dtype=np.int64) - np.asarray(target, dtype=np.int64)
    surplus = {int(g): int(n) for g, n in enumerate(difference) if n > 0}
    for deficit_gene in np.flatnonzero(difference < 0):
        need = int(-difference[deficit_gene])
        while need:
            source = next(iter(surplus))
            rows = np.flatnonzero(integer[:, source] > 0)
            score = ((expected[rows, deficit_gene] - integer[rows, deficit_gene]) -
                     (expected[rows, source] - integer[rows, source]))
            row = int(rows[np.argmax(score)])
            integer[row, source] -= 1
            integer[row, deficit_gene] += 1
            surplus[source] -= 1
            if not surplus[source]:
                del surplus[source]
            need -= 1
    if surplus or not np.array_equal(integer.sum(0, dtype=np.int64), target):
        raise AssertionError("Exact column repair failed")
    return integer


def _exact_column_moves(integer, expected, difference):
    source = 0
    for deficit_gene in range(len(difference)):
        need = -difference[deficit_gene]
        while need > 0:
            while source < len(difference) and difference[source] <= 0:
                source += 1
            if source == len(difference):
                raise AssertionError("Exact column repair exhausted surplus")
            best_row, best_score = -1, -np.inf
            for row in range(len(integer)):
                if integer[row, source] > 0:
                    score = ((expected[row, deficit_gene] - integer[row, deficit_gene]) -
                             (expected[row, source] - integer[row, source]))
                    if score > best_score:
                        best_row, best_score = row, score
            if best_row < 0:
                raise AssertionError("Surplus gene has no positive row")
            integer[best_row, source] -= 1
            integer[best_row, deficit_gene] += 1
            difference[source] -= 1
            need -= 1


if njit is not None:
    _exact_column_moves = njit(cache=True)(_exact_column_moves)


def repair_exact_columns_fast(integer, expected, target):
    """Numba version of the same ordered count transfers, opt-in only."""
    if njit is None:
        raise ImportError("--fast-repair requires numba")
    difference = integer.sum(0, dtype=np.int64) - np.asarray(target, dtype=np.int64)
    _exact_column_moves(integer, expected, difference)
    if np.any(difference > 0) or not np.array_equal(integer.sum(0, dtype=np.int64), target):
        raise AssertionError("Exact column repair failed")
    return integer


def shape_group(counts, decoder, k, seed, fixed_depth=False, dual_moment=False,
                flip_direction=False, strength=.1, selection_score=None,
                selection_mode="lfc", native_depth=False):
    rng = np.random.default_rng(seed)
    depth = counts.sum(1, dtype=np.float64)
    x = counts / depth[:, None]
    mean = x.mean(0)
    control_mean = decoder.mean / 1e6
    lfc = np.log2((mean[decoder.gate] + 1e-12) / (control_mean[decoder.gate] + 1e-12))
    if selection_score is not None:
        ranking = np.asarray(selection_score)[decoder.gate]
    elif selection_mode == "lfc":
        ranking = np.abs(lfc)
    else:
        statistic = np.asarray([
            decoder.psi(i, x[:, gene] * 1e6).mean() / decoder.x.shape[0]
            for i, gene in enumerate(decoder.gate)
        ])
        ranking = (np.abs(statistic - .5) if selection_mode == "rank" else
                   -np.sign(lfc) * (statistic - .5))
    ranking = np.nan_to_num(ranking, nan=-np.inf)
    scale = (1 if native_depth else max(1, int(1_000_000 // depth.max()))) if dual_moment else 1
    fitting_depth = depth * scale
    if dual_moment:
        called_local, columns = [], []
        selected_mass = np.zeros(len(x))
        for local in np.argsort(ranking)[::-1]:
            gene = decoder.gate[local]
            if mean[gene] <= 0:
                continue
            column = shaped_column(
                mean[gene] * len(x), counts[:, gene].sum() * scale,
                fitting_depth, decoder, local,
                np.sign(lfc[local]) * (-1 if flip_direction else 1), strength, rng,
            )
            if np.max(selected_mass + column) >= .98:
                continue
            called_local.append(local)
            columns.append(column)
            selected_mass += column
            if len(columns) == min(k, len(ranking)):
                break
        if not columns:
            raise ValueError("No feasible called genes")
        shaped = np.column_stack(columns)
    else:
        called_local = np.argpartition(ranking, -min(k, len(ranking)))[-min(k, len(ranking)):]
        shaped = np.zeros((len(x), len(called_local)), dtype=np.float64)
        for j, local in enumerate(called_local):
            goal = .5 + np.sign(lfc[local]) * .1
            low, high = 1 / len(x), 1.0
            gene_mean = mean[decoder.gate[local]] * 1e6
            for _ in range(24):
                fraction = (low + high) / 2
                nonzero = max(1, round(len(x) * fraction))
                values = np.zeros(len(x))
                values[:nonzero] = gene_mean * len(x) / nonzero
                if decoder.psi(local, values).mean() / decoder.x.shape[0] < goal:
                    low = fraction
                else:
                    high = fraction
            nonzero = max(1, round(len(x) * (low + high) / 2))
            rows = rng.permutation(len(x))[:nonzero]
            shaped[rows, j] = mean[decoder.gate[local]] * len(x) / nonzero
        selected_mass = shaped.sum(1)
    if selected_mass.max() >= .98:
        raise ValueError(f"Called genes consume {selected_mass.max():.3f} of a cell")
    called_local = np.asarray(called_local)
    called = decoder.gate[called_local]
    keep = np.ones(x.shape[1], dtype=bool)
    keep[called] = False
    column_sum = x[:, keep].sum(0)
    if dual_moment:
        rest = balance_two_moments(
            x[:, keep], 1 - selected_mass, column_sum,
            counts[:, keep].sum(0) * scale, fitting_depth,
        )
    else:
        rest = balance_rest(x[:, keep], 1 - selected_mass, column_sum)
    probability = np.empty_like(x)
    probability[:, keep] = rest
    probability[:, called] = shaped
    probability /= probability.sum(1, keepdims=True)
    output_depth = fitting_depth if dual_moment else (
        np.full(len(x), 1_000_000.0) if fixed_depth else depth
    )
    expected = probability * output_depth[:, None]
    integer = np.stack([
        round_to_depth(row, total) for row, total in zip(expected, output_depth)
    ]).astype(np.int32)
    integer = repair_bulk_totals(integer, expected).astype(np.float32)
    if dual_moment:
        target_columns = counts.sum(0, dtype=np.float64).astype(np.int64) * scale
        integer = repair_exact_columns(integer, expected, target_columns)
    if not np.array_equal(integer.sum(1), output_depth):
        raise AssertionError("Rounding changed cell depths")
    return integer, float(np.abs(probability.mean(0) - mean).sum())


