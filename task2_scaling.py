import math
import multiprocessing as mp
import statistics
import time

from harness import emit_verify_tag


N_STRONG = 1_600_000
N_WEAK_PER_WORKER = 250_000
P_PHYS = 6

WARMUP_RUNS = 1
TIMED_RUNS = 3


def stage_a(n: int) -> float:
    total = 0.0
    for i in range(n // 20):
        x = (i % 1000 + 1) / 1000.0
        for _ in range(6):
            x = math.sin(x)
            x = math.sqrt(abs(x))
        total += x
    return total


def stage_b_range(bounds: tuple[int, int]) -> float:
    start, end = bounds
    total = 0.0

    for i in range(start, end):
        x = (i % 1000 + 1) / 1000.0
        for _ in range(35):
            x = math.sin(x * 0.5 + 0.1)
        total += x

    return total


def partitions(n: int, p: int) -> list[tuple[int, int]]:
    return [(n * t // p, n * (t + 1) // p) for t in range(p)]


def trial(pool, n: int, p: int) -> tuple[float, float]:
    """Возвращает время всего конвейера и Stage A в миллисекундах."""
    start = time.perf_counter_ns()

    serial_result = stage_a(n)
    after_a = time.perf_counter_ns()

    partial_results = pool.map(stage_b_range, partitions(n, p))
    end = time.perf_counter_ns()

    # Результаты реально вычислены и получены родительским процессом.
    _ = serial_result + sum(partial_results)

    return (end - start) / 1e6, (after_a - start) / 1e6


def measure(pool, n: int, p: int) -> tuple[float, float]:
    for _ in range(WARMUP_RUNS):
        trial(pool, n, p)

    samples = [trial(pool, n, p) for _ in range(TIMED_RUNS)]
    total_ms = statistics.median(sample[0] for sample in samples)
    stage_a_ms = statistics.median(sample[1] for sample in samples)
    return total_ms, stage_a_ms


def main():
    student_id = 230103082
    p_log = mp.cpu_count()

    strong_results = {}
    weak_results = {}

    print(f"P_log: {p_log}", flush=True)
    print("Starting Phase 2; progress will appear after each measurement.", flush=True)

    for p in range(1, p_log + 1):
        print(f"\nCreating pool for p={p}...", flush=True)

        # Один пул на p; он создаётся вне измеряемой области и
        # используется повторно для обоих экспериментов.
        with mp.Pool(processes=p) as pool:
            print(f"  Measuring strong scaling, p={p}...", flush=True)
            strong_results[p] = measure(pool, N_STRONG, p)
            print(
                f"  Strong median: {strong_results[p][0]:.3f} ms",
                flush=True,
            )

            print(f"  Measuring weak scaling, p={p}...", flush=True)
            weak_results[p] = measure(pool, p * N_WEAK_PER_WORKER, p)
            print(
                f"  Weak median: {weak_results[p][0]:.3f} ms",
                flush=True,
            )

    strong_t1 = strong_results[1][0]
    weak_t1 = weak_results[1][0]

    print("\nTable 2A: Strong scaling")
    print("p  median_ms  stageA_ms  speedup  efficiency")
    for p in range(1, p_log + 1):
        total, stage_a_ms = strong_results[p]
        speedup = strong_t1 / total
        print(
            f"{p:2d} {total:10.3f} {stage_a_ms:10.3f} "
            f"{speedup:8.4f} {speedup / p:11.4f}"
        )

    print("\nTable 2B: Weak scaling")
    print("p  N  median_ms  stageA_ms  efficiency  alpha  Gustafson")
    for p in range(1, p_log + 1):
        total, stage_a_ms = weak_results[p]
        alpha = stage_a_ms / total
        gustafson = p - alpha * (p - 1)
        print(
            f"{p:2d} {p * N_WEAK_PER_WORKER:7d} {total:10.3f} "
            f"{stage_a_ms:10.3f} {weak_t1 / total:11.4f} "
            f"{alpha:6.4f} {gustafson:10.4f}"
        )

    serial_fraction = strong_results[1][1] / strong_t1
    amdahl_max = 1 / serial_fraction
    amdahl_plog = 1 / (serial_fraction + (1 - serial_fraction) / p_log)
    observed_plog = strong_t1 / strong_results[p_log][0]

    weak_total, weak_a = weak_results[p_log]
    gustafson_plog = p_log - (weak_a / weak_total) * (p_log - 1)

    print("\nAmdahl analysis")
    print(f"Serial fraction s: {serial_fraction:.6f}")
    print(f"Theoretical maximum speedup: {amdahl_max:.4f}")
    print(f"Theoretical speedup at P_log: {amdahl_plog:.4f}")
    print(f"Observed strong speedup at P_log: {observed_plog:.4f}")

    if p_log > P_PHYS and P_PHYS >= 2:
        speedups = {
            p: strong_t1 / strong_results[p][0]
            for p in (P_PHYS - 1, P_PHYS, P_PHYS + 1)
        }
        before = speedups[P_PHYS] - speedups[P_PHYS - 1]
        after = speedups[P_PHYS + 1] - speedups[P_PHYS]

        print("\nBoundary analysis")
        print(f"Marginal speedup {P_PHYS - 1}->{P_PHYS}: {before:.6f}")
        print(f"Marginal speedup {P_PHYS}->{P_PHYS + 1}: {after:.6f}")
        if before != 0:
            drop = (before - after) / before * 100
            print(f"Change expressed as percentage drop: {drop:.2f}%")
        else:
            print("Percentage drop undefined: previous increment is zero.")

    print(f"\nPHASE2 primary metric: {observed_plog:.4f}")
    print(f"PHASE2 secondary metric: {gustafson_plog:.4f}")
    print(emit_verify_tag("PHASE2", student_id, observed_plog, gustafson_plog))


if __name__ == "__main__":
    mp.freeze_support()
    main()