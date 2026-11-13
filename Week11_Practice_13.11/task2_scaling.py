import math
import multiprocessing as mp
import statistics
import time

from harness import emit_verify_tag


STUDENT_ID = 230103082
P_PHYS = 6

STRONG_N = 1_200_000
WEAK_N0 = 200_000

WARMUP = 2
RUNS = 7


def stage_a(n: int) -> float:
    """Последовательная стадия: 10 вызовов sin/sqrt на элемент."""
    result = 0.0

    for i in range(n // 20):
        x = (i % 100 + 1) / 100.0
        for _ in range(5):
            x = math.sqrt(abs(math.sin(x)) + 1.0)
        result += x

    return result


def stage_b_worker(bounds: tuple[int, int]) -> float:
    """Параллельная стадия: 30 арифметических/тригонометрических
    операций на элемент, если считать умножение и сложение отдельно.
    """
    start, end = bounds
    result = 0.0

    for i in range(start, end):
        x = (i % 100 + 1) / 100.0
        for _ in range(10):
            x = math.sin(x * 0.5 + 0.1)
        result += x

    return result


def partitions(n: int, p: int) -> list[tuple[int, int]]:
    return [(n * i // p, n * (i + 1) // p) for i in range(p)]


def measure_pipeline(pool, n: int, p: int) -> tuple[float, float]:
    """Возвращает медианы общего времени и времени Stage A в секундах."""
    jobs = partitions(n, p)
    total_samples = []
    stage_a_samples = []

    for trial in range(WARMUP + RUNS):
        total_start = time.perf_counter()

        a_start = time.perf_counter()
        a_result = stage_a(n)
        a_time = time.perf_counter() - a_start

        b_results = pool.map(stage_b_worker, jobs)
        total_time = time.perf_counter() - total_start

        # Используем результаты обеих стадий.
        checksum = a_result + sum(b_results)
        if not math.isfinite(checksum):
            raise RuntimeError("Pipeline returned a non-finite result")

        if trial >= WARMUP:
            total_samples.append(total_time)
            stage_a_samples.append(a_time)

    return statistics.median(total_samples), statistics.median(stage_a_samples)


def main() -> None:
    p_log = mp.cpu_count()
    strong = {}
    weak = {}

    print(f"Logical CPUs: {p_log}", flush=True)
    print(f"Warmup: {WARMUP}, measured runs: {RUNS}", flush=True)

    for p in range(1, p_log + 1):
        print(f"\nMeasuring p={p} ...", flush=True)

        # Один пул на p, созданный вне измеряемых циклов.
        with mp.Pool(processes=p) as pool:
            strong[p] = measure_pipeline(pool, STRONG_N, p)
            weak[p] = measure_pipeline(pool, p * WEAK_N0, p)

        print(
            f"  strong: {strong[p][0]:.3f} s; "
            f"weak: {weak[p][0]:.3f} s",
            flush=True,
        )

    strong_t1 = strong[1][0]
    weak_t1 = weak[1][0]
    serial_fraction = strong[1][1] / strong_t1

    print("\n=== Strong scaling ===")
    print(" p    total_s   stage_A_s   speedup   efficiency")

    for p in range(1, p_log + 1):
        total, stage_a_time = strong[p]
        speedup = strong_t1 / total
        efficiency = speedup / p
        print(
            f"{p:2d} {total:10.3f} {stage_a_time:11.3f}"
            f" {speedup:9.3f} {efficiency:12.3f}"
        )

    print(f"\nEstimated serial fraction s: {serial_fraction:.4f}")
    print(f"Amdahl asymptotic cap 1/s: {1 / serial_fraction:.3f}")

    print("\n=== Weak scaling ===")
    print(" p    total_s   stage_A_s   efficiency   Gustafson")

    for p in range(1, p_log + 1):
        total, stage_a_time = weak[p]
        efficiency = weak_t1 / total
        alpha = stage_a_time / total
        gustafson = p - alpha * (p - 1)
        print(
            f"{p:2d} {total:10.3f} {stage_a_time:11.3f}"
            f" {efficiency:12.3f} {gustafson:11.3f}"
        )

    if P_PHYS + 1 <= p_log:
        s_before = strong_t1 / strong[P_PHYS - 1][0]
        s_phys = strong_t1 / strong[P_PHYS][0]
        s_after = strong_t1 / strong[P_PHYS + 1][0]
        print("\n=== Marginal strong-speedup change ===")
        print(f"p={P_PHYS - 1} -> {P_PHYS}: {s_phys - s_before:.3f}")
        print(f"p={P_PHYS} -> {P_PHYS + 1}: {s_after - s_phys:.3f}")

    strong_speedup_log = strong_t1 / strong[p_log][0]
    weak_total_log, weak_a_log = weak[p_log]
    alpha_log = weak_a_log / weak_total_log
    gustafson_log = p_log - alpha_log * (p_log - 1)

    print()
    print(
        emit_verify_tag(
            "PHASE2",
            STUDENT_ID,
            strong_speedup_log,
            gustafson_log,
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()