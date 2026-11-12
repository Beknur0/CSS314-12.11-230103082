import multiprocessing as mp

from harness import run_benchmark


P_PHYS = 6
ITERATIONS = 2_500_000
FLOPS_PER_WORKER = ITERATIONS * 16 * 2


def compute_worker(_):
    a = 0.999999
    b = 0.000001

    x0 = x1 = x2 = x3 = 0.5
    x4 = x5 = x6 = x7 = 0.5
    x8 = x9 = x10 = x11 = 0.5
    x12 = x13 = x14 = x15 = 0.5

    for _ in range(ITERATIONS):
        x0 = x0 * a + b
        x1 = x1 * a + b
        x2 = x2 * a + b
        x3 = x3 * a + b
        x4 = x4 * a + b
        x5 = x5 * a + b
        x6 = x6 * a + b
        x7 = x7 * a + b
        x8 = x8 * a + b
        x9 = x9 * a + b
        x10 = x10 * a + b
        x11 = x11 * a + b
        x12 = x12 * a + b
        x13 = x13 * a + b
        x14 = x14 * a + b
        x15 = x15 * a + b

    return sum((
        x0, x1, x2, x3, x4, x5, x6, x7,
        x8, x9, x10, x11, x12, x13, x14, x15,
    ))


def main():
    print("Measuring compute throughput...", flush=True)

    with mp.Pool(processes=P_PHYS) as pool:
        stats = run_benchmark(
            lambda: pool.map(compute_worker, range(P_PHYS)),
            warmup_runs=1,
            timed_runs=3,
        )

    total_flops = P_PHYS * FLOPS_PER_WORKER
    seconds = stats.median_ms / 1000
    gflops = total_flops / seconds / 1e9

    print(f"FLOPs per worker: {FLOPS_PER_WORKER}")
    print(f"Compute median: {stats.median_ms:.3f} ms")
    print(f"Compute IQR: {stats.iqr_ms:.3f} ms")
    print(f"Python compute throughput: {gflops:.6f} GFLOP/s")


if __name__ == "__main__":
    mp.freeze_support()
    main()