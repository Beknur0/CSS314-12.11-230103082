import multiprocessing as mp

from harness import run_benchmark


P_PHYS = 6
FLOPS_PER_WORKER = 60_000_000

# 16 выражений вида a*b+c, по 2 FLOP каждое.
ITERATIONS = FLOPS_PER_WORKER // 32


def compute_worker(_: int) -> float:
    a0 = a1 = a2 = a3 = 0.1
    a4 = a5 = a6 = a7 = 0.2
    a8 = a9 = a10 = a11 = 0.3
    a12 = a13 = a14 = a15 = 0.4

    b = 0.999999
    c = 0.000001

    for _ in range(ITERATIONS):
        a0 = a0 * b + c
        a1 = a1 * b + c
        a2 = a2 * b + c
        a3 = a3 * b + c
        a4 = a4 * b + c
        a5 = a5 * b + c
        a6 = a6 * b + c
        a7 = a7 * b + c
        a8 = a8 * b + c
        a9 = a9 * b + c
        a10 = a10 * b + c
        a11 = a11 * b + c
        a12 = a12 * b + c
        a13 = a13 * b + c
        a14 = a14 * b + c
        a15 = a15 * b + c

    return (
        a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
        + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15
    )


def main() -> None:
    with mp.Pool(processes=P_PHYS) as pool:

        def trial() -> None:
            results = pool.map(compute_worker, range(P_PHYS))
            if len(results) != P_PHYS:
                raise RuntimeError("Не все процессы вернули результат")

        stats = run_benchmark(trial)

    total_flops = P_PHYS * FLOPS_PER_WORKER
    gflops = total_flops / (stats.median_ms / 1000) / 1e9

    print(f"Workers: {P_PHYS}")
    print(f"FLOPs per worker: {FLOPS_PER_WORKER}")
    print(f"Median: {stats.median_ms:.2f} ms")
    print(f"IQR: {stats.iqr_ms:.2f} ms")
    print(f"Compute throughput: {gflops:.4f} GFLOP/s")


if __name__ == "__main__":
    mp.freeze_support()
    main()