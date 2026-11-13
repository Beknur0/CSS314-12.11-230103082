import math
import multiprocessing as mp
import time

from harness import emit_verify_tag


STUDENT_ID = 230103082
N = 8_000
P_PHYS = 6
CHUNK_SIZES = (1, 8, 64, 512)


def process_item(i: int) -> float:
    iterations = 1 + (i * i) // 1600
    x = (i % 100 + 1) / 100.0

    for _ in range(iterations):
        x = math.sqrt(abs(math.sin(x) + math.cos(x)) + 1.0)

    return x


def worker(
    worker_id: int,
    strategy: str,
    chunk_size: int,
    counter,
    worker_times,
    checksums,
) -> None:
    start_time = time.perf_counter()
    total = 0.0

    if strategy == "block":
        start = N * worker_id // P_PHYS
        end = N * (worker_id + 1) // P_PHYS
        for i in range(start, end):
            total += process_item(i)

    elif strategy == "cyclic":
        for i in range(worker_id, N, P_PHYS):
            total += process_item(i)

    elif strategy == "dynamic":
        while True:
            with counter.get_lock():
                start = counter.value
                counter.value += chunk_size

            if start >= N:
                break

            for i in range(start, min(start + chunk_size, N)):
                total += process_item(i)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    worker_times[worker_id] = time.perf_counter() - start_time
    checksums[worker_id] = total


def run_trial(strategy: str, chunk_size: int = 0):
    counter = mp.Value("i", 0)
    worker_times = mp.Array("d", P_PHYS)
    checksums = mp.Array("d", P_PHYS)

    processes = [
        mp.Process(
            target=worker,
            args=(
                i, strategy, chunk_size, counter,
                worker_times, checksums,
            ),
        )
        for i in range(P_PHYS)
    ]

    start_time = time.perf_counter()

    for process in processes:
        process.start()

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"Worker {process.pid} failed: {process.exitcode}"
            )

    wall_time = time.perf_counter() - start_time
    times = list(worker_times)
    mean_time = sum(times) / P_PHYS
    imbalance = max(times) / mean_time

    return wall_time, imbalance, times, sum(checksums)


def main() -> None:
    configs = [
        ("block", 0),
        ("cyclic", 0),
        *[("dynamic", chunk) for chunk in CHUNK_SIZES],
    ]
    results = {}

    print(f"N={N}, workers={P_PHYS}", flush=True)
    print("Each configuration runs once; wall time includes process startup.",
          flush=True)

    for strategy, chunk in configs:
        label = strategy if strategy != "dynamic" else f"dynamic C={chunk}"
        print(f"\nRunning {label} ...", flush=True)

        result = run_trial(strategy, chunk)
        results[(strategy, chunk)] = result

        wall, imbalance, times, checksum = result
        print(f"Wall time: {wall:.3f} s", flush=True)
        print(f"Imbalance λ: {imbalance:.3f}", flush=True)
        print(
            "Worker times: " + ", ".join(f"{t:.3f}" for t in times),
            flush=True,
        )
        print(f"Checksum: {checksum:.6f}", flush=True)

    best_chunk = min(
        CHUNK_SIZES, key=lambda c: results[("dynamic", c)][0]
    )
    block_wall = results[("block", 0)][0]
    best_wall = results[("dynamic", best_chunk)][0]

    print(f"\nBest dynamic chunk: {best_chunk}")
    print(
        emit_verify_tag(
            "PHASE4", STUDENT_ID, block_wall, best_wall
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()