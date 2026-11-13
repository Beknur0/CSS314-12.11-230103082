import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
import sys

from harness import emit_verify_tag, run_benchmark


M = 16_000_000
BLOCK_SIZE = 4096
P_PHYS = 6
STRIDES = (1, 2, 4, 8, 16, 32)
MASK = 0xFFFFFFFFFFFFFFFF


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    try:
        acc = arr[slot_idx]
        for i in range(iters):
            acc = (acc * 6364136223846793005 + (i | 1)) & MASK
            arr[slot_idx] = acc
    finally:
        del arr
        shm.close()


def run_trial(shm, p: int, stride: int) -> None:
    if (p - 1) * stride >= 512:
        raise ValueError("Ячейки не помещаются в блок SharedMemory")

    shm.buf[:] = bytes(BLOCK_SIZE)

    # Распределяем и остаток, чтобы суммарно всегда было ровно M шагов.
    base, remainder = divmod(M, p)
    processes = []

    for worker_id in range(p):
        iters = base + (1 if worker_id < remainder else 0)
        process = mp.Process(
            target=fs_worker,
            args=(shm.name, worker_id * stride, iters),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"Процесс {process.pid} завершился с кодом {process.exitcode}"
            )


def measure(shm, p: int, stride: int):
    return run_benchmark(lambda: run_trial(shm, p, stride))


def scaling_counts(p_log: int) -> list[int]:
    counts = []
    p = 1
    while p <= p_log:
        counts.append(p)
        p *= 2
    if counts[-1] != p_log:
        counts.append(p_log)
    return counts


def main():

    student_id = 230103082
    p_log = mp.cpu_count()
    shm = shared_memory.SharedMemory(create=True, size=BLOCK_SIZE)

    try:
        sweep = {}

        print("\n=== Stride sweep: p =", P_PHYS, "===")
        print("stride  bytes    median_ms     IQR_ms    Mops/s")

        for stride in STRIDES:
            stats = measure(shm, P_PHYS, stride)
            sweep[stride] = stats
            throughput = M / (stats.median_ms * 1000)
            print(
                f"{stride:6d} {stride * 8:6d}"
                f" {stats.median_ms:12.2f} {stats.iqr_ms:10.2f}"
                f" {throughput:9.2f}",
                flush=True,
            )

        print("\n=== Process scaling ===")
        print("stride    p    median_ms    speedup  efficiency  Karp-Flatt")

        for stride in (1, 16):
            baseline = None
            for p in scaling_counts(p_log):
                stats = measure(shm, p, stride)

                if baseline is None:
                    baseline = stats.median_ms

                speedup = baseline / stats.median_ms
                efficiency = speedup / p
                if p == 1:
                    karp_flatt = "—"
                else:
                    e = (1 / speedup - 1 / p) / (1 - 1 / p)
                    karp_flatt = f"{e:.4f}"

                print(
                    f"{stride:6d} {p:4d} {stats.median_ms:12.2f}"
                    f" {speedup:10.3f} {efficiency:11.3f}"
                    f" {karp_flatt:>11}",
                    flush=True,
                )

        print()
        print(
            emit_verify_tag(
                "PHASE1",
                student_id,
                sweep[1].median_ms,
                sweep[16].median_ms,
            )
        )
    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()