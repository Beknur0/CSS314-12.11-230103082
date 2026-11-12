import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import emit_verify_tag, run_benchmark


TOTAL_OPS = 24_000_000
P_PHYS = 6
STRIDES = (1, 2, 4, 8, 16, 32)


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005

    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc

    del arr
    shm.close()


def run_trial(shm, p: int, stride: int) -> None:
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    for t in range(p):
        arr[t * stride] = 0
    del arr

    base, remainder = divmod(TOTAL_OPS, p)
    workers = [
        mp.Process(
            target=fs_worker,
            args=(shm.name, t * stride, base + (t < remainder)),
        )
        for t in range(p)
    ]

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(f"Worker failed: {worker.exitcode}")


def measure(shm, p: int, stride: int):
    return run_benchmark(
        lambda: run_trial(shm, p, stride),
        warmup_runs=2,
        timed_runs=7,
    )


def main():
    student_id = 230103082
    p_log = mp.cpu_count()
    shm = shared_memory.SharedMemory(create=True, size=4096)

    try:
        print("\nTable 1A: stride sweep", flush=True)
        print("stride  bytes  median_ms  iqr_ms  Mops/s", flush=True)

        stride_results = {}
        for stride in STRIDES:
            stats = measure(shm, P_PHYS, stride)
            stride_results[stride] = stats
            throughput = TOTAL_OPS / (stats.median_ms * 1000)
            print(
                f"{stride:6d} {stride * 8:6d} "
                f"{stats.median_ms:10.3f} {stats.iqr_ms:7.3f} "
                f"{throughput:8.3f}",
                flush=True,
            )

        # Степени двойки до Plog; добавляем Plog, если это не степень двойки.
        counts = []
        p = 1
        while p <= p_log:
            counts.append(p)
            p *= 2
        if counts[-1] != p_log:
            counts.append(p_log)

        print("\nTable 1B: process scaling", flush=True)
        print("stride  p  median_ms  iqr_ms  speedup  efficiency  e(p)", flush=True)

        for stride in (1, 16):
            t1 = None
            for p in counts:
                stats = measure(shm, p, stride)
                if p == 1:
                    t1 = stats.median_ms

                speedup = t1 / stats.median_ms
                efficiency = speedup / p
                e_text = (
                    "-"
                    if p == 1
                    else f"{(1 / speedup - 1 / p) / (1 - 1 / p):.4f}"
                )

                print(
                    f"{stride:6d} {p:2d} {stats.median_ms:10.3f} "
                    f"{stats.iqr_ms:7.3f} {speedup:8.4f} "
                    f"{efficiency:11.4f} {e_text:>8}",
                    flush=True,
                )

        m1 = stride_results[1].median_ms
        m16 = stride_results[16].median_ms
        print(f"\nPHASE1 primary metric: {m1:.4f}")
        print(f"PHASE1 secondary metric: {m16:.4f}")
        print(emit_verify_tag("PHASE1", student_id, m1, m16))

    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()