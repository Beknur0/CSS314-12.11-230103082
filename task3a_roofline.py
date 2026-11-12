import multiprocessing as mp
from multiprocessing import shared_memory

from harness import run_benchmark


SIZE = 96_000_000
P_PHYS = 6


def copy_part(args):
    a_name, b_name, start, end = args
    shm_a = shared_memory.SharedMemory(name=a_name)
    shm_b = shared_memory.SharedMemory(name=b_name)

    try:
        shm_a.buf[start:end] = shm_b.buf[start:end]
    finally:
        shm_a.close()
        shm_b.close()


def main():
    print("Allocating two 96 MB buffers...", flush=True)
    shm_a = shared_memory.SharedMemory(create=True, size=SIZE)
    shm_b = None

    try:
        shm_b = shared_memory.SharedMemory(create=True, size=SIZE)

        # Заполняем страницы до измерения.
        chunk = b"\x01" * 1_000_000
        for start in range(0, SIZE, len(chunk)):
            end = min(start + len(chunk), SIZE)
            shm_a.buf[start:end] = chunk[:end - start]
            shm_b.buf[start:end] = chunk[:end - start]

        tasks = [
            (shm_a.name, shm_b.name, SIZE * t // P_PHYS,
             SIZE * (t + 1) // P_PHYS)
            for t in range(P_PHYS)
        ]

        print("Measuring copies...", flush=True)
        with mp.Pool(processes=P_PHYS) as pool:
            stats = run_benchmark(
                lambda: pool.map(copy_part, tasks),
                warmup_runs=2,
                timed_runs=7,
            )

        # 2 * SIZE — условный объём данных: чтение + запись.
        seconds = stats.median_ms / 1000
        bandwidth = (2 * SIZE) / seconds / 1e9

        print(f"Copy median: {stats.median_ms:.3f} ms")
        print(f"Copy IQR: {stats.iqr_ms:.3f} ms")
        print(f"Copy bandwidth: {bandwidth:.3f} GB/s")

    finally:
        if shm_b is not None:
            shm_b.close()
            shm_b.unlink()
        shm_a.close()
        shm_a.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()