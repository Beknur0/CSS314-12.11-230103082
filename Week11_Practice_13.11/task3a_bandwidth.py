import multiprocessing as mp
from multiprocessing import shared_memory

from harness import run_benchmark


P_PHYS = 6
BUFFER_SIZE = 64_000_000  # 64 MB в десятичных байтах

# Эти переменные инициализируются отдельно в каждом процессе пула.
source_shm = None
destination_shm = None


def init_worker(source_name: str, destination_name: str) -> None:
    global source_shm, destination_shm
    source_shm = shared_memory.SharedMemory(name=source_name)
    destination_shm = shared_memory.SharedMemory(name=destination_name)


def copy_slice(bounds: tuple[int, int]) -> int:
    start, end = bounds
    destination_shm.buf[start:end] = source_shm.buf[start:end]
    return end - start


def main() -> None:
    source = shared_memory.SharedMemory(create=True, size=BUFFER_SIZE)

    try:
        destination = shared_memory.SharedMemory(
            create=True, size=BUFFER_SIZE
        )
        try:
            # Инициализация выполняется до замера.
            source.buf[:] = b"\x01" * BUFFER_SIZE
            destination.buf[:] = b"\x00" * BUFFER_SIZE

            partitions = [
                (BUFFER_SIZE * i // P_PHYS,
                 BUFFER_SIZE * (i + 1) // P_PHYS)
                for i in range(P_PHYS)
            ]

            with mp.Pool(
                processes=P_PHYS,
                initializer=init_worker,
                initargs=(source.name, destination.name),
            ) as pool:

                def trial() -> None:
                    copied = sum(pool.map(copy_slice, partitions))
                    if copied != BUFFER_SIZE:
                        raise RuntimeError("Скопирован неверный объём данных")

                stats = run_benchmark(trial)

            # Проверка после измерения.
            if destination.buf != source.buf:
                raise RuntimeError("Буферы после копирования не совпадают")

            # Считаем чтение и запись: 2 × 64 MB.
            bandwidth = (2 * BUFFER_SIZE) / (stats.median_ms / 1000) / 1e9

            print(f"Workers: {P_PHYS}")
            print(f"Buffer size: {BUFFER_SIZE} bytes each")
            print(f"Median: {stats.median_ms:.2f} ms")
            print(f"IQR: {stats.iqr_ms:.2f} ms")
            print(f"Copy bandwidth: {bandwidth:.2f} GB/s")

        finally:
            destination.close()
            destination.unlink()
    finally:
        source.close()
        source.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()