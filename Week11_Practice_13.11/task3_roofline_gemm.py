import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import emit_verify_tag, run_benchmark


STUDENT_ID = 230103082
N = 256
P_PHYS = 6
TILE_SIZES = (8, 16, 32, 64, 96, 128)
MATRIX_BYTES = N * N * 8

# Заполняются отдельно в каждом процессе пула.
worker_shms = []
A = None
B = None
C = None


def init_worker(a_name: str, b_name: str, c_name: str) -> None:
    global worker_shms, A, B, C

    worker_shms = [
        shared_memory.SharedMemory(name=a_name),
        shared_memory.SharedMemory(name=b_name),
        shared_memory.SharedMemory(name=c_name),
    ]
    matrix_type = ctypes.c_double * (N * N)
    A = matrix_type.from_buffer(worker_shms[0].buf)
    B = matrix_type.from_buffer(worker_shms[1].buf)
    C = matrix_type.from_buffer(worker_shms[2].buf)


def gemm_worker(job: tuple[str, int, int, int]) -> None:
    variant, row_start, row_end, bs = job

    if variant == "naive":
        # i-j-k: B читается с шагом N.
        for i in range(row_start, row_end):
            row = i * N
            for j in range(N):
                value = 0.0
                for k in range(N):
                    value += A[row + k] * B[k * N + j]
                C[row + j] = value

    elif variant == "reordered":
        # i-k-j: последовательный проход по строкам B и C.
        for i in range(row_start, row_end):
            row = i * N
            for k in range(N):
                aik = A[row + k]
                b_row = k * N
                for j in range(N):
                    C[row + j] += aik * B[b_row + j]

    elif variant == "blocked":
        # Плитки ii-jj-kk; внутри плитки порядок i-k-j.
        for ii in range(row_start, row_end, bs):
            i_end = min(ii + bs, row_end)
            for jj in range(0, N, bs):
                j_end = min(jj + bs, N)
                for kk in range(0, N, bs):
                    k_end = min(kk + bs, N)
                    for i in range(ii, i_end):
                        row = i * N
                        for k in range(kk, k_end):
                            aik = A[row + k]
                            b_row = k * N
                            for j in range(jj, j_end):
                                C[row + j] += aik * B[b_row + j]
    else:
        raise ValueError(f"Unknown variant: {variant}")


def make_jobs(variant: str, bs: int = 0):
    return [
        (variant, N * worker // P_PHYS, N * (worker + 1) // P_PHYS, bs)
        for worker in range(P_PHYS)
    ]


def main() -> None:
    blocks = []

    try:
        for _ in range(3):
            blocks.append(shared_memory.SharedMemory(
                create=True, size=MATRIX_BYTES
            ))

        a_shm, b_shm, c_shm = blocks
        matrix_type = ctypes.c_double * (N * N)
        a = matrix_type.from_buffer(a_shm.buf)
        b = matrix_type.from_buffer(b_shm.buf)
        c = matrix_type.from_buffer(c_shm.buf)

        try:
            for i in range(N * N):
                a[i] = (i % 17) / 17.0
                b[i] = (i % 13) / 13.0

            zero_matrix = bytes(MATRIX_BYTES)

            with mp.Pool(
                processes=P_PHYS,
                initializer=init_worker,
                initargs=(a_shm.name, b_shm.name, c_shm.name),
            ) as pool:

                def measure(variant: str, bs: int = 0):
                    jobs = make_jobs(variant, bs)

                    def trial():
                        # Обнуление включено во время всех вариантов,
                        # в том числе naive, для единообразия.
                        c_shm.buf[:] = zero_matrix
                        pool.map(gemm_worker, jobs)

                    stats = run_benchmark(trial)
                    gflops = (2 * N**3) / (stats.median_ms * 1e6)
                    return stats, gflops

                print("=== GEMM: N=256, workers=6 ===", flush=True)

                naive_stats, naive_gflops = measure("naive")
                print(
                    f"Naive: median={naive_stats.median_ms:.2f} ms, "
                    f"IQR={naive_stats.iqr_ms:.2f} ms, "
                    f"GFLOP/s={naive_gflops:.4f}",
                    flush=True,
                )

                reordered_stats, reordered_gflops = measure("reordered")
                reference = list(c)
                print(
                    f"Reordered: median={reordered_stats.median_ms:.2f} ms, "
                    f"IQR={reordered_stats.iqr_ms:.2f} ms, "
                    f"GFLOP/s={reordered_gflops:.4f}",
                    flush=True,
                )

                print("\n=== Blocked tile sweep ===", flush=True)
                print(
                    " BS   median_ms     IQR_ms   GFLOP/s"
                    "   footprint_B   intensity   max_error",
                    flush=True,
                )

                results = {}
                for bs in TILE_SIZES:
                    stats, gflops = measure("blocked", bs)
                    max_error = max(
                        abs(c[i] - reference[i]) for i in range(N * N)
                    )
                    if max_error >= 1e-9:
                        raise RuntimeError(
                            f"BS={bs}: max error {max_error} >= 1e-9"
                        )

                    results[bs] = (stats, gflops)
                    print(
                        f"{bs:3d} {stats.median_ms:11.2f}"
                        f" {stats.iqr_ms:10.2f} {gflops:9.4f}"
                        f" {3 * bs * bs * 8:13d}"
                        f" {bs / 12:11.2f} {max_error:11.3e}",
                        flush=True,
                    )

                best_bs = min(
                    results, key=lambda bs: results[bs][0].median_ms
                )
                best_gflops = results[best_bs][1]

                print(f"\nBest BS: {best_bs}")
                print(
                    emit_verify_tag(
                        "PHASE3", STUDENT_ID, naive_gflops, best_gflops
                    )
                )

        finally:
            # Освобождаем ctypes-представления до закрытия SharedMemory.
            del a, b, c

    finally:
        for block in blocks:
            block.close()
            block.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()