import multiprocessing as mp
import time

from harness import emit_verify_tag


def main():
    student_id = 230103082
    reads = 2_000_000

    start = time.perf_counter_ns()
    for _ in range(reads):
        time.perf_counter_ns()
    elapsed_ns = time.perf_counter_ns() - start

    avg_ns = elapsed_ns / reads
    logical_cpus = mp.cpu_count()

    print(f"Clock reads: {reads}")
    print(f"Logical CPUs: {logical_cpus}")
    print(f"Average clock-call latency: {avg_ns:.4f} ns")
    print(f"Primary metric: {avg_ns:.4f}")
    print(f"Secondary metric: {logical_cpus:.4f}")
    print(emit_verify_tag("PHASE0", student_id, avg_ns, logical_cpus))


if __name__ == "__main__":
    main()