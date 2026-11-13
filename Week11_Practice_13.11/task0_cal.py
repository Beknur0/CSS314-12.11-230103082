import multiprocessing as mp
import sys
import time

from harness import emit_verify_tag


def main():

    student_id = 230103082
    logical_cpus = mp.cpu_count()
    calls = 1_000_000

    start = time.perf_counter_ns()
    for _ in range(calls):
        time.perf_counter_ns()
    elapsed_ns = time.perf_counter_ns() - start

    average_ns = elapsed_ns / calls

    print(f"Логических CPU: {logical_cpus}")
    print(f"Среднее время вызова: {average_ns:.2f} нс")
    print(emit_verify_tag("PHASE0", student_id, logical_cpus, average_ns))


if __name__ == "__main__":
    main()