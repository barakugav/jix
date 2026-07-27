def record(benchmark, *, case, library, size, **extra):
    """Attach the fields report.py groups by to this benchmark's JSON entry.

    Extra keyword fields (e.g. `raw_bytes`, `stored_bytes`, `ratio` from the compress
    workload) are stashed alongside so report.py can render them without a separate run.
    """
    benchmark.extra_info.update(case=case, library=library, size=int(size), **extra)
