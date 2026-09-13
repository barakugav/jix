"""The report's plot specs, and the parameters the benchmarks run.

One entry per plot. The benchmarks parametrize themselves from the case lists here and tag their
results with the labels here, and the renderer reads the same file - so an x tick label and the
configuration that produced it cannot drift apart.

Canonical array is ``[130_000, 200]`` everywhere, Rust and Python alike. Math benchmarks run both
``f32`` and ``i32``; reads run ``i32`` only.

``jix`` and ``blosc2`` always mean the byte-shuffled build - that is the configuration anyone would
actually use. The unfiltered variants appear only in the compression section, where the filter is
what is being measured, and are named ``-noshuffle`` there so the plain names never change meaning.
"""

SHAPE = (130_000, 200)
ITEMSIZE = 4
RAW_BYTES = SHAPE[0] * SHAPE[1] * ITEMSIZE
ARRAY = f"[{SHAPE[0]}, {SHAPE[1]}] = {RAW_BYTES / 1e6:.0f} MB"

PLATFORMS = ["linux-x86_64", "linux-aarch64"]

# Every non-compression plot draws these, plus whatever else that plot is about.
CORE = ["numpy", "jix-plain", "jix", "blosc2", "zarr"]
READ_LIBRARIES = ["numpy", "jix-plain", "jix", "blosc2", "blosc2-chunked", "zarr"]
COMPRESS_LIBRARIES = ["numpy", "jix-noshuffle", "jix", "blosc2-noshuffle", "blosc2", "zarr"]

# Data distributions. `smooth` is a sampled sine-plus-gradient field; `random` is uniform over the
# whole dtype range; the rest are literal distinct-value counts.
DISTRIBUTIONS = ["random", "smooth", "16 unique", "4 unique"]

DTYPES = ["f32", "i32"]


def _size(nbytes):
    if nbytes >= 1e6:
        return f"{nbytes / 1e6:.0f} MB"
    return f"{nbytes / 1e3:.0f} KB" if nbytes >= 1e3 else f"{nbytes} B"


# (block shape, read shape). A read shape of `None` means the whole array.
READ_CASES = [
    ((16, 200), (1, 200)),
    ((16, 200), (16, 200)),
    ((64, 200), (16, 16)),
    ((64, 200), (256, 200)),
    ((64, 200), (4096, 200)),
    ((64, 200), None),
]
RUST_READ_CASES = [
    ((32, 32), (32, 32)),
    ((32, 32), (1, 200)),
    ((512, 32), (128, 200)),
]


def read_label(block, read):
    """The x tick for one read case: block shape, read shape, and the bytes the read returns."""
    region = SHAPE if read is None else read
    name = "whole array" if read is None else f"{read[0]}x{read[1]}"
    return f"block {block[0]}x{block[1]}\nread {name}\n{_size(region[0] * region[1] * ITEMSIZE)}"


# (reduction, dtype, axis)
REDUCTION_CASES = [
    ("sum", "f32", 0),
    ("sum", "f32", 1),
    ("sum", "f32", None),
    ("sum", "i32", 0),
    ("sum", "i32", None),
    ("std", "f32", 0),
    ("std", "f32", None),
]


def reduction_label(op, dtype, axis):
    return f"{op} {dtype}\n{'all' if axis is None else f'axis {axis}'}"


# Number of cheap elementwise steps, then the transcendental chain that behaves differently.
CHAIN_CASES = [1, 2, 4, 8, "exp/log"]


def chain_label(steps):
    if steps == "exp/log":
        return "exp/log"
    return "1 op" if steps == 1 else f"{steps} ops"


SECTIONS = [
    {
        "key": "read",
        "title": "Reading a random region",
        "subtitle": f"i32 {ARRAY}; every call reads a different one of 1024 random regions",
        "baseline": "numpy",
        "metric": "time",
        "cases": [read_label(*case) for case in READ_CASES],
        "libraries": READ_LIBRARIES,
    },
    {
        "key": "compress_ratio",
        "title": "Compressed size",
        "subtitle": f"i32 {ARRAY}; stored bytes as a fraction of the raw array (prod(shape) * itemsize)",
        "baseline": "numpy",
        "metric": "stored",
        "per_platform": False,  # a stored-byte count does not vary by CPU
        "cases": DISTRIBUTIONS,
        "libraries": COMPRESS_LIBRARIES,
    },
    {
        "key": "compress",
        "title": "Compression throughput",
        "subtitle": f"i32 {ARRAY}; original array bytes processed per second",
        "baseline": None,
        "metric": "throughput",
        "raw_bytes": RAW_BYTES,
        "unit": "MB/s",
        "cases": DISTRIBUTIONS,
        "libraries": [lib for lib in COMPRESS_LIBRARIES if lib != "numpy"],
    },
    {
        "key": "negate",
        "title": "Negate",
        "subtitle": f"{ARRAY}, smooth data",
        "baseline": "numpy",
        "metric": "time",
        "cases": DTYPES,
        "libraries": CORE,
    },
    {
        "key": "negate_dist",
        "title": "Negate, by how well the data compresses",
        "subtitle": f"f32 {ARRAY}; same operation, same code path, only the data changes",
        "baseline": "numpy",
        "metric": "time",
        "cases": DISTRIBUTIONS,
        "libraries": CORE,
    },
    {
        "key": "add",
        "title": "Add",
        "subtitle": f"{ARRAY}, smooth data, two independent operands",
        "baseline": "numpy",
        "metric": "time",
        "cases": DTYPES,
        "libraries": CORE,
    },
    {
        "key": "reduction",
        "title": "Reductions",
        "subtitle": f"{ARRAY}, smooth data",
        "baseline": "numpy",
        "metric": "time",
        "cases": [reduction_label(*case) for case in REDUCTION_CASES],
        "libraries": CORE,
    },
    {
        "key": "chain",
        "title": "Operation chains",
        "subtitle": f"f32 {ARRAY}; cheap elementwise steps, plus one chain dominated by exp/log",
        "baseline": "numpy",
        "metric": "time",
        "cases": [chain_label(case) for case in CHAIN_CASES],
        "libraries": ["numpy", "jix-plain", "jix"],
    },
    {
        "key": "chain_memory",
        "title": "Operation chains: peak memory",
        "subtitle": f"f32 {ARRAY}; peak RSS of a fresh subprocess running the same chain",
        "baseline": "numpy",
        "metric": "bytes",
        "cases": [chain_label(case) for case in CHAIN_CASES if case != "exp/log"],
        "libraries": ["numpy", "jix-plain", "jix"],
    },
    {
        "key": "rust_read",
        "title": "Rust: reading a random region",
        "subtitle": f"i32 {ARRAY}; ndarray reads by slicing an array held whole in memory",
        "baseline": "ndarray",
        "metric": "time",
        "cases": [read_label(*case) for case in RUST_READ_CASES],
        "case_ids": ["b32x32_r32x32", "b32x32_r1x200", "b512x32_r128x200"],
        "libraries": ["ndarray", "jix-plain", "jix"],
    },
    {
        "key": "rust_op",
        "title": "Rust: elementwise and reductions",
        "subtitle": f"{ARRAY}, smooth data",
        "baseline": "ndarray",
        "metric": "time",
        "cases": [
            "negate\nf32",
            "negate\ni32",
            "add\nf32",
            "add\ni32",
            "sum f32\naxis 0",
            "sum f32\naxis 1",
            "sum i32\nall",
        ],
        "case_ids": [
            "negate_f32",
            "negate_i32",
            "add_f32",
            "add_i32",
            "sum_f32_axis0",
            "sum_f32_axis1",
            "sum_i32_all",
        ],
        "libraries": ["ndarray", "jix-plain", "jix"],
    },
    {
        "key": "rust_chain",
        "title": "Rust: operation chains",
        "subtitle": f"f32 {ARRAY}; ndarray allocates one intermediate per step",
        "baseline": "ndarray",
        "metric": "time",
        "cases": ["1 op", "2 ops", "4 ops", "8 ops", "normalize\naxis 0", "normalize\naxis 1"],
        "case_ids": ["1op", "2op", "4op", "8op", "normalize_axis0", "normalize_axis1"],
        "libraries": ["ndarray", "jix-plain", "jix"],
    },
    {
        "key": "rust_axis_order",
        "title": "Rust: axis order",
        "subtitle": "negate on [300, 400, 500] = 240 MB as f32; ndarray classifies layout four ways",
        "baseline": "ndarray",
        "metric": "time",
        "cases": [
            "3-D rotate\n[1,2,0] f32",
            "3-D rotate\n[1,2,0] i32",
            "3-D reverse\n[2,1,0] f32",
            "2-D\ntranspose f32",
        ],
        "case_ids": ["rotate_f32", "rotate_i32", "reverse_f32", "transpose2d_f32"],
        "libraries": ["ndarray", "jix-plain"],
    },
]

BY_KEY = {section["key"]: section for section in SECTIONS}

# Criterion benchmark ids have to be filesystem-safe, so the Rust suite uses short case ids and
# this maps them back to the labels the plots show.
RUST_CASE_LABELS = {
    (section["key"], case_id): case
    for section in SECTIONS
    if "case_ids" in section
    for case_id, case in zip(section["case_ids"], section["cases"], strict=True)
}
