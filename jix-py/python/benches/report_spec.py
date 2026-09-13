"""The report's plot specs.

One entry per plot. Benchmarks tag their results with a ``section`` and a ``case`` from here, and
the renderer reads the same file, so the x axis of every plot is declared in exactly one place.

Canonical array is ``[130_000, 200]`` everywhere, Rust and Python alike. Math benchmarks run both
``f32`` and ``i32``; reads run ``i32`` only.

``jix`` and ``blosc2`` always mean the byte-shuffled build - that is the configuration anyone would
actually use. The unfiltered variants appear only in the compression section, where the filter is
what is being measured, and are named ``-noshuffle`` there so the plain names never change meaning.
"""

SHAPE = (130_000, 200)
ITEMSIZE = 4
RAW_BYTES = SHAPE[0] * SHAPE[1] * ITEMSIZE
RAW_MB = RAW_BYTES / 1e6
ARRAY = f"[{SHAPE[0]}, {SHAPE[1]}] = {RAW_MB:.0f} MB"

PLATFORMS = ["macos-aarch64", "linux-x86_64", "linux-aarch64"]

# Every non-compression plot draws these, plus whatever else that plot is about.
CORE = ["numpy", "jix-plain", "jix", "blosc2", "zarr"]
DISTRIBUTIONS = ["random", "smooth", "16 unique", "4 unique"]


def _read_case(block, read, rows, cols):
    """An x tick for the read plot: block shape, read shape, and the bytes that read returns."""
    size = rows * cols * ITEMSIZE
    human = f"{size / 1e6:.0f} MB" if size >= 1e6 else (f"{size / 1e3:.0f} KB" if size >= 1e3 else f"{size} B")
    return f"block {block}\nread {read}\n{human}"


SECTIONS = [
    {
        "key": "read",
        "title": "Reading a random region",
        "subtitle": f"i32 {ARRAY}; every call reads a different one of 1024 random regions",
        "baseline": "numpy",
        "metric": "time",
        "cases": [
            _read_case("16x200", "1x200", 1, 200),
            _read_case("16x200", "16x200", 16, 200),
            _read_case("64x200", "16x16", 16, 16),
            _read_case("64x200", "256x200", 256, 200),
            _read_case("64x200", "4096x200", 4096, 200),
            _read_case("64x200", "whole array", *SHAPE),
        ],
        "libraries": ["numpy", "jix-plain", "jix", "blosc2", "blosc2-chunked", "zarr"],
    },
    {
        "key": "compress_ratio",
        "title": "Compressed size",
        "subtitle": f"i32 {ARRAY}; stored bytes as a fraction of the raw array (prod(shape) * itemsize)",
        "baseline": "numpy",
        "metric": "stored",
        "per_platform": False,  # a stored-byte count does not vary by CPU
        "cases": DISTRIBUTIONS,
        "libraries": ["numpy", "jix-noshuffle", "jix", "blosc2-noshuffle", "blosc2", "zarr"],
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
        "libraries": ["jix-noshuffle", "jix", "blosc2-noshuffle", "blosc2", "zarr"],
    },
    {
        "key": "negate",
        "title": "Negate",
        "subtitle": f"{ARRAY} as f32, smooth data",
        "baseline": "numpy",
        "metric": "time",
        "cases": ["f32", "i32"],
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
        "subtitle": f"{ARRAY} as f32, smooth data, two independent operands",
        "baseline": "numpy",
        "metric": "time",
        "cases": ["f32", "i32"],
        "libraries": CORE,
    },
    {
        "key": "reduction",
        "title": "Reductions",
        "subtitle": f"{ARRAY} as f32, smooth data",
        "baseline": "numpy",
        "metric": "time",
        "cases": [
            "sum f32\naxis 0",
            "sum f32\naxis 1",
            "sum f32\nall",
            "sum i32\naxis 0",
            "sum i32\nall",
            "std f32\naxis 0",
            "std f32\nall",
        ],
        "libraries": CORE,
    },
    {
        "key": "chain",
        "title": "Operation chains",
        "subtitle": f"f32 {ARRAY}; cheap elementwise steps, plus one chain dominated by exp/log",
        "baseline": "numpy",
        "metric": "time",
        "cases": ["1 op", "2 ops", "4 ops", "8 ops", "exp/log"],
        "libraries": ["numpy", "jix-plain", "jix"],
    },
    {
        "key": "chain_memory",
        "title": "Operation chains: peak memory",
        "subtitle": f"f32 {ARRAY}; peak RSS of a fresh subprocess running the same chain",
        "baseline": "numpy",
        "metric": "bytes",
        "cases": ["1 op", "2 ops", "4 ops", "8 ops"],
        "libraries": ["numpy", "jix-plain", "jix"],
    },
    {
        "key": "rust_read",
        "title": "Rust: reading a random region",
        "subtitle": f"i32 {ARRAY}; ndarray reads by slicing an array held whole in memory",
        "baseline": "ndarray",
        "metric": "time",
        "cases": [
            _read_case("32x32", "32x32", 32, 32),
            _read_case("32x32", "1x200", 1, 200),
            _read_case("512x32", "128x200", 128, 200),
        ],
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
        "libraries": ["ndarray", "jix-plain", "jix"],
    },
    {
        "key": "rust_chain",
        "title": "Rust: operation chains",
        "subtitle": f"f32 {ARRAY}; ndarray allocates one intermediate per step",
        "baseline": "ndarray",
        "metric": "time",
        "cases": ["1 op", "2 ops", "4 ops", "8 ops", "normalize\naxis 0", "normalize\naxis 1"],
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
        "libraries": ["ndarray", "jix-plain"],
    },
]

BY_KEY = {section["key"]: section for section in SECTIONS}
