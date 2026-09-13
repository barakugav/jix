"""The report's plot specs.

One entry per plot. Benchmarks tag their results with a ``section`` and a ``case`` from here, and
the renderer reads the same file, so the x axis of every plot is declared in exactly one place.

Canonical array is ``[130_000, 200]`` everywhere, Rust and Python alike. Math benchmarks run both
``f32`` and ``i32``; reads run ``i32`` only.
"""

SHAPE = (130_000, 200)
PLATFORMS = ["macos-aarch64", "linux-x86_64", "linux-aarch64"]

COMPRESSED = ["jix", "jix-shuffle", "blosc2", "blosc2-shuffle", "zarr"]
ALL_PY = ["numpy", "jix-plain", *COMPRESSED]
DISTRIBUTIONS = ["random", "smooth", "16 unique", "4 unique"]

SECTIONS = [
    {
        "key": "read",
        "title": "Reading a random region",
        "subtitle": "i32 [130000, 200]; every call reads a different one of 1024 random regions",
        "baseline": "numpy",
        "metric": "time",
        "cases": [
            "b16x200\nr1x200",
            "b16x200\nr16x200",
            "b64x200\nr16x16",
            "b64x200\nr256x200",
            "b64x200\nr4096x200",
            "b64x200\nfull",
        ],
        "libraries": ["numpy", "jix", "jix-shuffle", "blosc2", "blosc2-chunked", "zarr"],
    },
    {
        "key": "compress_ratio",
        "title": "Compression ratio",
        "subtitle": "i32 [130000, 200]; stored bytes against the raw array (prod(shape) * itemsize)",
        "baseline": "numpy",
        "metric": "ratio",
        "per_platform": False,  # a stored-byte count does not vary by CPU
        "cases": DISTRIBUTIONS,
        "libraries": ALL_PY[:1] + COMPRESSED,
    },
    {
        "key": "compress",
        "title": "Compression throughput",
        "subtitle": "i32 [130000, 200]; numpy is a plain memcpy of the same array",
        "baseline": "numpy",
        "metric": "time",
        "cases": DISTRIBUTIONS,
        "libraries": ALL_PY[:1] + COMPRESSED,
    },
    {
        "key": "negate",
        "title": "Negate",
        "subtitle": "[130000, 200], smooth data",
        "baseline": "numpy",
        "metric": "time",
        "cases": ["f32", "i32"],
        "libraries": ALL_PY,
    },
    {
        "key": "negate_dist",
        "title": "Negate, by how well the data compresses",
        "subtitle": "f32 [130000, 200]; same operation, same code path, only the data changes",
        "baseline": "numpy",
        "metric": "time",
        "cases": DISTRIBUTIONS,
        "libraries": ["numpy", "jix-shuffle", "blosc2-shuffle"],
    },
    {
        "key": "add",
        "title": "Add",
        "subtitle": "[130000, 200], smooth data, two independent operands",
        "baseline": "numpy",
        "metric": "time",
        "cases": ["f32", "i32"],
        "libraries": ALL_PY,
    },
    {
        "key": "reduction",
        "title": "Reductions",
        "subtitle": "[130000, 200], smooth data",
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
        "libraries": ["numpy", "jix-plain", "jix-shuffle", "blosc2-shuffle", "zarr"],
    },
    {
        "key": "chain",
        "title": "Operation chains",
        "subtitle": "[130000, 200]; cheap elementwise steps, plus one chain dominated by exp/log",
        "baseline": "numpy",
        "metric": "time",
        "cases": [
            "f32\n1 op",
            "f32\n2 ops",
            "f32\n4 ops",
            "f32\n8 ops",
            "f32\nexp/log",
            "i32\n1 op",
            "i32\n4 ops",
            "i32\n8 ops",
        ],
        "libraries": ["numpy", "jix-plain", "jix-shuffle"],
    },
    {
        "key": "chain_memory",
        "title": "Operation chains: peak memory",
        "subtitle": "peak RSS of a fresh subprocess running the same chain",
        "baseline": "numpy",
        "metric": "bytes",
        "cases": ["f32\n1 op", "f32\n2 ops", "f32\n4 ops", "f32\n8 ops"],
        "libraries": ["numpy", "jix-plain", "jix-shuffle"],
    },
    {
        "key": "rust_read",
        "title": "Rust: reading a random region",
        "subtitle": "i32 [130000, 200]; ndarray reads by slicing an array held whole in memory",
        "baseline": "ndarray",
        "metric": "time",
        "cases": ["b32x32\nr32x32", "b32x32\nr1x200", "b512x32\nr128x200"],
        "libraries": ["ndarray", "jix", "jix-shuffle"],
    },
    {
        "key": "rust_op",
        "title": "Rust: elementwise and reductions",
        "subtitle": "[130000, 200], smooth data",
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
        "libraries": ["ndarray", "jix-plain", "jix-shuffle"],
    },
    {
        "key": "rust_chain",
        "title": "Rust: operation chains",
        "subtitle": "[130000, 200] f32; ndarray allocates one intermediate per step",
        "baseline": "ndarray",
        "metric": "time",
        "cases": ["1 op", "2 ops", "4 ops", "8 ops", "normalize\naxis 0", "normalize\naxis 1"],
        "libraries": ["ndarray", "jix-plain"],
    },
    {
        "key": "rust_axis_order",
        "title": "Rust: axis order",
        "subtitle": "negate on a 3-D array; ndarray classifies layout four ways and has no axis sort",
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
