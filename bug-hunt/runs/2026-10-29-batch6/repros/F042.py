# F042 repro: float sum() of the same data / the same .jix file depends on the CPU cache sizes the
# host reports (ArrayParams::tune defaults block_size = L1d and read_size = (max(L1d/2, L2/16,
# block), L2/2), jix/src/storage/params.rs:366-367,452-459; read_size is not stored in the archive).
#
# Each "host" runs in a child process whose /sys/devices/system/cpu/cpu0/cache/index{0,2}/size are
# replaced by bind mounts in a private mount namespace (`unshare -m`, needs root), so jix's real
# detection (jix/src/util/cpu_cache.rs) sees that host's L1d / L2. Nothing else differs: same
# interpreter, same extension, same data, same file bytes.
#
# Asserts the CORRECT behavior (all hosts agree bit for bit), so it fails while the bug exists.
import os
import subprocess
import sys
import tempfile

import numpy as np

import jix

HOSTS = {
    # name: (index0 = L1d size, index2 = L2 size), as sysfs prints them
    "L1d 48K, L2 2M (e.g. Golden Cove)": ("48K", "2048K"),
    "L1d 32K, L2 512K (e.g. Zen3)": ("32K", "512K"),
    "L1d 32K, L2 1M (e.g. Zen4)": ("32K", "1024K"),
    "L1d 128K, L2 12M (e.g. Apple M1)": ("128K", "12288K"),
}
CACHE = "/sys/devices/system/cpu/cpu0/cache"


def data():
    return np.random.default_rng(1).standard_normal(3_000_000).astype(np.float32)


def child(path):
    x = data()
    s_plain = jix.asarray(x).sum().item()
    s_file = jix.read_array(path).sum().item()
    print(f"{np.float32(s_plain).view(np.uint32)} {np.float32(s_file).view(np.uint32)}")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
        return
    with open(f"{CACHE}/index0/level") as f0, open(f"{CACHE}/index2/level") as f2:
        assert f0.read().strip() == "1" and f2.read().strip() == "2", "unexpected sysfs layout"
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "x.jix")
    # Written once, with a fixed 48 KiB block, as a writer with L1d = 48K would.
    jix.write_array(jix.compact(data(), params={"block_size": 48 * 1024}), path)

    rows = {}
    for name, (l1, l2) in HOSTS.items():
        f1, f2 = os.path.join(tmp, "l1"), os.path.join(tmp, "l2")
        with open(f1, "w") as f:
            f.write(l1 + "\n")
        with open(f2, "w") as f:
            f.write(l2 + "\n")
        script = (
            f"mount --bind {f1} {CACHE}/index0/size && mount --bind {f2} {CACHE}/index2/size && "
            f"exec {sys.executable} {os.path.abspath(__file__)} --child {path}"
        )
        out = subprocess.run(["unshare", "-m", "sh", "-c", script], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        b_plain, b_file = (np.uint32(int(v)) for v in out.stdout.split())
        rows[name] = (b_plain.view(np.float32), b_file.view(np.float32))
        p, q = (float(v) for v in rows[name])
        print(f"{name:<36} asarray(x).sum() = {p!r:<20} read_array(file).sum() = {q!r}")

    x = data()
    print(f"exact (f64) = {float(x.astype(np.float64).sum())!r}, numpy f32 sum = {float(x.sum())!r}")
    plain = {v[0] for v in rows.values()}
    file = {v[1] for v in rows.values()}
    assert len(plain) == 1, f"jix.asarray(x).sum() depends on the host's cache sizes: {sorted(map(float, plain))}"
    assert len(file) == 1, f"read_array(file).sum() depends on the reader's cache sizes: {sorted(map(float, file))}"


if __name__ == "__main__":
    main()
