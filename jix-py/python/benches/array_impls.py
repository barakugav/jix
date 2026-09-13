"""One wrapper per library under test, with the operations the report measures.

Each wrapper exposes the same set of operations and every one of them returns a plain NumPy array,
so the comparison is like for like: every library pays for producing an uncompressed result, and
none of them is credited for leaving it compressed.

`jix` and `blosc2` always mean the byte-shuffled build. The unfiltered variants exist only for the
compression section, where the filter is what is being measured.
"""

import io
from typing import ClassVar

import blosc2
import numexpr
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

import jix

ZSTD_LEVEL = 3
NTHREADS = 1

# `read_size` is plumbed through `from_numpy` because it is the knob that decides whether a lazy
# chain's intermediates stay in cache. The benchmarks leave it at the default - see
# bench-report/FINDINGS.md for what tuning it is worth, and why the right value differs between
# compressed and uncompressed storage.
CODEC_DESC = f"zstd level {ZSTD_LEVEL}, byte-shuffle, {NTHREADS} thread (jix, blosc2, zarr matched)"

blosc2.set_nthreads(NTHREADS)
blosc2.nthreads = NTHREADS
# blosc2 evaluates lazy expressions through numexpr, which keeps its own thread pool (one per core
# by default). Pinning only blosc2's threads would leave the elementwise arms multi-threaded while
# jix and numpy run on one core.
numexpr.set_num_threads(NTHREADS)

# The elementwise chain, built from array operands only - no scalar constants.
#
# jix has no specialized loop for a 0-stride operand, so a scalar is read through the same strided
# loop as a real array. A chain of scalar ops therefore measures that gap rather than the thing
# this section is about, which is whether fusing a chain beats materializing each step.
#
# Reusing the two source arrays keeps every step a real binary op. NumPy re-reads an operand and
# writes an intermediate per step; jix reads both arrays once and keeps the running value in
# registers. That difference is the whole point.
#
# Starting from `out = a`, the cycle builds up the shape of an expression like `a*a + b*b + a*b`:
# the first step is a square, and products of both operands accumulate from there. Squares are
# spelled `a*a` rather than `a**2` on purpose - an exponent is a scalar operand, which is exactly
# what this chain is avoiding.
#
# The cycle alternates multiplication with addition and subtraction to keep magnitudes bounded.
# Repeated multiplication alone would decay toward denormals, which are slow on some CPUs and would
# end up measuring that instead.
CHAIN_STEPS = [
    ("out * a", lambda out, a, b: out * a),
    ("out + b", lambda out, a, b: out + b),
    ("out * b", lambda out, a, b: out * b),
    ("out - a", lambda out, a, b: out - a),
]


def chain_steps(count):
    """The first `count` steps of the canonical chain, cycling if more are asked for."""
    return [CHAIN_STEPS[i % len(CHAIN_STEPS)] for i in range(count)]


class AbstractArray:
    """Base implementation. Subclasses set `name` and implement the interface.

    Every operation returns a NumPy array.
    """

    name = "abstract"

    def __init__(self, raw):
        self.raw = raw

    @classmethod
    def from_numpy(cls, data, *, block_shape=None, read_size=None):
        """Build this library's array. `read_size` is a jix-only hint; others ignore it."""
        raise NotImplementedError()

    def read(self, index):
        raise NotImplementedError()

    def negate(self):
        raise NotImplementedError()

    def add(self, other):
        raise NotImplementedError()

    def reduce(self, op, axis):
        raise NotImplementedError()

    def _operand(self):
        """The object the chain's operators are applied to."""
        return self.raw

    def _materialize(self, value):
        """Turn the result of a chain into a NumPy array."""
        raise NotImplementedError()

    def chain(self, other, count):
        """`count` steps of the canonical chain over two arrays.

        Shared rather than reimplemented per library on purpose: the arms have to run identical
        arithmetic, and four copies of this loop is four chances for them to stop doing so.
        """
        a, b = self._operand(), other._operand()
        out = a
        for _, step in chain_steps(count):
            out = step(out, a, b)
        return self._materialize(out)

    def exp_log(self, other):
        """`log(exp(a) + exp(b))` - dominated by libm rather than by memory traffic.

        The counter-example to the chain results: when one expensive kernel decides the outcome,
        the intermediates a fused pipeline saves stop mattering.
        """
        raise NotImplementedError()

    def stored_bytes(self):
        raise NotImplementedError()


class NumpyArray(AbstractArray):
    name = "numpy"

    @classmethod
    def from_numpy(cls, data, *, block_shape=None, read_size=None):
        return cls(np.ascontiguousarray(data))

    def read(self, index):
        return self.raw[index].copy()

    def negate(self):
        return -self.raw

    def add(self, other):
        return self.raw + other.raw

    def reduce(self, op, axis):
        return np.asarray(getattr(self.raw, op)(axis=axis))

    def _materialize(self, value):
        return value

    def exp_log(self, other):
        return np.log(np.exp(self.raw) + np.exp(other.raw))

    def stored_bytes(self):
        return int(self.raw.nbytes)


class JixArray(AbstractArray):
    """Block-compressed jix storage with the byte-shuffle filter."""

    name = "jix"
    filters: ClassVar[list] = ["byte-shuffle"]

    @classmethod
    def from_numpy(cls, data, *, block_shape=None, read_size=None):
        params = {
            "block_shape": block_shape,
            "codec": "zstd",
            "compression_level": ZSTD_LEVEL,
            "filters": cls.filters,
        }
        if read_size is not None:
            params["read_size"] = read_size
        return cls(jix.compact(np.ascontiguousarray(data), params=params))

    def read(self, index):
        return self.raw.numpy(index)

    def negate(self):
        return np.asarray((-self.raw).numpy())

    def add(self, other):
        return np.asarray((self.raw + other.raw).numpy())

    def reduce(self, op, axis):
        return np.asarray(getattr(self.raw, op)(axis=axis).numpy())

    def _materialize(self, value):
        return np.asarray(value.numpy())

    def exp_log(self, other):
        return np.asarray((self.raw.exp() + other.raw.exp()).log().numpy())

    def stored_bytes(self):
        buf = io.BytesIO()
        self.raw.write_to(buf)
        return buf.getbuffer().nbytes


class JixNoShuffleArray(JixArray):
    """jix with no filter pipeline - only meaningful next to the shuffled build."""

    name = "jix-noshuffle"
    filters: ClassVar[list] = []


class JixPlainArray(JixArray):
    """jix over an ordinary uncompressed buffer.

    Isolates the cost of jix's operation machinery from the cost of decompression: whatever this
    costs above numpy is overhead that compressed storage would also pay.
    """

    name = "jix-plain"

    @classmethod
    def from_numpy(cls, data, *, block_shape=None, read_size=None):
        params = None if read_size is None else {"read_size": read_size}
        return cls(jix.asarray(np.ascontiguousarray(data), params=params))

    def stored_bytes(self):
        return int(np.prod(self.raw.shape)) * self.raw.dtype.itemsize


class Blosc2Array(AbstractArray):
    """Blosc2 with the shuffle filter and its own choice of chunk shape."""

    name = "blosc2"
    filters: ClassVar[list] = [blosc2.Filter.SHUFFLE]
    match_chunks_to_blocks = False

    @classmethod
    def from_numpy(cls, data, *, block_shape=None, read_size=None):
        return cls(
            blosc2.asarray(
                np.ascontiguousarray(data),
                blocks=block_shape,
                chunks=block_shape if cls.match_chunks_to_blocks else None,
                cparams=blosc2.CParams(
                    codec=blosc2.Codec.ZSTD,
                    clevel=ZSTD_LEVEL,
                    filters=cls.filters,
                    nthreads=NTHREADS,
                ),
                dparams=blosc2.DParams(nthreads=NTHREADS),
            )
        )

    def read(self, index):
        return np.asarray(self.raw[index])

    def negate(self):
        return np.asarray((-self.raw)[:])

    def add(self, other):
        return np.asarray((self.raw + other.raw)[:])

    def reduce(self, op, axis):
        return np.asarray(getattr(blosc2, op)(self.raw, axis=axis))

    def _materialize(self, value):
        return np.asarray(value[:])

    def exp_log(self, other):
        return np.asarray(blosc2.log(blosc2.exp(self.raw) + blosc2.exp(other.raw))[:])

    def stored_bytes(self):
        return int(self.raw.schunk.cbytes)


class Blosc2NoShuffleArray(Blosc2Array):
    name = "blosc2-noshuffle"
    filters: ClassVar[list] = []


class Blosc2ChunkedArray(Blosc2Array):
    """Blosc2 with `chunks` pinned equal to `blocks`.

    Left to itself Blosc2 picks chunks thousands of times larger than the block. The block is still
    the decompression unit, but walking a large chunk's block table costs real time on every read -
    and a requested block shape is silently adjusted unless `chunks` is pinned too. This arm shows
    Blosc2 at its best rather than at its default.
    """

    name = "blosc2-chunked"
    match_chunks_to_blocks = True


class ZarrArray(AbstractArray):
    name = "zarr"

    @classmethod
    def from_numpy(cls, data, *, block_shape=None, read_size=None):
        return cls(
            zarr.create_array(
                store=zarr.storage.MemoryStore(),
                data=np.ascontiguousarray(data),
                chunks="auto" if block_shape is None else block_shape,
                compressors=[BloscCodec(cname="zstd", clevel=ZSTD_LEVEL, shuffle=BloscShuffle.shuffle)],
            )
        )

    def read(self, index):
        return np.asarray(self.raw[index])

    # Zarr has no compute layer: every operation decompresses to NumPy first. That is the honest
    # comparison - it is what a caller would have to write.
    def _dense(self):
        return np.asarray(self.raw[:])

    def negate(self):
        return -self._dense()

    def add(self, other):
        return self._dense() + other._dense()

    def reduce(self, op, axis):
        return np.asarray(getattr(self._dense(), op)(axis=axis))

    def _operand(self):
        return self._dense()

    def _materialize(self, value):
        return value

    def exp_log(self, other):
        return np.log(np.exp(self._dense()) + np.exp(other._dense()))

    def stored_bytes(self):
        return int(self.raw.nbytes_stored())


ARRAY_IMPLS = {
    cls.name: cls
    for cls in (
        NumpyArray,
        JixPlainArray,
        JixArray,
        JixNoShuffleArray,
        Blosc2Array,
        Blosc2NoShuffleArray,
        Blosc2ChunkedArray,
        ZarrArray,
    )
}
