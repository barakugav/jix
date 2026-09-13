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
CODEC_DESC = f"zstd level {ZSTD_LEVEL}, byte-shuffle, {NTHREADS} thread (jix, blosc2, zarr matched)"

blosc2.set_nthreads(NTHREADS)
blosc2.nthreads = NTHREADS
# blosc2 evaluates lazy expressions through numexpr, which keeps its own thread pool (one per core
# by default). Pinning only blosc2's threads would leave the elementwise arms multi-threaded while
# jix and numpy run on one core.
numexpr.set_num_threads(NTHREADS)

# The cheap elementwise chain, as (operator, constant) steps. Alternating multiply and add keeps
# each step memory-bound rather than arithmetic-bound, and stops a compiler folding the chain into
# one operation. Every library runs exactly these steps.
#
# The last step subtracts rather than adding a negative, because blosc2 gets
# `(nested expr) + -3.0` wrong - it builds an expression string and mis-evaluates the negative
# literal once the left side is itself an expression. `(nested expr) - 3.0` is correct, and so is
# `bare array + -3.0`; only the nested form is broken.
CHAIN_STEPS = [("mul", 2.0), ("add", 1.0), ("mul", 0.5), ("sub", 3.0)]

APPLY = {
    "mul": lambda value, constant: value * constant,
    "add": lambda value, constant: value + constant,
    "sub": lambda value, constant: value - constant,
}


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
    def from_numpy(cls, data, *, block_shape=None):
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

    def chain(self, count):
        """`count` steps of the canonical chain.

        Shared rather than reimplemented per library on purpose: the arms have to run identical
        arithmetic, and four copies of this loop is four chances for them to stop doing so.
        """
        out = self._operand()
        for step, constant in chain_steps(count):
            out = APPLY[step](out, constant)
        return self._materialize(out)

    def exp_log(self):
        """The transcendental chain `log(exp(a) * 0.5 + 1)` - dominated by libm, not by memory."""
        raise NotImplementedError()

    def stored_bytes(self):
        raise NotImplementedError()


class NumpyArray(AbstractArray):
    name = "numpy"

    @classmethod
    def from_numpy(cls, data, *, block_shape=None):
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

    def exp_log(self):
        return np.log(np.exp(self.raw) * 0.5 + 1.0)

    def stored_bytes(self):
        return int(self.raw.nbytes)


class JixArray(AbstractArray):
    """Block-compressed jix storage with the byte-shuffle filter."""

    name = "jix"
    filters: ClassVar[list] = ["byte-shuffle"]

    @classmethod
    def from_numpy(cls, data, *, block_shape=None):
        return cls(
            jix.compact(
                np.ascontiguousarray(data),
                params={
                    "block_shape": block_shape,
                    "codec": "zstd",
                    "compression_level": ZSTD_LEVEL,
                    "filters": cls.filters,
                },
            )
        )

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

    def exp_log(self):
        return np.asarray((self.raw.exp() * 0.5 + 1.0).log().numpy())

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
    def from_numpy(cls, data, *, block_shape=None):
        return cls(jix.asarray(np.ascontiguousarray(data)))

    def stored_bytes(self):
        return int(np.prod(self.raw.shape)) * self.raw.dtype.itemsize


class Blosc2Array(AbstractArray):
    """Blosc2 with the shuffle filter and its own choice of chunk shape."""

    name = "blosc2"
    filters: ClassVar[list] = [blosc2.Filter.SHUFFLE]
    match_chunks_to_blocks = False

    @classmethod
    def from_numpy(cls, data, *, block_shape=None):
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

    def exp_log(self):
        return np.asarray(blosc2.log(blosc2.exp(self.raw) * 0.5 + 1.0)[:])

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
    def from_numpy(cls, data, *, block_shape=None):
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

    def exp_log(self):
        dense = self._dense()
        return np.log(np.exp(dense) * 0.5 + 1.0)

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
