import blosc2
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

import jix

ZSTD_LEVEL = 3
NTHREADS = 1
CODEC_DESC = f"zstd level {ZSTD_LEVEL}, byte-shuffle, {NTHREADS} thread (jix, blosc2, zarr matched)"

blosc2.set_nthreads(NTHREADS)
blosc2.nthreads = NTHREADS


class AbstractArray:
    """Base implementation. Subclasses set `name` and implement the interface."""

    name = "abstract"

    def __init__(self, raw):
        self.raw = raw

    @classmethod
    def from_numpy(cls, data, *, block_shape=None):
        raise NotImplementedError()

    def read(self, index):
        raise NotImplementedError()


class JixArray(AbstractArray):
    name = "jix"

    @classmethod
    def from_numpy(cls, data, *, block_shape=None):
        arr = jix.compact(
            data,
            params={
                "block_shape": block_shape,
                "codec": "zstd",
                "compression_level": ZSTD_LEVEL,
                "filters": [],  # no shuffle
            },
        )
        return cls(arr)

    def read(self, index):
        return self.raw.numpy(index)


class JixShuffleArray(AbstractArray):
    name = "jix-shuffle"

    @classmethod
    def from_numpy(cls, data, *, block_shape=None):
        arr = jix.compact(
            data,
            params={
                "block_shape": block_shape,
                "codec": "zstd",
                "compression_level": ZSTD_LEVEL,
                "filters": ["byte-shuffle"],
            },
        )
        return cls(arr)

    def read(self, index):
        return self.raw.numpy(index)


class JixPlainArray(JixArray):
    """jix over a plain in-memory buffer (no compression) - isolates op-chain overhead."""

    name = "jix-plain"

    @classmethod
    def from_numpy(cls, data, *, block_shape=None):
        return cls(jix.asarray(np.ascontiguousarray(data)))


class NumpyArray(AbstractArray):
    name = "numpy"

    @classmethod
    def from_numpy(cls, data, *, block_shape=None):
        return cls(np.ascontiguousarray(data))

    def read(self, index):
        return self.raw[index].copy()


class Blosc2Array(AbstractArray):
    name = "blosc2"

    @classmethod
    def from_numpy(cls, data, *, block_shape=None):
        nd = blosc2.asarray(
            np.ascontiguousarray(data),
            blocks=block_shape,
            chunks=None,  # auto select chunk shape, only force block shape
            cparams=blosc2.CParams(
                codec=blosc2.Codec.ZSTD,
                clevel=ZSTD_LEVEL,
                filters=[],  # no shuffle
                nthreads=NTHREADS,
            ),
            dparams=blosc2.DParams(nthreads=NTHREADS),
        )
        return cls(nd)

    def read(self, index):
        return np.asarray(self.raw[index])


class Blosc2ShuffleArray(AbstractArray):
    name = "blosc2-shuffle"

    @classmethod
    def from_numpy(cls, data, *, block_shape=None):
        nd = blosc2.asarray(
            np.ascontiguousarray(data),
            blocks=block_shape,
            chunks=None,  # auto select chunk shape, only force block shape
            cparams=blosc2.CParams(
                codec=blosc2.Codec.ZSTD,
                clevel=ZSTD_LEVEL,
                filters=[blosc2.Filter.SHUFFLE],
                nthreads=NTHREADS,
            ),
            dparams=blosc2.DParams(nthreads=NTHREADS),
        )
        return cls(nd)

    def read(self, index):
        return np.asarray(self.raw[index])


class ZarrArray(AbstractArray):
    name = "zarr"

    @classmethod
    def from_numpy(cls, data, *, block_shape=None):
        z = zarr.create_array(
            store=zarr.storage.MemoryStore(),
            data=data,
            chunks="auto" if block_shape is None else block_shape,
            compressors=[BloscCodec(cname="zstd", clevel=ZSTD_LEVEL, shuffle=BloscShuffle.shuffle)],
        )
        return cls(z)

    def read(self, index):
        return np.asarray(self.raw[index])


ARRAY_IMPLS = {
    c.name: c
    for c in (
        JixArray,
        JixShuffleArray,
        JixPlainArray,
        NumpyArray,
        Blosc2Array,
        Blosc2ShuffleArray,
        ZarrArray,
    )
}
