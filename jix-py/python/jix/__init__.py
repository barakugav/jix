from .jix import *  # noqa: F403
from .jix import __version__ as __version__
from .jix import absolute, clamp, concatenate, divide, multiply, negative, power, product, subtract

__doc__ = jix.__doc__  # noqa: F405
if hasattr(jix, "__all__"):  # noqa: F405
    __all__ = jix.__all__  # noqa: F405

pow = power
"""Alias for [`jix.power`][jix.power]"""

abs = absolute
"""Alias for [`jix.absolute`][jix.absolute]"""

concat = concatenate
"""Alias for [`jix.concatenate`][jix.concatenate]"""

clip = clamp
"""Alias for [`jix.clamp`][jix.clamp]"""

prod = product
"""Alias for [`jix.product`][jix.product]"""

sub = subtract
"""Alias for [`jix.subtract`][jix.subtract]"""

mul = multiply
"""Alias for [`jix.multiply`][jix.multiply]"""

div = divide
"""Alias for [`jix.divide`][jix.divide]"""

neg = negative
"""Alias for [`jix.negative`][jix.negative]"""
