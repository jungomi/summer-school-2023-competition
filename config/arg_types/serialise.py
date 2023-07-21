from typing import Type, TypeVar

from simple_parsing.helpers.serialization import encode, register_decoding_fn

T = TypeVar("T")


def serialise_arg(cls: Type[T]) -> Type[T]:
    """
    This is a very simple decorator that can be added to a class which is used for
    argument parsing to make them serialisable. These all have the same structure, where
    the constructor takes a single argument (the arg as string) and the instance can be
    represented as an arg string by converting it to a string.

    i.e. arg == str(Cls(arg))

    The serialisation technically works without this, but it gives warnings, hence this
    avoids that with a simple decorator for the class definitions.
    """
    encode.register(cls, str)
    register_decoding_fn(cls, cls)
    return cls
