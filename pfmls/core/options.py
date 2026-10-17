"""Typed option dataclasses: field type checks and ``KEY=VALUE`` parsing for command lines (and the help formatter
of the example command lines).

The option dataclasses of the level sets (:class:`pfmls.PFMLS2DOptions`, :class:`pfmls.PFMLS3DOptions`) and of the
free-surface backend (:class:`pfmls.fluid.PFMBackendConfig`) annotate every field with one of the types in
:data:`FIELD_TYPES`. :func:`check_field_types` enforces them (a string ``"false"`` for a bool field would otherwise
read as True), and :func:`parse_options` converts ``KEY=VALUE`` strings by the same annotations.
"""

from __future__ import annotations

import argparse
import dataclasses
import numbers

import numpy as np

#: Accepted value types by field annotation (bools are accepted only by bool fields, and only bools are).
FIELD_TYPES = {"bool": (bool,), "int": (numbers.Integral,), "int | None": (numbers.Integral, None),
               "float": (numbers.Real,), "float | None": (numbers.Real, None), "str": (str,)}

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _type_ok(value, allowed: tuple) -> bool:
    if value is None:
        return None in allowed
    if isinstance(value, (bool, np.bool_)):
        return bool in allowed
    return any(isinstance(value, t) for t in allowed if t is not None and t is not bool)


def check_field_types(obj) -> None:
    """Raise ``TypeError`` if a field of the dataclass instance ``obj`` does not match its annotation.

    Fields annotated with a type outside :data:`FIELD_TYPES` (e.g. a nested dataclass) are not checked here.
    """
    for f in dataclasses.fields(obj):
        allowed = FIELD_TYPES.get(f.type)
        if allowed is None:
            continue
        v = getattr(obj, f.name)
        if not _type_ok(v, allowed):
            raise TypeError(f"option {f.name} must be {f.type}, got {v!r}")


def unknown_keys(cls, keys) -> list[str]:
    """The keys that are not fields of the dataclass ``cls`` (sorted)."""
    names = {f.name for f in dataclasses.fields(cls)}
    return sorted(set(keys) - names)


def convert_value(cls, key: str, value: str):
    """``value`` (a string) converted by the annotation of field ``key`` of ``cls``.

    Bools accept ``1/0, true/false, yes/no, on/off`` (any case); ``none`` gives None where the annotation allows
    it. Raises ``ValueError`` for an unknown key or an unconvertible value.
    """
    types = {f.name: f.type for f in dataclasses.fields(cls)}
    if key not in types:
        raise ValueError(f"{key!r} is not a {cls.__name__} field (fields: {', '.join(sorted(types))})")
    allowed = FIELD_TYPES.get(types[key])
    if allowed is None:
        raise ValueError(f"{cls.__name__}.{key} cannot be set from a string")
    value = value.strip()
    low = value.lower()
    if low == "none" and None in allowed:
        return None
    if bool in allowed:
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"option {key} needs a boolean (true/false), got {value!r}")
    if numbers.Integral in allowed:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"option {key} needs an integer, got {value!r}") from None
    if numbers.Real in allowed:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"option {key} needs a number, got {value!r}") from None
    return value


def split_pair(pair: str) -> tuple[str, str]:
    """``"KEY=VALUE"`` -> ``(KEY, VALUE)`` (``ValueError`` without ``=``)."""
    key, sep, value = pair.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"expected KEY=VALUE, got {pair!r}")
    return key.strip(), value


def parse_options(cls, pairs) -> dict:
    """Typed keyword arguments for the dataclass ``cls`` from ``"KEY=VALUE"`` strings (see :func:`convert_value`)."""
    out = {}
    for pair in pairs:
        key, value = split_pair(pair)
        out[key] = convert_value(cls, key, value)
    return out


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keeps the line breaks of the script docstrings and appends ``(default: ...)`` where a default is set (not for
    flags without a value, unset defaults or empty lists)."""

    def _get_help_string(self, action):
        text = action.help or ""
        if ("%(default)" not in text and action.option_strings and action.nargs != 0
                and action.default not in (None, argparse.SUPPRESS) and action.default != []):
            text += " (default: %(default)s)"
        return text
