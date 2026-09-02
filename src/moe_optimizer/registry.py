"""A tiny name -> factory registry.

Every compression method, clustering rule and residual codec registers itself
here.  The point is not indirection for its own sake: the research question is a
*comparison* between methods at matched byte budgets, so every method must be
reachable through one uniform entry point and be constructible from a plain
config dict.  A method that cannot be built this way cannot appear in the Pareto
table.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if name in self._items:
                raise KeyError(f"{self.kind} {name!r} already registered")
            self._items[name] = obj  # type: ignore[assignment]
            return obj

        return deco

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"unknown {self.kind} {name!r}; available: {sorted(self._items)}"
            ) from None

    def build(self, spec: dict[str, Any]) -> Any:
        """Build from ``{"name": ..., **kwargs}``."""
        spec = dict(spec)
        name = spec.pop("name")
        return self.get(name)(**spec)

    def names(self) -> list[str]:
        return sorted(self._items)


COMPRESSORS = Registry("compressor")
CLUSTERERS = Registry("clusterer")
RESIDUALS = Registry("residual")
QUANTIZERS = Registry("quantizer")
