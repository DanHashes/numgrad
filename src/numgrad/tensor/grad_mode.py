from __future__ import annotations

from types import TracebackType

_grad_enabled = True


def is_grad_enabled() -> bool:
    return _grad_enabled


class no_grad:
    """Context manager that disables computation-graph building for its duration."""

    def __enter__(self) -> no_grad:
        global _grad_enabled
        self._prev_state = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        global _grad_enabled
        _grad_enabled = self._prev_state
