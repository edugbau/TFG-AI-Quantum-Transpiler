from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator


@contextlib.contextmanager
def suppress_output(*, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
