from __future__ import annotations

from typing import TYPE_CHECKING

from baguetter.indices import TextPreprocessor

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable


class MockTextPreprocessor(TextPreprocessor):
    def process(self, item: str) -> list:
        return item.split()

    def process_many(
        self,
        items: Iterable[str],
        *,
        n_workers: int = 0,
        chunksize: int = 128,
        show_progress: bool = False,
        return_generator: bool = True,
    ) -> Generator[list[str], None, None]:
        def generate():
            yield from (self.process(item) for item in items)

        return generate() if return_generator else list(generate())
