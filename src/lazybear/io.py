from __future__ import annotations

from typing import Any, Iterator

import polars as pl


class IOMixin:
    def to_arrow(self, limit: int | None = None):
        """Execute and return a pyarrow.Table"""
        try:
            import pyarrow as pa  # type: ignore
        except Exception as exc:
            raise RuntimeError('to_arrow requires optional dependency pyarrow; install it to use this method') from exc
        sel = self.to_select()
        if limit is not None:
            sel = sel.limit(limit)
        with self._engine.connect() as conn:  # type: ignore[attr-defined]
            res = conn.execute(sel)
            rows = res.fetchall()
            names = list(res.keys())
        # build arrow table from column-wise lists to preserve types where possible
        columns: dict[str, list] = {n: [] for n in names}
        for row in rows:
            for n, v in zip(names, row):
                columns[n].append(v)
        return pa.table(columns)

    def write_parquet(self, file, chunk_size: int | bool | None = None, start_index: int = 0, **kwargs):
        """Write results to Parquet.

        By default materializes the entire result and writes a single Parquet file.
        If ``chunk_size`` is provided, the query is streamed in chunks and written to
        multiple Parquet part files sharing a common prefix.
        """
        from pathlib import Path

        if chunk_size is None:
            df = self.collect()  # type: ignore[attr-defined]
            df.write_parquet(file, **kwargs)
            return

        if chunk_size is True:
            chunk_size = 10_000
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError('chunk_size must be a positive integer when provided')
        if not isinstance(start_index, int) or start_index < 0:
            raise ValueError('start_index must be a non-negative integer')

        outpath = Path(file)
        outpath.parent.mkdir(exist_ok=True)
        file_prefix = str(outpath.with_suffix('').stem)

        idx = start_index
        for chunk in self.collect_batches(chunk_size=chunk_size):  # type: ignore[attr-defined]
            outfile = f'{file_prefix}-{idx:05d}.parquet'
            chunk.write_parquet(outpath.parent / outfile, **kwargs)
            idx += 1

    def write_csv(self, file, chunk_size: int | None | bool = None, **kwargs) -> None:
        """Write results to a csv file using polars.

        By default materializes then writes a single csv. If ``chunk_size`` is provided,
        the query is streamed and appended to the same csv file, writing the header only
        once for the first chunk.
        """
        from pathlib import Path

        # normalize alias kwargs
        if 'has_header' in kwargs and 'include_header' not in kwargs:
            kwargs['include_header'] = kwargs.pop('has_header')
        if 'sep' in kwargs and 'separator' not in kwargs:
            kwargs['separator'] = kwargs.pop('sep')

        if chunk_size is None:
            df = self.collect()  # type: ignore[attr-defined]
            df.write_csv(file, **kwargs)
            return

        if chunk_size is True:
            chunk_size = 10_000
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError('chunk_size must be a positive integer when provided')

        path = Path(file)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        first = True
        with open(path, 'w', newline='', encoding=kwargs.pop('encoding', 'utf8')) as out:
            for chunk in self.collect_batches(chunk_size=chunk_size):  # type: ignore[attr-defined]
                if first:
                    include_header = kwargs.get('include_header', True)
                    chunk.write_csv(out, include_header=include_header,
                                    **{k: v for k, v in kwargs.items() if k != 'include_header'})
                    first = False
                else:
                    local_kwargs = {k: v for k, v in kwargs.items() if k != 'include_header'}
                    local_kwargs['include_header'] = False
                    chunk.write_csv(out, **local_kwargs)
