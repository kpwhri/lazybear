from __future__ import annotations

from pathlib import Path


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
        """Write results to parquet.

        By default materializes the entire result and writes a single parquet file.
        If ``chunk_size`` is provided, the query is streamed in chunks and written to
        multiple parquet part files sharing a common prefix.

        Args:
            file: Filepath where parquet file will be written. Becomes a prefix for multiple files if
                chunk_size is specified.
            chunk_size: Number of rows/items to include in each batch. If ``True``,
                defaults to ``10_000``. Must be a positive integer.
            start_index: Initial numeric index used in generated file names. Must be
                a non-negative integer.
            **kwargs: Additional keyword arguments passed to each batch's
                ``write_parquet()`` method.
        """
        if chunk_size is None or chunk_size is False:
            df = self.collect()  # type: ignore[attr-defined]
            df.write_parquet(file, **kwargs)
            return
        # else, create a directory
        file = Path(file)
        self.write_parquet_dir(file.parent, chunk_size=chunk_size, start_index=start_index, prefix=file.stem, **kwargs)

    def write_parquet_dir(self, outdir: Path, chunk_size: int | bool | None = 10_000, start_index: int = 0,
                          prefix='part', **kwargs):
        """Write batched data to a directory as parquet files.

        Writes batches returned by ``collect_batches()`` into ``outdir`` as
        individual parquet files. File names use the provided prefix followed by a
        zero-padded numeric index, for example ``part-000000.parquet``.

        Args:
            outdir: Directory where parquet files will be written. The directory is
                created if it does not already exist.
            chunk_size: Number of rows/items to include in each batch. If ``True``,
                defaults to ``10_000``. Must be a positive integer.
            start_index: Initial numeric index used in generated file names. Must be
                a non-negative integer.
            prefix: Prefix used for generated parquet file names.
            **kwargs: Additional keyword arguments passed to each batch's
                ``write_parquet()`` method.

        Raises:
            ValueError: If ``chunk_size`` is not a positive integer, or if
                ``start_index`` is not a non-negative integer.

        Example:
            Write batches to a directory using the default file name prefix:

            ```python
            writer.write_parquet_dir(Path("output/parquet"), chunk_size=10_000)
            ```
        """
        if chunk_size is True:
            chunk_size = 10_000
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError('chunk_size must be a positive integer when provided')
        if not isinstance(start_index, int) or start_index < 0:
            raise ValueError('start_index must be a non-negative integer')

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        idx = start_index
        for chunk in self.collect_batches(chunk_size=chunk_size):  # type: ignore[attr-defined]
            chunk.write_parquet(outdir / f'{prefix}-{idx:06d}.parquet', **kwargs)
            idx += 1

    def write_csv(self, file, chunk_size: int | None | bool = None, **kwargs) -> None:
        """Write results to a csv file using polars.

        By default materializes then writes a single csv. If ``chunk_size`` is provided,
        the query is streamed and appended to the same csv file, writing the header only
        once for the first chunk.
        """
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
