from __future__ import annotations

from typing import Any, Mapping, Sequence, Iterator

import warnings
import uuid
import sqlalchemy as sa
from sqlalchemy.engine import Engine

import polars as pl

from lazybear.db.insert.dispatch import bulk_insert_fast
from lazybear.io import IOMixin
from lazybear.expressions import Expr, AliasedExpr
from lazybear.engine import _inline_for_select, _normalize_predicate, _same_server


def _to_sa(x: Any, lf: 'LazyBearFrame') -> sa.ColumnElement[Any]:
    # avoid circular import by local import
    from .expressions import _to_sa as _real_to_sa
    return _real_to_sa(x, lf)


class LazyBearFrame(IOMixin):
    """A lazy sql builder with polars-like syntax backed by sqlalchemy.

    - Immutable: every transform returns a new LazyFrame.
    - Backed by a single sqlalchemy engine (server). Joins require same server.

    Construction
    ------------
    Use `scan_table(table_name, engine, schema=None)` to create a LazyBearFrame from a database table.
    """

    def __init__(
            self,
            engine: Engine,
            selectable: sa.sql.Selectable,
            columns: Mapping[str, sa.ColumnElement[Any]],
            order_keys: list[tuple[str, bool]] | None = None,
            limit: int | None = None,
            _upstream: Sequence[LazyBearFrame] | None = None,
    ):
        self._engine = engine
        self._selectable = selectable  # a FromClause/Subquery providing column namespace
        self._columns = dict(columns)  # name -> ColumnElement bound to selectable
        # store ordering as (column_name, descending?)
        self._order_keys: list[tuple[str, bool]] = list(order_keys) if order_keys else []
        self._limit = limit
        self._upstream = list(_upstream) if _upstream else []

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def columns(self) -> list[str]:
        return list(self._columns.keys())

    def _resolve_column(self, name: str) -> sa.ColumnElement[Any]:
        if name not in self._columns:
            raise KeyError(f"Column {name!r} not found. Available: {sorted(self._columns)}")
        return self._columns[name]

    def select(self, *items: Any) -> 'LazyBearFrame':
        """Project columns/expressions.

        Parameters:
            - `items` accepts:
                - column names: `select('a', 'b')`
                - tuples: `(alias, Expr|value)` like `('y2', col('y') * 2)`
                - aliased expressions: `col('y').alias('y2')`
        """
        if not items:
            return self
        sa_cols: list[sa.ColumnElement[Any]] = []
        if len(items) == 1 and isinstance(items[0], list):
            items = items[0]
        for it in items:
            if isinstance(it, str):
                expr = self._resolve_column(it).label(it)
                sa_cols.append(expr)
            elif isinstance(it, AliasedExpr):
                expr = _inline_for_select(_to_sa(it, self)).label(it._alias)
                sa_cols.append(expr)
            elif isinstance(it, Expr):
                # anonymous expression: attempt to derive name
                expr = _inline_for_select(_to_sa(it, self))
                sa_cols.append(expr)
            elif isinstance(it, tuple) and len(it) == 2 and isinstance(it[0], str):
                alias, value = it
                expr = _inline_for_select(_to_sa(value, self)).label(alias)
                sa_cols.append(expr)
            else:
                raise TypeError('select() items must be str, Expr, AliasedExpr, or (alias, value)')
        sel = sa.select(*sa_cols).select_from(self._selectable)
        sq = sel.subquery()
        cols = {c.key: sq.c[c.key] for c in sq.c}
        return self._rebuild(sq, cols)

    def _rebuild(self, selectable: sa.sql.Selectable, columns: Mapping[str, sa.ColumnElement[Any]]) -> 'LazyBearFrame':
        return LazyBearFrame(self._engine, selectable, columns, self._order_keys, self._limit, [self])

    def filter(self, predicate: Any) -> 'LazyBearFrame':
        """Emulate polars filter, but must use `col` rather than `pl.col`"""
        where_expr = _normalize_predicate(_to_sa(predicate, self))
        # build a select that preserves current columns without introducing an extra nesting level
        base_cols = [self._selectable.c[name].label(name) for name in self.columns]
        sel = sa.select(*base_cols).select_from(self._selectable).where(where_expr)
        # rebuild column namespace from the subquery
        sq = sel.subquery()
        cols = {c.key: sq.c[c.key] for c in sq.c}
        return self._rebuild(sq, cols)

    def with_columns(self, *exprs: Any, **named: Any) -> 'LazyBearFrame':
        """Add or replace columns using expressions, keeping existing columns.

        Examples:
            lf.with_columns(('y2', col('y') * 2), total=col('a') + col('b'))
            lf.with_columns(col('y') * 2).select('...', ('y2', col('y') * 2))  # positional expr must be AliasedExpr or tuple
        """
        select_list: list[sa.ColumnElement[Any]] = [self._selectable.c[name] for name in self.columns]

        def _labelled(x: Any) -> sa.ColumnElement[Any]:
            if isinstance(x, AliasedExpr):
                return _to_sa(x, self).label(x._alias)
            if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str):
                alias, value = x
                return _to_sa(value, self).label(alias)
            raise TypeError('positional expressions must be AliasedExpr or (name, expr) tuples')

        for e in exprs:
            select_list.append(_labelled(e))
        for name, value in named.items():
            select_list.append(_to_sa(value, self).label(name))

        sel = sa.select(*select_list).select_from(self._selectable)
        sq = sel.subquery()
        cols = {c.key: sq.c[c.key] for c in sq.c}
        return self._rebuild(sq, cols)

    def order_by(self, *keys: Any) -> 'LazyBearFrame':
        if len(keys) == 1 and isinstance(keys[0], (list, tuple)):
            keys = tuple(keys[0])
        order_keys: list[tuple[str, bool]] = []
        for k in keys:
            if isinstance(k, str):
                descending = False
                name = k
                if k.startswith('-'):
                    descending = True
                    name = k[1:]
                elif k.startswith('+'):
                    name = k[1:]
                # validate column exists now to catch typos early
                _ = self._resolve_column(name)
                order_keys.append((name, descending))
            elif isinstance(k, Expr):
                raise TypeError('order_by() only supports column names (str) in this API')
            else:
                raise TypeError('order_by() keys must be str or Expr')
        new_lf = self._rebuild(self._selectable, self._columns)
        new_lf._order_keys = order_keys
        return new_lf

    def sort(self, by: Any, *more_by: Any, descending: bool | Sequence[bool] = False) -> 'LazyBearFrame':
        """order_by with polars-style API"""
        if isinstance(by, (list, tuple)):
            keys = by
        else:
            keys = [by]

        for more in more_by:
            keys.append(more)

        def _flatten_keys(x: Any) -> list[Any]:
            if isinstance(x, (list, tuple)):
                return list(x)
            return [x]

        flat = []
        for k in keys:
            flat.extend(_flatten_keys(k))
        if isinstance(descending, bool):
            desc_flags = [descending] * len(flat)
        else:
            desc_flags = list(descending)
            if len(desc_flags) != len(flat):
                raise ValueError('descending flags length must match number of sort keys')
        order_keys: list[tuple[str, bool]] = []
        for k, desc in zip(flat, desc_flags):
            if isinstance(k, str):
                _ = self._resolve_column(k)
                order_keys.append((k, bool(desc)))
            elif isinstance(k, Expr):
                raise TypeError('sort keys must be column names (str) in this API')
            else:
                raise TypeError('sort keys must be str or Expr')
        new_lf = self._rebuild(self._selectable, self._columns)
        new_lf._order_keys = order_keys
        return new_lf

    def limit(self, n: int) -> 'LazyBearFrame':
        if not isinstance(n, int) or n < 0:
            raise ValueError('limit must be a non-negative integer')
        # materialize ordering + limit into a subquery so subsequent filters apply after the limit
        base_cols = [self._selectable.c[name].label(name) for name in self.columns]
        sel = sa.select(*base_cols).select_from(self._selectable)
        if self._order_keys:
            order_exprs = []
            for name, desc in self._order_keys:
                col = self._selectable.c[name]
                order_exprs.append(col.desc() if desc else col.asc())
            sel = sel.order_by(*order_exprs)
        sel = sel.limit(n)
        sq = sel.subquery()
        cols = {c.key: sq.c[c.key] for c in sq.c}
        # after making limit into a subquery, clear order/limit state to avoid reapplying outside
        return self._rebuild(sq, cols)

    def join(
            self,
            other: 'LazyBearFrame',
            on: Any | None = None,
            *,
            left_on: str | Sequence[str] | None = None,
            right_on: str | Sequence[str] | None = None,
            how: str = 'inner',
            suffixes: tuple[str, str] = ('_x', '_y'),
    ) -> 'LazyBearFrame':
        if not isinstance(other, LazyBearFrame):
            raise TypeError('other must be a LazyBearFrame')
        if not _same_server(self._engine, other._engine):
            raise ValueError('cannot join frames from different servers')

        def _as_list(x: str | Sequence[str]) -> list[str]:
            return [x] if isinstance(x, str) else list(x)

        if on is not None and (left_on is not None or right_on is not None):
            raise ValueError('specify either on= or left_on=/right_on=, not both')
        if on is not None:
            left_keys = right_keys = _as_list(on)
        else:
            if left_on is None or right_on is None:
                raise ValueError('must provide both left_on and right_on when on is not used')
            left_keys, right_keys = _as_list(left_on), _as_list(right_on)
        if len(left_keys) != len(right_keys):
            raise ValueError('left_on and right_on must have the same number of keys')

        join_type = how.lower()
        if join_type not in {'inner', 'left', 'right', 'full'}:
            raise ValueError("how must be one of 'inner', 'left', 'right', 'full'")

        on_clauses = [self._selectable.c[lk] == other._selectable.c[rk] for lk, rk in zip(left_keys, right_keys)]
        on_expr = sa.and_(*on_clauses)

        if join_type == 'inner':
            j = sa.join(self._selectable, other._selectable, on_expr, isouter=False)
        elif join_type == 'left':
            j = sa.join(self._selectable, other._selectable, on_expr, isouter=True)
        elif join_type == 'right':
            j = sa.join(other._selectable, self._selectable, on_expr, isouter=True)
        else:  # full
            j = sa.outerjoin(self._selectable, other._selectable, on_expr, full=True)

        # build select list with suffix handling for overlapping names
        left_names = set(self.columns)
        overlap = left_names & set(other.columns)
        right_suffix = suffixes[1]
        select_list: list[sa.ColumnElement[Any]] = []
        # select left columns with original names
        for name in self.columns:
            select_list.append(self._selectable.c[name].label(name))
        # select right columns, suffix overlaps
        for name in other.columns:
            lbl = name if name not in overlap else f'{name}{right_suffix}'
            select_list.append(other._selectable.c[name].label(lbl))

        sel = sa.select(*select_list).select_from(j)
        sq = sel.subquery()
        cols = {c.key: sq.c[c.key] for c in sq.c}
        out = self._rebuild(sq, cols)
        out._upstream = [self, other]
        return out

    def group_by(self, *keys: str) -> 'GroupedLazyBearFrame':
        for k in keys:
            if k not in self._columns:
                raise KeyError(f"Group key {k!r} not found in columns {sorted(self._columns)}")
        return GroupedLazyBearFrame(self, list(keys))

    def to_select(self) -> sa.Select:
        """Select all columns from current selectable"""
        sel = sa.select(*[self._selectable.c[name].label(name) for name in self.columns]).select_from(self._selectable)
        if self._order_keys:
            order_exprs = []
            for name, desc in self._order_keys:
                col = self._selectable.c[name]
                order_exprs.append(col.desc() if desc else col.asc())
            sel = sel.order_by(*order_exprs)
        if self._limit is not None:
            sel = sel.limit(self._limit)
        return sel

    def _find_temp_frames(self) -> list[TempLazyBearFrame]:
        """Traverse the upstream chain to find all TempLazyBearFrames that need to be materialized."""
        temps: list[TempLazyBearFrame] = []
        if isinstance(self, TempLazyBearFrame):
            temps.append(self)
        for up in self._upstream:
            temps.extend(up._find_temp_frames())
        # de-duplicate by table name to avoid redundant inserts if joined/re-used
        seen = set()
        unique_temps = []
        for t in temps:
            if t._table_name not in seen:
                unique_temps.append(t)
                seen.add(t._table_name)
        return unique_temps

    def collect(self, limit: int | None = None, *, infer_schema_length=200) -> pl.DataFrame:
        """Execute and return a polars DataFrame.

        If `limit` is provided, apply a limit/top where supported by the backend (sqlalchemy will translate limit).
        """
        temp_frames = self._find_temp_frames()

        sel = self.to_select()
        if limit is not None:
            sel = sel.limit(limit)

        with self._engine.connect() as conn:
            if temp_frames:
                with conn.begin():
                    for tf in temp_frames:
                        tf._create_and_insert(conn)
                    try:
                        res = conn.execute(sel)
                        rows = res.fetchall()
                        names = res.keys()
                    finally:
                        for tf in temp_frames:
                            tf._cleanup(conn)
            else:
                res = conn.execute(sel)
                rows = res.fetchall()
                names = res.keys()

        return pl.DataFrame(rows, schema=list(names), infer_schema_length=infer_schema_length)

    def explain(self) -> str:
        sel = self.to_select()
        try:
            compiled = sel.compile(self._engine, compile_kwargs={'literal_binds': True})
            return str(compiled)
        except Exception:
            return str(sel)

    def collect_batches(self, chunk_size: int = 10_000) -> Iterator[pl.DataFrame]:
        """Yield polars DataFrames in chunks without loading all rows in memory."""
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError('chunk_size must be a positive integer')
        
        temp_frames = self._find_temp_frames()
        sel = self.to_select()
        
        with self._engine.connect() as conn:
            if temp_frames:
                with conn.begin():
                    for tf in temp_frames:
                        tf._create_and_insert(conn)
                    try:
                        res = conn.execution_options(stream_results=True).execute(sel)
                        names = list(res.keys())
                        while True:
                            rows = res.fetchmany(chunk_size)
                            if not rows:
                                break
                            yield pl.DataFrame(rows, schema=names)
                    finally:
                        for tf in temp_frames:
                            tf._cleanup(conn)
            else:
                res = conn.execution_options(stream_results=True).execute(sel)
                names = list(res.keys())
                while True:
                    rows = res.fetchmany(chunk_size)
                    if not rows:
                        break
                    yield pl.DataFrame(rows, schema=names)

    def iter_rows(self, *, named: bool = False, chunk_size: int = 10_000) -> Iterator[tuple[Any, ...] | dict[str, Any]]:
        """Iterate over result rows with polars-compatible semantics."""
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError('chunk_size must be a positive integer')
        for chunk in self.collect_batches(chunk_size=chunk_size):
            for row in chunk.iter_rows(named=named):
                yield row


class GroupedLazyBearFrame:
    def __init__(self, lf: LazyBearFrame, keys: list[str]):
        self._lf = lf
        self._keys = keys

    def agg(self, **aggregations: tuple[str, str] | tuple[Expr | str, str] | tuple[str] | tuple[Expr]):
        # build select list: group keys + aggregations
        select_list: list[sa.ColumnElement[Any]] = [self._lf._resolve_column(k).label(k) for k in self._keys]
        for out_name, spec in aggregations.items():
            if not isinstance(spec, tuple) or not (1 <= len(spec) <= 2):
                raise TypeError('Aggregation spec must be a tuple (expr, func?)')
            if len(spec) == 1:
                expr, func = spec[0], 'count'
            else:
                expr, func = spec
            if isinstance(expr, str):
                col = self._lf._resolve_column(expr)
            elif isinstance(expr, Expr):
                col = _to_sa(expr, self._lf)
            else:
                raise TypeError('Aggregation expression must be str or Expr')
            func_l = func.lower()
            if func_l == 'count':
                out_expr = sa.func.count(col)
            elif func_l == 'sum':
                out_expr = sa.func.sum(col)
            elif func_l == 'avg' or func_l == 'mean':
                out_expr = sa.func.avg(col)
            elif func_l == 'min':
                out_expr = sa.func.min(col)
            elif func_l == 'max':
                out_expr = sa.func.max(col)
            else:
                raise ValueError(f'Unsupported aggregation function: {func}')
            select_list.append(out_expr.label(out_name))

        sel = sa.select(*select_list).select_from(self._lf._selectable).group_by(
            *[self._lf._resolve_column(k) for k in self._keys]
        )
        sq = sel.subquery()
        cols = {c.key: sq.c[c.key] for c in sq.c}
        return self._lf._rebuild(sq, cols)


class TempLazyBearFrame(LazyBearFrame):
    """A LazyBearFrame that holds a polars DataFrame and inserts it into a temp table upon collection.

    This allows using a local polars DataFrame in lazy sql operations like joins.
    """

    def __init__(
            self,
            engine: Engine,
            df: pl.DataFrame,
            table_name: str | None = None,
            selectable: sa.sql.Selectable | None = None,
            columns: Mapping[str, sa.ColumnElement[Any]] | None = None,
            order_keys: list[tuple[str, bool]] | None = None,
            limit: int | None = None,
            _upstream: Sequence[LazyBearFrame] | None = None,
    ):
        self._df = df
        self._table_name = table_name or f'lb_temp_{uuid.uuid4().hex[:8]}'
        self._temp_table_created = False

        if columns is None:
            # create initial selectable from the df schema
            # we use sa.table and sa.column to represent the eventual temp table
            cols = {name: sa.column(name) for name in df.columns}
            selectable = sa.table(self._table_name, *[sa.column(n) for n in df.columns])
        else:
            cols = dict(columns)
            # Use the provided selectable (which could be a subquery)
            if selectable is None:
                selectable = sa.table(self._table_name, *[sa.column(n) for n in df.columns])

        super().__init__(engine, selectable, cols, order_keys, limit, _upstream)

    def _rebuild(self, selectable: sa.sql.Selectable, columns: Mapping[str, sa.ColumnElement[Any]]) -> 'LazyBearFrame':
        if isinstance(self, TempLazyBearFrame):
            return TempLazyBearFrame(
                self._engine, self._df, self._table_name, selectable, columns, self._order_keys, self._limit, [self]
            )
        return LazyBearFrame(self._engine, selectable, columns, self._order_keys, self._limit, [self])

    def _create_and_insert(self, conn: sa.Connection, *, chunk_size=10_000):
        """Create the temp table and insert the dataframe data."""
        dialect = self._engine.dialect.name
        # normalize dialect names
        if 'sqlite' in dialect:
            dialect = 'sqlite'
        elif 'oracle' in dialect:
            dialect = 'oracle'
        elif 'mssql' in dialect:
            dialect = 'mssql'
        elif 'teradata' in dialect:
            dialect = 'teradata'
        elif 'postgres' in dialect:
            dialect = 'postgresql'
        else:
            warnings.warn(f'Unsupported dialect for temp tables: {dialect}. Falling back to default behavior.')

        # create table definition
        metadata = sa.MetaData()

        # map polars types to sqlalchemy types
        type_map = {
            pl.Int64: sa.BigInteger,
            pl.Int32: sa.Integer,
            pl.Int16: sa.SmallInteger,
            pl.Int8: sa.SmallInteger,
            pl.UInt64: sa.BigInteger,
            pl.UInt32: sa.BigInteger,
            pl.Float64: sa.Float,
            pl.Float32: sa.Float,
            pl.Boolean: sa.Boolean,
            pl.Utf8: sa.String,
            pl.String: sa.String,
            pl.Date: sa.Date,
            pl.Datetime: sa.DateTime,
        }

        sa_cols = []
        for name, dtype in self._df.schema.items():
            sa_type = type_map.get(dtype, sa.String)
            sa_cols.append(sa.Column(name, sa_type))

        # dialect-specific temp table creation
        prefixes = []
        create_kwargs = {}
        if dialect == 'sqlite':
            prefixes = ['TEMPORARY']
        elif dialect == 'postgresql':
            prefixes = ['TEMPORARY']
        elif dialect == 'mssql':
            if not self._table_name.startswith('#'):
                self._table_name = f'#{self._table_name}'
        elif dialect == 'oracle':
            prefixes = ['GLOBAL TEMPORARY']
            create_kwargs['oracle_on_commit'] = 'PRESERVE ROWS'
        elif dialect == 'teradata':
            prefixes = ['VOLATILE']
            create_kwargs['teradata_on_commit'] = 'PRESERVE ROWS'

        table = sa.Table(
            self._table_name,
            metadata,
            *sa_cols,
            prefixes=prefixes,
            **create_kwargs,
        )

        create_stmt = sa.schema.CreateTable(table)
        conn.execute(create_stmt)

        # try for connection-specific fast bulk insert
        try:
            bulk_insert_fast(conn, self._df, self._table_name, self._df.columns)
        except Exception as e:
            batch = []  # stream rows from polars
            for row in self._df.iter_rows(named=True, buffer_size=chunk_size):
                batch.append(row)
                if len(batch) >= chunk_size:
                    conn.execute(table.insert(), batch)
                    batch.clear()
            if batch:
                conn.execute(table.insert(), batch)

        self._temp_table_created = True

    def _cleanup(self, conn: sa.Connection):
        """Best-effort cleanup of the temp table."""
        if self._temp_table_created:
            try:
                conn.execute(sa.text(f'DROP TABLE {self._table_name}'))
            except Exception as e:
                # best-effort
                warnings.warn(f'Failed to cleanup temp table {self._table_name}: {e}')
            finally:
                self._temp_table_created = False

    def to_select(self) -> sa.Select:
        return super().to_select()

    def collect(self, limit: int | None = None, *, infer_schema_length=200) -> pl.DataFrame:
        return super().collect(limit=limit, infer_schema_length=infer_schema_length)

    def collect_batches(self, chunk_size: int = 10_000) -> Iterator[pl.DataFrame]:
        return super().collect_batches(chunk_size=chunk_size)
