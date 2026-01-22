from __future__ import annotations

from typing import Any, Mapping, Sequence, Iterator

import sqlalchemy as sa
from sqlalchemy.engine import Engine

import polars as pl

from .io import IOMixin
from .expressions import Expr, AliasedExpr, _to_sa
from .engine import _inline_for_select, _normalize_predicate, _same_server


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
    ):
        self._engine = engine
        self._selectable = selectable  # a FromClause/Subquery providing column namespace
        self._columns = dict(columns)  # name -> ColumnElement bound to selectable
        # store ordering as (column_name, descending?)
        self._order_keys: list[tuple[str, bool]] = list(order_keys) if order_keys else []
        self._limit = limit

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
        return LazyBearFrame(self._engine, sq, cols, order_keys=self._order_keys, limit=self._limit)

    def filter(self, predicate: Any) -> 'LazyBearFrame':
        """Emulate polars filter, but must use `col` rather than `pl.col`"""
        where_expr = _normalize_predicate(_to_sa(predicate, self))
        # build a select that preserves current columns without introducing an extra nesting level
        base_cols = [self._selectable.c[name] for name in self.columns]
        sel = sa.select(*base_cols).select_from(self._selectable).where(where_expr)
        # rebuild column namespace from the subquery
        sq = sel.subquery()
        cols = {c.key: sq.c[c.key] for c in sq.c}
        return LazyBearFrame(self._engine, sq, cols, order_keys=self._order_keys, limit=self._limit)

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
        return LazyBearFrame(self._engine, sq, cols, order_keys=self._order_keys, limit=self._limit)

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
        out = LazyBearFrame(self._engine, self._selectable, self._columns, order_keys=order_keys, limit=self._limit)
        return out

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
        out = LazyBearFrame(self._engine, self._selectable, self._columns, order_keys=order_keys, limit=self._limit)
        return out

    def limit(self, n: int) -> 'LazyBearFrame':
        if not isinstance(n, int) or n < 0:
            raise ValueError('limit must be a non-negative integer')
        # materialize ordering + limit into a subquery so subsequent filters apply after the limit
        base_cols = [self._selectable.c[name] for name in self.columns]
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
        return LazyBearFrame(self._engine, sq, cols)

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

        left_cols = {name: self._resolve_column(name) for name in self.columns}
        right_cols = {name: other._resolve_column(name) for name in other.columns}

        on_clauses = [left_cols[lk] == right_cols[rk] for lk, rk in zip(left_keys, right_keys)]
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
            select_list.append(self._resolve_column(name).label(name))
        # select right columns, suffix overlaps
        for name in other.columns:
            lbl = name if name not in overlap else f'{name}{right_suffix}'
            select_list.append(other._resolve_column(name).label(lbl))

        sel = sa.select(*select_list).select_from(j)
        sq = sel.subquery()
        cols = {c.key: sq.c[c.key] for c in sq.c}
        return LazyBearFrame(self._engine, sq, cols)

    def group_by(self, *keys: str) -> 'GroupedLazyBearFrame':
        for k in keys:
            if k not in self._columns:
                raise KeyError(f"Group key {k!r} not found in columns {sorted(self._columns)}")
        return GroupedLazyBearFrame(self, list(keys))

    def to_select(self) -> sa.Select:
        """Select all columns from current selectable"""
        sel = sa.select(*[self._selectable.c[name] for name in self.columns]).select_from(self._selectable)
        if self._order_keys:
            order_exprs = []
            for name, desc in self._order_keys:
                col = self._selectable.c[name]
                order_exprs.append(col.desc() if desc else col.asc())
            sel = sel.order_by(*order_exprs)
        if self._limit is not None:
            sel = sel.limit(self._limit)
        return sel

    def collect(self, limit: int | None = None, *, infer_schema_length=200) -> pl.DataFrame:
        """Execute and return a polars DataFrame.

        If `limit` is provided, apply a limit/top where supported by the backend (sqlalchemy will translate limit).
        """
        sel = self.to_select()
        if limit is not None:
            sel = sel.limit(limit)
        with self._engine.connect() as conn:
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
        sel = self.to_select()
        with self._engine.connect() as conn:
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
        return LazyBearFrame(self._lf._engine, sq, cols)
