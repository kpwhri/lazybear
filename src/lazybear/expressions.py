from __future__ import annotations

from typing import Any

import sqlalchemy as sa

import polars as pl


class Expr:
    """An expression that can be compiled to a sqlalchemy expression in the context of a LazyFrame.

    The expression is a function: (LazyFrame -> sqlalchemy ColumnElement)
    """

    def __init__(self, func):
        self._func = func

    def to_sa(self, lf: 'LazyBearFrame') -> sa.ColumnElement[Any]:
        return self._func(lf)

    # arithmetic / comparison / boolean operators
    def _bin(self, other, op):
        return Expr(lambda lf: op(self.to_sa(lf), _to_sa(other, lf)))

    def __add__(self, other):
        return self._bin(other, sa.sql.operators.add)

    def __sub__(self, other):
        return self._bin(other, sa.sql.operators.sub)

    def __mul__(self, other):
        return self._bin(other, sa.sql.operators.mul)

    def __truediv__(self, other):
        return self._bin(other, sa.sql.operators.truediv)

    def __mod__(self, other):
        return self._bin(other, sa.sql.operators.mod)

    def __eq__(self, other):  # type: ignore[override]
        return self._bin(other, sa.sql.operators.eq)

    def __ne__(self, other):  # type: ignore[override]
        return self._bin(other, sa.sql.operators.ne)

    def __gt__(self, other):
        return self._bin(other, sa.sql.operators.gt)

    def __ge__(self, other):
        return self._bin(other, sa.sql.operators.ge)

    def __lt__(self, other):
        return self._bin(other, sa.sql.operators.lt)

    def __le__(self, other):
        return self._bin(other, sa.sql.operators.le)

    def __and__(self, other):
        return self._bin(other, sa.sql.operators.and_)

    def __or__(self, other):
        return self._bin(other, sa.sql.operators.or_)

    def is_null(self) -> 'Expr':
        return Expr(lambda lf: self.to_sa(lf).is_(None))

    def is_not_null(self) -> 'Expr':
        return Expr(lambda lf: self.to_sa(lf).is_not(None))

    def is_in(self, other: Any) -> 'Expr':
        """Return an expression that checks membership (SQL IN), similar to Polars `is_in`.

        Notes:
        - Strings are treated as scalars, not iterables.
        - Supports Polars Series, Python iterables, SQLAlchemy expressions, and LazyBearFrame subqueries.
        - Empty iterable yields SQL FALSE.
        """

        def _compile(lf: 'LazyBearFrame') -> sa.ColumnElement[Any]:
            left = self.to_sa(lf)
            candidate = other

            # polars series -> list of python values
            try:
                import polars as _pl  # local import to avoid cycle
                if isinstance(candidate, _pl.Series):
                    candidate = candidate.to_list()
            except Exception:
                pass

            # LazyBearFrame -> subquery
            try:
                from .core import LazyBearFrame as _LBF  # local import to avoid cycle
                if isinstance(candidate, _LBF):
                    subq = candidate.to_select()
                    return left.in_(subq)
            except Exception:
                pass

            # sqlalchemy selectable/element
            if isinstance(candidate, (sa.SelectBase, sa.sql.ClauseElement)):
                return left.in_(candidate)

            # treat strings as scalars
            if isinstance(candidate, (str, bytes)):
                values = [candidate]
            else:
                try:
                    values = list(candidate)  # type: ignore[arg-type]
                except TypeError:
                    values = [candidate]

            if len(values) == 0:
                return sa.false()
            return left.in_(values)

        return Expr(_compile)

    @property
    def str(self):  # noqa: A003 - mimic polars .str namespace
        from .namespaces import _StrNS

        return _StrNS(self)

    @property
    def startswith(self):  # noqa: A003 - callable proxy
        from .namespaces import _StartsWithProxy

        return _StartsWithProxy(self)

    def alias(self, name: str) -> 'AliasedExpr':
        return AliasedExpr(self, name)


class AliasedExpr(Expr):
    def __init__(self, expr: Expr, alias: str):
        super().__init__(expr._func)
        self._alias = alias


def col(name: str) -> Expr:
    """Reference a column by name. Resolved in the context of the LazyFrame."""
    return Expr(lambda lf: lf._resolve_column(name))


def lit(value: Any) -> Expr:
    return Expr(lambda lf: sa.literal(value))


def _to_sa(x: Any, lf: 'LazyBearFrame') -> sa.ColumnElement[Any]:
    """convert user input to a sqlalchemy expression

    - Expr -> compile in context
    - sqlalchemy element -> pass-through
    - other -> literal bind parameter
    - if a polars expression is passed accidentally, raise TypeError for clarity
    """
    try:
        if isinstance(x, pl.Expr):
            raise TypeError('received a Polars expression in a sql lazy context')
    except Exception:
        # if polars is unavailable or type check fails, continue with normal handling
        pass
    if isinstance(x, Expr):
        return x.to_sa(lf)
    if isinstance(x, sa.ColumnElement):
        return x
    return sa.literal(x)
