from __future__ import annotations

from typing import Any, Sequence

import sqlalchemy as sa

from lazybear.expressions import Expr, _to_sa


def _escape_like(val: str, escape_char: str = '\\') -> str:
    """escape %, _ and the escape character itself for SQL LIKE.

    note: this is conservative and uses backslash as the escape character.
    """
    if escape_char not in ('\\',):
        # support only backslash to keep things simple and portable
        escape_char = '\\'
    s = val.replace(escape_char, escape_char + escape_char)
    s = s.replace('%', escape_char + '%').replace('_', escape_char + '_')
    return s


class _StrNS:
    def __init__(self, expr: Expr):
        self._expr = expr

    def contains(self, pattern: Any, *, literal: bool | None = None) -> Expr:
        """substring/pattern match similar to polars `.str.contains`.

        - if literal is True or None: treat pattern as a literal substring
        - if literal is False: treat pattern as a SQL LIKE pattern (not regex)
        """

        def _compile(lf: 'LazyBearFrame') -> sa.ColumnElement[Any]:
            col = self._expr.to_sa(lf)
            pat = _to_sa(pattern, lf)
            if literal is False:
                # pattern is a LIKE pattern; if it's a bind param/expr we can pass through
                return col.like(pat)
            # default literal substring: wrap with wildcards and escape
            if isinstance(pat, sa.BindParameter):
                # bound literal -> compute final pattern here to avoid dialect concat
                val = pat.value
                if not isinstance(val, str):
                    val = str(val)
                like = f"%{_escape_like(val)}%"
                return col.like(like, escape='\\')
            # otherwise rely on SQL concat if available (varchar || varchar)
            return col.like(sa.concat('%', pat, '%'))

        return Expr(_compile)

    def contains_any(self, patterns: Sequence[Any], *, literal: bool | None = None) -> Expr:
        vals = list(patterns) if patterns is not None else []
        if len(vals) == 0:
            return Expr(lambda lf: sa.false())

        def _compile(lf: 'LazyBearFrame') -> sa.ColumnElement[Any]:
            return sa.or_(*[self.contains(p, literal=literal).to_sa(lf) for p in vals])

        return Expr(_compile)

    def starts_with(self, prefix: Any, *, ascii_case_insensitive=False) -> Expr:
        def _compile(lf: 'LazyBearFrame') -> sa.ColumnElement[Any]:
            col = self._expr.to_sa(lf)
            rhs = _to_sa(prefix, lf)
            if not ascii_case_insensitive:
                if isinstance(rhs, sa.BindParameter):
                    val = rhs.value
                    if not isinstance(val, str):
                        val = str(val)
                    like = f"{_escape_like(val)}%"
                    return col.like(like, escape='\\')
                return col.like(sa.concat(rhs, '%'))
            # ascii case-insensitive: upper both sides
            func = sa.func
            if isinstance(rhs, sa.BindParameter):
                val = rhs.value
                if not isinstance(val, str):
                    val = str(val)
                like = f"{_escape_like(val.upper())}%"
                return func.upper(col).like(like, escape='\\')
            return func.upper(col).like(sa.concat(func.upper(rhs), '%'))

        return Expr(_compile)

    def startswith_any(self, prefixes: Sequence[str]) -> Expr:  # alias compat
        return self.starts_with_any(prefixes)

    def starts_with_any(self, prefixes: Sequence[Any]) -> Expr:
        vals = list(prefixes) if prefixes is not None else []
        if len(vals) == 0:
            return Expr(lambda lf: sa.false())
        exprs = [self.starts_with(v) for v in vals]

        def _compile(lf: 'LazyBearFrame') -> sa.ColumnElement[Any]:
            return sa.or_(*[e.to_sa(lf) for e in exprs])

        return Expr(_compile)


class _StartsWithProxy:
    """callable proxy so `col('x').startswith('a')` works like `.str.starts_with('a')`."""

    def __init__(self, expr: Expr):
        self._expr = expr

    def __call__(self, prefix: Any) -> Expr:
        return _StrNS(self._expr).starts_with(prefix)
