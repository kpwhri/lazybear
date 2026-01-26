from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class _EngineInfo:
    drivername: str
    host: str | None
    database: str | None

    @classmethod
    def of(cls, engine: Engine) -> '_EngineInfo':
        url = engine.url
        return cls(drivername=url.drivername, host=url.host, database=url.database)


def _same_server(a: Engine, b: Engine) -> bool:
    ia, ib = _EngineInfo.of(a), _EngineInfo.of(b)
    return ia.drivername == ib.drivername and ia.host == ib.host


def _inline_sql_literal(value: Any) -> str:
    """Render a simple Python value as an inline SQL literal string.

    This is conservative and supports common scalar types to avoid bound params in SELECT lists on backends
    like Teradata that do not permit parameters in the select-list.
    """
    import datetime as _dt

    if value is None:
        return 'NULL'
    if isinstance(value, bool):
        return 'TRUE' if value else 'FALSE'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        # represent as hex literal
        return "X'" + value.hex() + "'"
    if isinstance(value, (_dt.date,)) and not isinstance(value, _dt.datetime):
        return f"DATE '{value.isoformat()}'"
    if isinstance(value, _dt.datetime):
        # ISO format; Teradata TIMESTAMP literal
        ts = value.replace(tzinfo=None).isoformat(sep=' ', timespec='seconds')
        return f"TIMESTAMP '{ts}'"
    # default: treat as string with single quotes escaped
    s = str(value).replace("'", "''")
    return f"'{s}'"


def _inline_for_select(expr: sa.ColumnElement[Any]) -> sa.ColumnElement[Any]:
    """If the expression is a bound parameter, convert it to an inline literal column.

    This avoids `SELECT ?` which some backends (Teradata) reject. Labels are preserved by callers.
    """
    if isinstance(expr, sa.BindParameter):
        return sa.literal_column(_inline_sql_literal(expr.value))
    return expr


def _normalize_predicate(expr: sa.ColumnElement[Any]) -> sa.ColumnElement[Any]:
    """Normalize boolean predicates to dialect-friendly constructs.

    - Convert bound boolean parameters to sa.true()/sa.false()
    - Convert Expr objects to sa expressions (if they leaked here)
    - Leave other expressions untouched
    """
    from .expressions import Expr
    if isinstance(expr, Expr):
        # This shouldn't happen with proper _to_sa usage but let's be safe
        raise TypeError('Expr object reached _normalize_predicate; it should have been converted to sa expression.')
    if isinstance(expr, sa.BindParameter):
        val = expr.value
        if isinstance(val, bool):
            return sa.true() if val else sa.false()
    return expr
