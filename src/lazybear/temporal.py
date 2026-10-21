from __future__ import annotations

from typing import Any

import sqlalchemy as sa

from lazybear.expressions import Expr, _to_sa


def calc_age(birth_date: Any, reference_date: Any) -> Expr:
    """Return age in completed years at a reference date.

    The reference date may be a LazyBear expression, a sqlalchemy column,
    or a Python ``date``. The result is NULL if either date is NULL.
    A Feb 29 birthday advances on Mar 1 in non-leap years.

    Examples:
        calculate_age(col('birth_date'), col('index_date'))
        calculate_age(col('birth_date'), date(2026, 10, 21))
    """

    def _compile(lf: 'LazyBearFrame') -> sa.ColumnElement[Any]:
        birth = _to_sa(birth_date, lf)
        reference = _to_sa(reference_date, lf)

        birth_month = sa.extract('month', birth)
        birth_day = sa.extract('day', birth)
        reference_month = sa.extract('month', reference)
        reference_day = sa.extract('day', reference)

        before_birthday = sa.or_(
            reference_month < birth_month,
            sa.and_(reference_month == birth_month, reference_day < birth_day),
        )

        return (
                sa.extract('year', reference)
                - sa.extract('year', birth)
                - sa.case((before_birthday, 1), else_=0)
        )

    return Expr(_compile)
