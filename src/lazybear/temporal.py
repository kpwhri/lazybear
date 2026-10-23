from __future__ import annotations

import sqlalchemy as sa

from lazybear.autoconvert import str_to_col, date_to_col_or_lit
from lazybear.expressions import Expr, _to_sa


def calc_age(birth_date: Expr, reference_date: Expr) -> Expr:
    """Calculates age in completed years at a reference date.

    Args:
        birth_date: Birth date expression (e.g., `lb.col('birth_date')`)
        reference_date: Reference date expression: `lb.col` to reference another column,
            or `lb.lit` for a literal date (e.g., `lb.col('dxdate')` or `lb.lit(date(2000, 1, 1))`

    Returns:
        Age in completed years at ``reference_date``. Returns NULL if either date
        is NULL.

    Notes:
        The reference date may be a LazyBear expression, a SQLAlchemy column, or a
        Python ``date``. A Feb 29 birthday advances on Mar 1 in non-leap years.

    Examples:
        calculate_age(col('birth_date'), col('index_date'))
        calculate_age(col('birth_date'), lit(date(2026, 10, 21)))
    """

    def _compile(lf: 'LazyBearFrame') -> sa.ColumnElement:
        birth = _to_sa(str_to_col(birth_date), lf)
        reference = _to_sa(date_to_col_or_lit(reference_date), lf)

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
