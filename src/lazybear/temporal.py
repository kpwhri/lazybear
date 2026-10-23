from __future__ import annotations

from numbers import Integral
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement

from lazybear.autoconvert import str_to_col, date_to_col_or_lit
from lazybear.dtypes import ColumnInput, DateInput, OffsetInput
from lazybear.expressions import Expr, _to_sa

# compile-time validated units are deliberately not sql parameters for cross-db support
_DATE_TRUNC_UNITS = ('day', 'month', 'quarter', 'year')
_DATE_PART_UNITS = ('year', 'month', 'day', 'quarter', 'weekday')


class _DateAdd(FunctionElement):
    type = sa.Date()
    inherit_cache = False  # 'unit' affects compiled SQL but is not a clause
    name = 'lazybear_date_add'

    def __init__(self, value: Any, amount: Any, unit: str):
        super().__init__(value, amount)
        self.unit = unit


class _DateDiff(FunctionElement):
    type = sa.Integer()
    inherit_cache = True
    name = 'lazybear_date_diff'


class _DateTrunc(FunctionElement):
    type = sa.Date()
    inherit_cache = False  # 'unit' affects compiled SQL but is not a clause
    name = 'lazybear_date_trunc'

    def __init__(self, value: Any, unit: str):
        super().__init__(value)
        self.unit = unit


class _IsoWeekday(FunctionElement):
    type = sa.Integer()
    inherit_cache = True
    name = 'lazybear_iso_weekday'


def _sql_args(element: FunctionElement, compiler: Any, **kw: Any) -> tuple[str, ...]:
    return tuple(compiler.process(arg, **kw) for arg in element.clauses)


@compiles(_DateAdd, 'sqlite')
def _add_sqlite(element: _DateAdd, compiler: Any, **kw: Any) -> str:
    value, amount = _sql_args(element, compiler, **kw)
    return f"DATE({value}, PRINTF('%+d {element.unit}s', {amount}))"


@compiles(_DateAdd, 'postgresql')
def _add_postgresql(element: _DateAdd, compiler: Any, **kw: Any) -> str:
    value, amount = _sql_args(element, compiler, **kw)
    if element.unit == 'day':
        return f'CAST({value} AS DATE) + CAST({amount} AS INTEGER)'
    return f"CAST(({value} + ({amount} * INTERVAL '1 {element.unit}')) AS DATE)"


@compiles(_DateAdd, 'mssql')
def _add_mssql(element: _DateAdd, compiler: Any, **kw: Any) -> str:
    value, amount = _sql_args(element, compiler, **kw)
    return f'DATEADD({element.unit}, {amount}, {value})'


@compiles(_DateAdd, 'mysql')
def _add_mysql(element: _DateAdd, compiler: Any, **kw: Any) -> str:
    value, amount = _sql_args(element, compiler, **kw)
    return f'DATE_ADD({value}, INTERVAL {amount} {element.unit.upper()})'


@compiles(_DateAdd, 'oracle')
def _add_oracle(element: _DateAdd, compiler: Any, **kw: Any) -> str:
    value, amount = _sql_args(element, compiler, **kw)
    if element.unit == 'day':
        return f'({value} + {amount})'
    if element.unit == 'year':
        return f'ADD_MONTHS({value}, 12 * {amount})'
    return f'ADD_MONTHS({value}, {amount})'


@compiles(_DateDiff, 'sqlite')
def _diff_sqlite(element: _DateDiff, compiler: Any, **kw: Any) -> str:
    start, end = _sql_args(element, compiler, **kw)
    return f'CAST((JULIANDAY(DATE({end})) - JULIANDAY(DATE({start}))) AS INTEGER)'


@compiles(_DateDiff, 'postgresql')
def _diff_postgresql(element: _DateDiff, compiler: Any, **kw: Any) -> str:
    start, end = _sql_args(element, compiler, **kw)
    return f'(CAST({end} AS DATE) - CAST({start} AS DATE))'


@compiles(_DateDiff, 'mssql')
def _diff_mssql(element: _DateDiff, compiler: Any, **kw: Any) -> str:
    start, end = _sql_args(element, compiler, **kw)
    return f'DATEDIFF(day, {start}, {end})'


@compiles(_DateDiff, 'mysql')
def _diff_mysql(element: _DateDiff, compiler: Any, **kw: Any) -> str:
    start, end = _sql_args(element, compiler, **kw)
    return f'DATEDIFF({end}, {start})'


@compiles(_DateDiff, 'oracle')
def _diff_oracle(element: _DateDiff, compiler: Any, **kw: Any) -> str:
    start, end = _sql_args(element, compiler, **kw)
    return f'CAST((TRUNC({end}) - TRUNC({start})) AS INTEGER)'


@compiles(_DateTrunc, 'sqlite')
def _trunc_sqlite(element: _DateTrunc, compiler: Any, **kw: Any) -> str:
    (value,) = _sql_args(element, compiler, **kw)
    if element.unit == 'day':
        return f'DATE({value})'
    if element.unit == 'month':
        return f"DATE({value}, 'start of month')"
    if element.unit == 'year':
        return f"DATE({value}, 'start of year')"
    return (
        f"DATE({value}, 'start of year', "
        f"PRINTF('+%d months', ((CAST(STRFTIME('%m', {value}) AS INTEGER) - 1) / 3) * 3))"
    )


@compiles(_DateTrunc, 'postgresql')
def _trunc_postgresql(element: _DateTrunc, compiler: Any, **kw: Any) -> str:
    (value,) = _sql_args(element, compiler, **kw)
    return f"CAST(DATE_TRUNC('{element.unit}', {value}) AS DATE)"


@compiles(_DateTrunc, 'mssql')
def _trunc_mssql(element: _DateTrunc, compiler: Any, **kw: Any) -> str:
    (value,) = _sql_args(element, compiler, **kw)
    if element.unit == 'day':
        return f'CAST({value} AS DATE)'
    if element.unit == 'year':
        return f'DATEFROMPARTS(YEAR({value}), 1, 1)'
    if element.unit == 'month':
        return f'DATEFROMPARTS(YEAR({value}), MONTH({value}), 1)'
    return f'DATEFROMPARTS(YEAR({value}), (DATEPART(quarter, {value}) - 1) * 3 + 1, 1)'


@compiles(_DateTrunc, 'mysql')
def _trunc_mysql(element: _DateTrunc, compiler: Any, **kw: Any) -> str:
    (value,) = _sql_args(element, compiler, **kw)
    if element.unit == 'day':
        return f'DATE({value})'
    if element.unit == 'year':
        return f'MAKEDATE(YEAR({value}), 1)'
    if element.unit == 'month':
        return f'DATE_SUB(DATE({value}), INTERVAL (DAYOFMONTH({value}) - 1) DAY)'
    return f'DATE_ADD(MAKEDATE(YEAR({value}), 1), INTERVAL ((QUARTER({value}) - 1) * 3) MONTH)'


@compiles(_DateTrunc, 'oracle')
def _trunc_oracle(element: _DateTrunc, compiler: Any, **kw: Any) -> str:
    (value,) = _sql_args(element, compiler, **kw)
    code = {'day': 'DD', 'month': 'MM', 'quarter': 'Q', 'year': 'YYYY'}[element.unit]
    return f"TRUNC({value}, '{code}')"


@compiles(_IsoWeekday, 'sqlite')
def _weekday_sqlite(element: _IsoWeekday, compiler: Any, **kw: Any) -> str:
    (value,) = _sql_args(element, compiler, **kw)
    return f"((CAST(STRFTIME('%w', {value}) AS INTEGER) + 6) % 7 + 1)"


@compiles(_IsoWeekday, 'postgresql')
def _weekday_postgresql(element: _IsoWeekday, compiler: Any, **kw: Any) -> str:
    (value,) = _sql_args(element, compiler, **kw)
    return f'CAST(EXTRACT(ISODOW FROM {value}) AS INTEGER)'


@compiles(_IsoWeekday, 'mssql')
def _weekday_mssql(element: _IsoWeekday, compiler: Any, **kw: Any) -> str:
    (value,) = _sql_args(element, compiler, **kw)
    # DATEPART(weekday) depends on DATEFIRST; this does not
    diff = f"DATEDIFF(day, '20000103', {value})"
    return f'((({diff} % 7) + 7) % 7 + 1)'


@compiles(_IsoWeekday, 'mysql')
def _weekday_mysql(element: _IsoWeekday, compiler: Any, **kw: Any) -> str:
    (value,) = _sql_args(element, compiler, **kw)
    return f'(WEEKDAY({value}) + 1)'


@compiles(_IsoWeekday, 'oracle')
def _weekday_oracle(element: _IsoWeekday, compiler: Any, **kw: Any) -> str:
    (value,) = _sql_args(element, compiler, **kw)
    return f"(TRUNC({value}) - TRUNC({value}, 'IW') + 1)"


def calc_age(birth_date: ColumnInput, reference_date: DateInput) -> Expr:
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


def date_add(value: DateInput, *, days: OffsetInput = 0, months: OffsetInput = 0, years: OffsetInput = 0) -> Expr:
    """Adds or subtracts calendar days, months, or years from a date.

    Args:
        value: Date expression (e.g., `lb.col('index_date')` or
            `lb.lit(date(2026, 1, 1))`).
        days: Number of days to add. Negative values subtract days.
            May be an integer or SQL expression. Defaults to 0.
        months: Number of months to add or subtract. Defaults to 0.
        years: Number of years to add or subtract. Defaults to 0.

    Returns:
        Date expression with the specified offsets applied. Returns NULL
        if the input date is NULL.

    Notes:
        Offsets are applied in years, months, days order. Month-end behavior
        may differ between databases (e.g., adding one month to Jan 31).
        Day offsets are recommended for reproducible clinical study windows.
        Python dates are accepted directly; strings are automatically
        converted with a warning.

    Examples:
        date_add(col('index_date'), days=90)
        date_add(col('index_date'), days=-365)
        date_add(col('index_date'), months=6)
        date_add(col('index_date'), years=1, days=-1)
    """
    for unit, amount in (('year', years), ('month', months), ('day', days)):
        if isinstance(amount, bool) or not isinstance(amount, (Integral, Expr, sa.ColumnElement)):
            raise TypeError(f'{unit}s must be an integer or a sql expression')

    def _compile(lf: 'LazyBearFrame') -> sa.ColumnElement:
        result = _to_sa(date_to_col_or_lit(value), lf)
        for unit, amount in (('year', years), ('month', months), ('day', days)):
            if isinstance(amount, Integral) and amount == 0:
                continue
            result = _DateAdd(result, _to_sa(amount, lf), unit)
        return result

    return Expr(_compile)


def date_diff(start_date: DateInput, end_date: DateInput) -> Expr:
    """Calculates the signed number of calendar days between two dates.

    Args:
        start_date: Starting date expression (e.g., `lb.col('index_date')`).
        end_date: Ending date expression (e.g., `lb.col('diagnosis_date')`).
            Either date may also be a Python ``date`` or `lb.lit` expression.

    Returns:
        Integer number of days calculated as ``end_date - start_date``.
        Positive values indicate a later end date, negative values an
        earlier end date. Returns NULL if either date is NULL.

    Notes:
        Compares calendar dates rather than elapsed 24-hour periods.
        Time-of-day is ignored. String arguments are automatically converted
        with a warning recommending explicit `lb.col` or `lb.lit` usage.

    Examples:
        date_diff(col('index_date'), col('diagnosis_date'))
        date_diff(col('birth_date'), lit(date(2026, 10, 21)))
    """
    return Expr(lambda lf: _DateDiff(
        _to_sa(date_to_col_or_lit(start_date), lf),
        _to_sa(date_to_col_or_lit(end_date), lf)
    ))


def date_trunc(value: DateInput, unit: Literal['day', 'month', 'quarter', 'year']) -> Expr:
    """Truncates a date to the beginning of a calendar period.

    Args:
        value: Date expression (e.g., `lb.col('diagnosis_date')`).
        unit: Calendar period to truncate to. One of 'day', 'month',
            'quarter', or 'year'.

    Returns:
        Date representing the beginning of the specified period.
        Returns NULL if the input date is NULL.

    Notes:
        For quarters, the result is the first day of January, April, July,
        or October. The result is a date rather than a timestamp.
        Raises ValueError for unsupported units.

    Examples:
        date_trunc(col('diagnosis_date'), 'month')
        date_trunc(col('admission_date'), 'quarter')
        date_trunc(col('index_date'), 'year')
    """
    if unit not in _DATE_TRUNC_UNITS:
        raise ValueError(f'unit must be one of {_DATE_TRUNC_UNITS}')
    return Expr(lambda lf: _DateTrunc(_to_sa(date_to_col_or_lit(value), lf), unit))


def date_part(value: DateInput, part: Literal['year', 'month', 'day', 'quarter', 'weekday']) -> Expr:
    """Extracts a calendar component from a date.

    Args:
        value: Date expression (e.g., `lb.col('diagnosis_date')`).
        part: Component to extract. One of 'year', 'month', 'day',
            'quarter', or 'weekday'.

    Returns:
        Integer expression containing the requested date component.
        Returns NULL if the input date is NULL.

    Notes:
        Quarters are numbered 1 through 4. Weekdays follow ISO conventions:
        Monday = 1 through Sunday = 7, consistently across databases.
        Raises ValueError for unsupported parts.

    Examples:
        date_part(col('diagnosis_date'), 'year')
        date_part(col('diagnosis_date'), 'month')
        date_part(col('admission_date'), 'weekday')
        date_part(col('index_date'), 'quarter')
    """
    if part not in _DATE_PART_UNITS:
        raise ValueError(f'part must be one of {_DATE_PART_UNITS}')

    def _compile(lf: 'LazyBearFrame') -> sa.ColumnElement:
        expr = _to_sa(date_to_col_or_lit(value), lf)
        if part == 'weekday':
            return _IsoWeekday(expr)
        if part == 'quarter':
            month = sa.extract('month', expr)
            return sa.case(
                (month <= 3, 1),
                (month <= 6, 2),
                (month <= 9, 3),
                (month <= 12, 4),
                else_=None,
            )
        return sa.extract(part, expr)

    return Expr(_compile)


def between(value: DateInput, start: DateInput, end: DateInput) -> Expr:
    """Checks whether a date falls between two inclusive boundaries.

    Args:
        value: Date expression to evaluate (e.g., `lb.col('dxdate')`).
        start: Inclusive lower date boundary, as a column, expression,
            or literal date.
        end: Inclusive upper date boundary, as a column, expression,
            or literal date.

    Returns:
        Boolean expression indicating whether ``start <= value <= end``.
        Returns NULL when a required comparison is unknown due to NULL.

    Notes:
        Both boundaries are inclusive. Equivalent to SQL BETWEEN.
        Boundaries should be provided in ascending order.
        For relative date windows, consider `in_interval` or `in_window`.

    Examples:
        between(col('dxdate'), col('start_date'), col('end_date'))
        between(col('dxdate'), date(2026, 1, 1), date(2026, 12, 31))
        col('dxdate').between(col('start_date'), col('end_date'))
    """
    return Expr(lambda lf: _to_sa(date_to_col_or_lit(value), lf).between(
        _to_sa(date_to_col_or_lit(start), lf),
        _to_sa(date_to_col_or_lit(end), lf),
    ))


def in_window(event_date: DateInput, reference_date: DateInput, *, start_days: int, end_days: int) -> Expr:
    """Checks whether an event occurs within a relative date window.

    Args:
        event_date: Event date expression (e.g., `lb.col('dxdate')`).
        reference_date: Date from which the window is calculated
            (e.g., `lb.col('index_date')`).
        start_days: Signed day offset defining the inclusive window start.
            Negative values indicate dates before the reference date.
        end_days: Signed day offset defining the inclusive window end.
            Positive values indicate dates after the reference date.

    Returns:
        Boolean expression indicating whether the event falls within the
        specified window, including both endpoints. Returns NULL when
        the result cannot be determined due to missing dates.

    Notes:
        Unlike `in_interval`, the window can have independent start and
        end offsets. Requires start_days <= end_days.
        Raises ValueError if the offsets are reversed.

    Examples:
        in_window(col('dxdate'), col('index_date'),
                  start_days=-365, end_days=0)
        in_window(col('dxdate'), col('index_date'),
                  start_days=-30, end_days=30)
        in_window(col('outcome_date'), col('surgery_date'),
                  start_days=1, end_days=90)
    """
    if isinstance(start_days, bool) or not isinstance(start_days, Integral):
        raise TypeError('start_days must be an integer')
    if isinstance(end_days, bool) or not isinstance(end_days, Integral):
        raise TypeError('end_days must be an integer')
    if start_days > end_days:
        raise ValueError('start_days must be <= end_days')
    return between(
        date_to_col_or_lit(event_date),
        date_add(date_to_col_or_lit(reference_date), days=date_to_col_or_lit(start_days)),
        date_add(date_to_col_or_lit(reference_date), days=date_to_col_or_lit(end_days)),
    )


def in_interval(event_date: DateInput, reference_date: DateInput, ndays: int = 90) -> Expr:
    """Checks whether an event occurs within a relative date interval.

    Args:
        event_date: Event date expression (e.g., `lb.col('diagnosis_date')`).
        reference_date: Date from which the interval is calculated
            (e.g., `lb.col('index_date')`).
        ndays: Signed number of days defining the interval.
            Positive values look forward; negative values look backward.
            Defaults to 90.

    Returns:
        Boolean expression indicating whether the event falls within the
        interval, including both endpoints. Returns NULL when the result
        cannot be determined due to missing dates.

    Notes:
        Positive ``ndays`` includes dates from the reference date through
        reference date + ndays. Negative values include dates from
        reference date + ndays through the reference date.
        Zero matches the reference date itself. An interval of 90 days
        includes day 0 and day 90 (91 calendar dates).

    Examples:
        in_interval(col('dxdate'), col('index_date'), ndays=90)
        in_interval(col('dxdate'), col('index_date'), ndays=-365)
        in_interval(col('outcome_date'), col('treatment_date'), ndays=30)
    """
    if isinstance(ndays, bool) or not isinstance(ndays, Integral):
        raise TypeError('ndays must be an integer')
    return in_window(
        date_to_col_or_lit(event_date),
        date_to_col_or_lit(reference_date),
        start_days=min(0, ndays),
        end_days=max(0, ndays),
    )


def intervals_overlap(a_start: DateInput, a_end: DateInput, b_start: DateInput, b_end: DateInput) -> Expr:
    """Checks whether two date intervals overlap.

    Args:
        a_start: Start date of the first interval.
        a_end: End date of the first interval.
        b_start: Start date of the second interval.
        b_end: End date of the second interval.
            Each argument may be a column expression or literal date.

    Returns:
        Boolean expression indicating whether the intervals overlap.
        NULL inputs follow SQL three-valued boolean logic.

    Notes:
        Intervals are half-open: [start, end). The start date is included
        and the end date is excluded. Intervals that touch at their
        endpoints do not overlap. Empty or reversed intervals return False.

        For clinical periods with inclusive end dates, add one day to each
        end date before calling this function.

    Examples:
        intervals_overlap(
            col('admission_date'), col('discharge_date'),
            col('study_start'), col('study_end'),
        )
        intervals_overlap(
            col('rx_start'), col('rx_end'),
            col('followup_start'), col('followup_end'),
        )
    """

    def _compile(lf: 'LazyBearFrame') -> sa.ColumnElement:
        a0, a1 = (
            _to_sa(date_to_col_or_lit(a_start), lf),
            _to_sa(date_to_col_or_lit(a_end), lf),
        )
        b0, b1 = (
            _to_sa(date_to_col_or_lit(b_start), lf),
            _to_sa(date_to_col_or_lit(b_end), lf),
        )
        return sa.and_(a0 < a1, b0 < b1, a0 < b1, b0 < a1)

    return Expr(_compile)
