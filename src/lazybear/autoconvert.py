"""Auto-conversion of types to make the package more beginner friendly. May be disabled at any time."""
import re
import warnings
from datetime import date, datetime

from lazybear.expressions import Expr, lit, col


def str_to_col(colname) -> Expr:
    if isinstance(colname, str):
        warnings.warn(f"Column name '{colname}' was passed as `str` instead of `col`, use `lb.col('{colname}')`")
        colname = col(colname)
    return colname


def date_to_col_or_lit(colname) -> Expr:
    if isinstance(colname, str):
        if m := re.match(r'(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})', colname):
            year = int(m.group('y'))
            month = int(m.group('m'))
            day = int(m.group('d'))
            warnings.warn(f"Literal date passed as date, instead, use `lb.lit(date({year}, {month}, {day}))`.")
            return lit(date(year, month, day))
        else:
            warnings.warn(f"Column name '{colname}' was passed as `str` instead of `col`, use `lb.col('{colname}')`")
            return col(colname)
    elif isinstance(colname, datetime):
        warnings.warn(f"Literal date passed as datetime, instead, use `lb.lit(dt.date())`.")
        return lit(colname.date())
    elif isinstance(colname, date):
        # Allow this without a warning -- I don't think this requires a `lb.lit`
        # warnings.warn(f"Literal date passed as date, instead, use `lb.lit(date)`.")
        return lit(colname)
    return colname  # already an expression
