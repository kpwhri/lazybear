from datetime import date, datetime
from typing import Any, TypeAlias

import sqlalchemy as sa

from lazybear.expressions import Expr

ColumnInput: TypeAlias = Expr | sa.ColumnElement[Any] | str

DateInput: TypeAlias = ColumnInput | date | datetime

OffsetInput: TypeAlias = int | Expr | sa.ColumnElement[Any]
