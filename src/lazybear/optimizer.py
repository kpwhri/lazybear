import sqlalchemy as sa

from sqlglot import exp
from sqlglot.dialects import Dialect
from sqlglot.optimizer import optimize as sqlglot_optimize


def optimize_query(select: sa.Select, engine: sa.Engine, optimize: str | None = None) -> sa.Select | sa.TextClause:
    if optimize is None:
        return select

    dialect = {
        'mssql': 'tsql',
    }.get(optimize.lower(), optimize.lower())

    # preserve backend sql syntax, but use named parameters
    named_dialect = type(engine.dialect)(paramstyle='named')

    compiled = select.compile(
        dialect=named_dialect,
        compile_kwargs={'render_postcompile': True},
    )

    # optimize via sqlglot
    glot_dialect = Dialect.get_or_raise(dialect)
    expression = sqlglot_optimize(
        str(compiled),
        dialect=glot_dialect,
    )

    # generate dialect-specific sql (retain sqlalchemy :name bind parameter convention)
    generator = glot_dialect.generator()
    generator.NAMED_PLACEHOLDER_TOKEN = ':'

    optimized_sql = generator.generate(expression)
    stmt = sa.text(optimized_sql)

    # validate that placeholders survived sql generation
    expected = {
        p.name
        for p in expression.find_all(exp.Placeholder)
        if p.this
    }
    actual = set(stmt.compile(dialect=named_dialect).params)

    if expected != actual:
        raise ValueError('sqlglot changed parameter placeholders')

    missing = actual - compiled.params.keys()
    if missing:
        raise ValueError(f'Missing parameter values: {missing}')

    # sqlglot may remove unused parameters
    params = {
        name: compiled.params[name]
        for name in actual
    }

    return stmt.bindparams(**params)
