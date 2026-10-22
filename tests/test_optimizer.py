import polars as pl
import sqlglot
from sqlglot import exp

import lazybear as lb
from lazybear.optimizer import optimize_query


def test_select_filter_collect_optimize(sqlite_engine):
    lf = lb.scan_table('users', sqlite_engine)
    lf = lf.filter(lb.col('age') > 30).select('id', 'age')

    expected = lf.collect()
    out = lf.collect(optimize='sqlite')

    assert isinstance(out, pl.DataFrame)
    assert set(out.columns) == {'id', 'age'}
    assert out.shape[0] == 2  # Tellervo, Ukko
    assert out.sort('id').equals(expected.sort('id'))


def test_select_filter_collect_optimize_rewrites(sqlite_engine):
    lf = lb.scan_table('users', sqlite_engine)
    lf = (
        lf.filter(lb.col('age') > lb.col('id'))
        .filter(lb.col('age') > lb.col('id'))
        .select('id', 'age')
    )

    expected = lf.collect()
    out = lf.collect(optimize='sqlite')

    assert out.sort('id').equals(expected.sort('id'))

    # Verify SQLGlot actually removed the redundant predicate.
    original = str(lf.to_select().compile(dialect=sqlite_engine.dialect))
    optimized = str(optimize_query(lf.to_select(), sqlite_engine, optimize='sqlite'))

    original_ast = sqlglot.parse_one(original, read='sqlite')
    optimized_ast = sqlglot.parse_one(optimized, read='sqlite')

    assert sum(isinstance(e, exp.GT) for e in original_ast.walk()) == 2
    assert sum(isinstance(e, exp.GT) for e in optimized_ast.walk()) == 1


def test_select_filter_collect_optimize_params(sqlite_engine):
    lf = lb.scan_table('users', sqlite_engine)
    lf = (
        lf.filter(lb.col('age') > 20)
        .filter(lb.col('age') < 100)
        .select('id', 'age')
    )

    expected_df = lf.collect()
    out_df = lf.collect(optimize='sqlite')

    assert out_df.sort('id').equals(expected_df.sort('id'))


def test_select_filter_collect_batches_optimize(sqlite_engine):
    lf = lb.scan_table('users', sqlite_engine)
    lf = lf.filter(lb.col('age') > 30).select('id', 'age')

    expected = lf.collect()
    batches = list(lf.collect_batches(chunk_size=1, optimize='sqlite'))

    assert len(batches) == 2
    assert all(isinstance(df, pl.DataFrame) for df in batches)
    assert all(df.shape[0] <= 1 for df in batches)

    out = pl.concat(batches)
    assert out.sort('id').equals(expected.sort('id'))


def test_select_filter_explain_optimize_params(sqlite_engine):
    lf = lb.scan_table('users', sqlite_engine)
    lf = (
        lf.filter(lb.col('age') > 20)
        .filter(lb.col('age') < 100)
        .select('id', 'age')
    )

    expected_str = lf.explain()
    out_str = lf.explain(optimize='sqlite')

    assert 'FROM (SELECT anon_4.id AS id' in expected_str
    assert out_str == (
        'SELECT "users"."id" AS "id", "users"."age" AS "age" '
        'FROM "users" AS "users" '
        'WHERE "users"."age" < 100 AND "users"."age" > 20'
    )


def test_select_filter_iterrows_optimize_params(sqlite_engine):
    lf = lb.scan_table('users', sqlite_engine)
    lf = (
        lf.filter(lb.col('age') > 20)
        .filter(lb.col('age') < 100)
        .select('id', 'age')
    )

    expected = list(lf.iter_rows())
    out = list(lf.iter_rows(optimize='sqlite'))

    assert out == expected


def test_optimize_query_none(sqlite_engine):
    lf = lb.scan_table('users', sqlite_engine)
    sel = lf.filter(lb.col('age') > 30).to_select()

    out = optimize_query(sel, sqlite_engine)

    assert out is sel
