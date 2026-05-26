import polars as pl
import pytest
import sqlalchemy as sa


@pytest.fixture()
def users_df():
    return pl.from_records([
        {'id': 1, 'name': 'Ahti', 'age': 30},
        {'id': 2, 'name': 'Kalma', 'age': 28},
        {'id': 3, 'name': 'Tellervo', 'age': 41},
        {'id': 4, 'name': 'Ukko', 'age': 41},
    ])


@pytest.fixture()
def orders_df():
    return pl.from_records([
        {'id': 10, 'user_id': 1, 'product_id': 100, 'amount': 12.5},
        {'id': 11, 'user_id': 1, 'product_id': 101, 'amount': 7.5},
        {'id': 12, 'user_id': 2, 'product_id': 102, 'amount': 99.0},
    ])


@pytest.fixture()
def products_df():
    return pl.from_records([
        {'id': 100, 'name': 'sampo', 'category': 'artifact'},
        {'id': 101, 'name': 'kantele', 'category': 'instrument'},
        {'id': 102, 'name': 'hiisi', 'category': 'myth'},
        {'id': 103, 'name': 'louhi', 'category': 'myth'},
    ])


@pytest.fixture()
def sqlite_engine(users_df, orders_df, products_df):
    eng = sa.create_engine('sqlite:///:memory:')
    meta = sa.MetaData()

    t_users = sa.Table(
        'users', meta,
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String),
        sa.Column('age', sa.Integer),
    )

    t_orders = sa.Table(
        'orders', meta,
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('user_id', sa.Integer),
        sa.Column('product_id', sa.Integer),
        sa.Column('amount', sa.Float),
    )

    t_products = sa.Table(
        'products', meta,
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String),
        sa.Column('category', sa.String),
    )

    meta.create_all(eng)

    with eng.begin() as conn:
        conn.execute(t_users.insert(), list(users_df.iter_rows(named=True)))
        conn.execute(t_orders.insert(), list(orders_df.iter_rows(named=True)))
        conn.execute(t_products.insert(), list(products_df.iter_rows(named=True)))

    yield eng


@pytest.fixture()
def sqlite_engine_mixcase():
    eng = sa.create_engine('sqlite:///:memory:')
    meta = sa.MetaData()
    t = sa.Table(
        'MixedCase', meta,
        sa.Column('ID', sa.Integer, primary_key=True),
        sa.Column('UserName', sa.String),
        sa.Column('AGE', sa.Integer),
    )
    meta.create_all(eng)
    with eng.begin() as conn:
        conn.execute(t.insert(), [{'ID': 1, 'UserName': 'Ahti', 'AGE': 30}])
    return eng
