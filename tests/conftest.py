from datetime import date

import polars as pl
import pytest
import sqlalchemy as sa

from lazybear import scan_table


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
def dated_orders_df():
    """Orders with dates chosen to exercise range boundaries and non-matches."""
    return pl.from_records([
        {'order_id': 10, 'product_id': 100, 'order_date': date(2024, 1, 1), 'amount': 12.5},
        {'order_id': 11, 'product_id': 101, 'order_date': date(2024, 2, 15), 'amount': 7.5},
        {'order_id': 12, 'product_id': 102, 'order_date': date(2024, 3, 31), 'amount': 99.0},
        {'order_id': 13, 'product_id': 103, 'order_date': date(2024, 4, 1), 'amount': 50.0},
    ])


@pytest.fixture()
def product_price_ranges_df():
    """Effective-dated prices for the products used throughout the test suite."""
    return pl.from_records([
        {
            'product_id': 100,
            'valid_from': date(2023, 12, 1),
            'valid_to': date(2023, 12, 31),
            'price': 10.0,
            'price_label': 'sampo holiday',
        },
        {
            'product_id': 100,
            'valid_from': date(2024, 1, 1),
            'valid_to': date(2024, 1, 31),
            'price': 12.5,
            'price_label': 'sampo winter',
        },
        {
            'product_id': 101,
            'valid_from': date(2024, 2, 1),
            'valid_to': date(2024, 2, 29),
            'price': 7.5,
            'price_label': 'kantele festival',
        },
        {
            'product_id': 102,
            'valid_from': date(2024, 3, 1),
            'valid_to': date(2024, 3, 31),
            'price': 99.0,
            'price_label': 'hiisi premium',
        },
        {
            'product_id': 103,
            'valid_from': date(2024, 5, 1),
            'valid_to': date(2024, 5, 31),
            'price': 50.0,
            'price_label': 'louhi summer',
        },
    ])


@pytest.fixture()
def hero_df():
    return pl.from_records([
        {'id': 1, 'first_name': 'Väinämöinen', 'nickname': None, 'backup_name': None, 'score': 100, 'bonus': None},
        {'id': 2, 'first_name': None, 'nickname': 'Jouk', 'backup_name': 'Joukahainen', 'score': None, 'bonus': 50},
        {'id': 3, 'first_name': None, 'nickname': None, 'backup_name': 'Ilmarinen', 'score': None, 'bonus': None},
        {'id': 4, 'first_name': 'Lemminkäinen', 'nickname': 'Lempi', 'backup_name': None, 'score': 90, 'bonus': 10},
        {'id': 5, 'first_name': 'Kullervo', 'nickname': None, 'backup_name': 'Kalervo', 'score': 70, 'bonus': None},
        {'id': 6, 'first_name': None, 'nickname': 'Louhi', 'backup_name': 'Mistress of Pohjola', 'score': None,
         'bonus': 95},
        {'id': 7, 'first_name': 'Aino', 'nickname': None, 'backup_name': None, 'score': None, 'bonus': None},
        {'id': 8, 'first_name': None, 'nickname': None, 'backup_name': 'Marjatta', 'score': 60, 'bonus': 5},
    ])


@pytest.fixture()
def sqlite_engine(
        users_df,
        orders_df,
        products_df,
        hero_df,
        dated_orders_df,
        product_price_ranges_df,
):
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

    t_heroes = sa.Table(
        'heroes', meta,
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('first_name', sa.String),
        sa.Column('nickname', sa.String),
        sa.Column('backup_name', sa.String),
        sa.Column('score', sa.Integer),
        sa.Column('bonus', sa.Integer),
    )

    t_dated_orders = sa.Table(
        'dated_orders', meta,
        sa.Column('order_id', sa.Integer, primary_key=True),
        sa.Column('product_id', sa.Integer),
        sa.Column('order_date', sa.Date),
        sa.Column('amount', sa.Float),
    )

    t_product_price_ranges = sa.Table(
        'product_price_ranges', meta,
        sa.Column('product_id', sa.Integer, primary_key=True),
        sa.Column('valid_from', sa.Date, primary_key=True),
        sa.Column('valid_to', sa.Date),
        sa.Column('price', sa.Float),
        sa.Column('price_label', sa.String),
    )

    meta.create_all(eng)

    with eng.begin() as conn:
        conn.execute(t_users.insert(), list(users_df.iter_rows(named=True)))
        conn.execute(t_orders.insert(), list(orders_df.iter_rows(named=True)))
        conn.execute(t_products.insert(), list(products_df.iter_rows(named=True)))
        conn.execute(t_heroes.insert(), list(hero_df.iter_rows(named=True)))
        conn.execute(t_dated_orders.insert(), list(dated_orders_df.iter_rows(named=True)))
        conn.execute(t_product_price_ranges.insert(), list(product_price_ranges_df.iter_rows(named=True)))

    yield eng


@pytest.fixture()
def dated_order_frames(sqlite_engine):
    """Frames for an equality join constrained by an effective date range."""
    return (
        scan_table('dated_orders', sqlite_engine),
        scan_table('product_price_ranges', sqlite_engine),
    )


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
