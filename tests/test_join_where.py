import polars as pl
import pytest
import sqlalchemy as sa

from lazybear import col, scan_df


def product_price_predicates():
    return (
        col('product_id') == col('product_id_right'),
        col('order_date') >= col('valid_from'),
        col('order_date') <= col('valid_to'),
    )


def test_join_where_supports_equality_and_inclusive_date_range(dated_order_frames):
    orders, price_ranges = dated_order_frames

    out_df = (
        orders
        .join_where(price_ranges, *product_price_predicates())
        .order_by('order_date')
        .collect()
    )

    assert out_df.columns == [
        'order_id', 'product_id', 'order_date', 'amount',
        'product_id_right', 'valid_from', 'valid_to', 'price', 'price_label',
    ]
    assert out_df['order_id'].to_list() == [10, 11, 12]
    assert out_df['price_label'].to_list() == ['sampo winter', 'kantele festival', 'hiisi premium']


def test_join_where_accepts_an_iterable_of_predicates(dated_order_frames):
    orders, price_ranges = dated_order_frames

    out_df = orders.join_where(price_ranges, product_price_predicates()).collect()

    assert out_df.height == 3


def test_join_where_left_preserves_unmatched_left_rows(dated_order_frames):
    orders, price_ranges = dated_order_frames

    out_df = (
        orders
        .join_where(price_ranges, *product_price_predicates(), how='left')
        .order_by('order_date')
        .collect()
    )

    assert out_df.height == 4
    unmatched = out_df.filter(pl.col('order_id') == 13).row(0, named=True)
    assert unmatched['price_label'] is None
    assert unmatched['product_id_right'] is None


def test_join_where_right_preserves_unmatched_right_rows(dated_order_frames):
    orders, price_ranges = dated_order_frames

    out_df = (
        orders
        .join_where(price_ranges, *product_price_predicates(), how='right')
        .order_by('valid_from')
        .collect()
    )

    assert out_df.height == 5
    right_only = out_df.filter(pl.col('order_id').is_null())
    assert set(right_only['price_label']) == {'sampo holiday', 'louhi summer'}
    unmatched = right_only.filter(pl.col('price_label') == 'louhi summer').row(0, named=True)
    assert unmatched['order_id'] is None
    assert unmatched['amount'] is None
    assert unmatched['product_id_right'] == 103


def test_join_where_custom_suffix_is_used_in_predicates_and_output(dated_order_frames):
    orders, price_ranges = dated_order_frames

    out_df = orders.join_where(
        price_ranges,
        col('product_id') == col('product_id_y'),
        col('order_date') >= col('valid_from_y'),
        col('order_date') <= col('valid_to_y'),
        suffix='_y',
    ).collect()

    assert out_df.height == 3
    assert out_df.columns == [
        'order_id', 'product_id', 'order_date', 'amount',
        'product_id_y', 'valid_from_y', 'valid_to_y', 'price_y', 'price_label_y',
    ]
    assert 'product_id_right' not in out_df.columns


def test_join_where_suffix_can_apply_only_to_overlapping_columns(dated_order_frames):
    orders, price_ranges = dated_order_frames

    out_df = orders.join_where(
        price_ranges,
        col('product_id') == col('product_id_y'),
        col('order_date') >= col('valid_from'),
        col('order_date') <= col('valid_to'),
        suffix='_y',
        apply_to_all=False,
    ).collect()

    assert out_df.height == 3
    assert out_df.columns == [
        'order_id', 'product_id', 'order_date', 'amount',
        'product_id_y', 'valid_from', 'valid_to', 'price', 'price_label',
    ]


def test_join_where_prefix_applies_to_all_right_columns(dated_order_frames):
    orders, price_ranges = dated_order_frames

    out_df = orders.join_where(
        price_ranges,
        col('product_id') == col('range_product_id'),
        col('order_date') >= col('range_valid_from'),
        col('order_date') <= col('range_valid_to'),
        prefix='range_',
    ).collect()

    assert out_df.height == 3
    assert out_df.columns == [
        'order_id', 'product_id', 'order_date', 'amount',
        'range_product_id', 'range_valid_from', 'range_valid_to',
        'range_price', 'range_price_label',
    ]


def test_join_where_prefix_takes_precedence_over_suffix(dated_order_frames):
    orders, price_ranges = dated_order_frames

    out_df = orders.join_where(
        price_ranges,
        col('product_id') == col('range_product_id'),
        col('order_date') >= col('range_valid_from'),
        col('order_date') <= col('range_valid_to'),
        prefix='range_',
        suffix='_ignored',
    ).collect()

    assert out_df.height == 3
    assert 'range_product_id' in out_df.columns
    assert 'product_id_ignored' not in out_df.columns


def test_join_where_can_drop_overlapping_right_columns(dated_order_frames):
    orders, price_ranges = dated_order_frames

    out_df = orders.join_where(
        price_ranges,
        col('product_id') == col('product_id_right'),
        col('order_date') >= col('valid_from'),
        col('order_date') <= col('valid_to'),
        duplicate_columns='drop',
    ).collect()

    assert out_df.height == 3
    assert out_df.columns == [
        'order_id', 'product_id', 'order_date', 'amount',
        'valid_from', 'valid_to', 'price', 'price_label',
    ]


def test_join_where_drop_with_prefix_can_drop_all_right_columns(dated_order_frames):
    orders, price_ranges = dated_order_frames

    out_df = orders.join_where(
        price_ranges,
        col('product_id') == col('range_product_id'),
        col('order_date') >= col('range_valid_from'),
        col('order_date') <= col('range_valid_to'),
        prefix='range_',
        duplicate_columns='drop',
    ).collect()

    assert out_df.height == 3
    assert out_df.columns == ['order_id', 'product_id', 'order_date', 'amount']


def test_join_where_generated_suffix_avoids_existing_left_names(dated_order_frames):
    orders, price_ranges = dated_order_frames
    orders = orders.with_columns(product_id_right=col('product_id'))

    out_df = orders.join_where(
        price_ranges,
        col('product_id') == col('product_id_right2'),
        col('order_date') >= col('valid_from'),
        col('order_date') <= col('valid_to'),
    ).collect()

    assert out_df.height == 3
    assert 'product_id_right2' in out_df.columns


def test_join_where_rejects_invalid_duplicate_column_strategy(dated_order_frames):
    orders, price_ranges = dated_order_frames

    with pytest.raises(ValueError, match='duplicate_columns must be one of'):
        orders.join_where(
            price_ranges,
            *product_price_predicates(),
            duplicate_columns='invalid',
        )


@pytest.mark.parametrize('how', ['full', 'outer', 'cross', 'bad'])
def test_join_where_rejects_unsupported_join_types(dated_order_frames, how):
    orders, price_ranges = dated_order_frames

    with pytest.raises(ValueError, match='how must be one of'):
        orders.join_where(price_ranges, *product_price_predicates(), how=how)


def test_join_where_requires_at_least_one_predicate(dated_order_frames):
    orders, price_ranges = dated_order_frames

    with pytest.raises(ValueError, match='at least one predicate'):
        orders.join_where(price_ranges)


@pytest.mark.parametrize('predicate', [True, 'id = id_right', [True]])
def test_join_where_rejects_non_expression_predicates(dated_order_frames, predicate):
    orders, price_ranges = dated_order_frames

    with pytest.raises(TypeError, match='predicates must be Expr objects'):
        orders.join_where(price_ranges, predicate)


def test_join_where_rejects_non_lazybearframe(sqlite_engine):
    events = scan_df(pl.DataFrame({'id': [1]}), sqlite_engine)

    with pytest.raises(TypeError, match='other must be a LazyBearFrame'):
        events.join_where(object(), col('id') == 1)


def test_join_where_rejects_frames_from_different_servers():
    left_engine = sa.create_engine('sqlite:///:memory:')
    right_engine = sa.create_engine('sqlite+pysqlite:///:memory:')
    left = scan_df(pl.DataFrame({'id': [1]}), left_engine)
    right = scan_df(pl.DataFrame({'id': [1]}), right_engine)

    with pytest.raises(ValueError, match='different servers'):
        left.join_where(right, col('id') == col('id_right'))


def test_join_where_rejects_suffix_that_creates_duplicate_label(sqlite_engine):
    left = scan_df(pl.DataFrame({'id': [1], 'id_y': [1]}), sqlite_engine)
    right = scan_df(pl.DataFrame({'id': [1]}), sqlite_engine)

    with pytest.raises(ValueError, match='choose a different prefix or suffix'):
        left.join_where(right, col('id') == col('id_y'), suffix='_y')
