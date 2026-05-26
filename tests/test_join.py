import pytest

import polars as pl

from lazybear import scan_table, col


def test_join(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)
    j = users.join(orders, on={'id': 'user_id'}, how='left', apply_to_all=False)
    out = j.select('id', 'name', 'amount').order_by('id', 'amount').collect()
    # user 3 and 4 have no orders, ensure left join kept them (amount null)
    ids = out['id'].to_list()
    assert 3 in ids and 4 in ids


def test_join_left_on_right_on_single(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)
    j = users.join(orders, left_on='id', right_on='user_id', how='left', apply_to_all=False)
    out = j.select('id', 'name', 'amount').order_by('id', 'amount').collect()
    ids = out['id'].to_list()
    assert 3 in ids and 4 in ids


def test_join_on_mapping_uses_left_keys_and_right_values(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    out_df = (
        users
        .join(orders, on={'id': 'user_id'}, how='left', apply_to_all=False)
        .select('id', 'name', 'amount')
        .order_by('id', 'amount')
        .collect()
    )

    assert out_df.filter(pl.col('id') == 1).height == 2
    assert out_df.filter(pl.col('id') == 2).height == 1
    assert out_df.filter(pl.col('id') == 3)['amount'][0] is None
    assert out_df.filter(pl.col('id') == 4)['amount'][0] is None


def test_join_on_string_uses_same_column_name_on_both_sides(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    users_again = scan_table('users', sqlite_engine)

    j = users.join(users_again, on='id', how='inner', apply_to_all=False)

    assert j.columns == ['id', 'name', 'age', 'name_right', 'age_right']
    out = j.select('id', 'name', 'name_right').order_by('id').collect()
    assert out['id'].to_list() == [1, 2, 3, 4]
    assert out['name'].to_list() == out['name_right'].to_list()


def test_join_on_sequence_uses_same_column_names_on_both_sides(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    users_again = scan_table('users', sqlite_engine)

    j = users.join(users_again, on=['id', 'age'], how='inner', apply_to_all=False)

    assert j.columns == ['id', 'name', 'age', 'name_right']
    out = j.select('id', 'age', 'name', 'name_right').order_by('id').collect()
    assert out['id'].to_list() == [1, 2, 3, 4]
    assert out['name'].to_list() == out['name_right'].to_list()


def test_join_left_on_right_on_sequences(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    j = users.join(orders, left_on=['id'], right_on=['user_id'], how='left', apply_to_all=False)

    assert j.columns == ['id', 'name', 'age', 'id_right', 'user_id', 'product_id', 'amount']
    out = j.select('id', 'user_id', 'amount').order_by('id', 'amount').collect()
    assert out.filter(pl.col('id') == 1).height == 2
    assert out.filter(pl.col('id') == 2).height == 1


def test_join_default_duplicate_columns_rename_only_overlaps(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    j = users.join(orders, on={'id': 'user_id'}, how='left')

    assert j.columns == ['id', 'name', 'age', 'id_right', 'user_id', 'product_id', 'amount']
    assert 'amount' in j.columns
    assert 'amount_right' not in j.columns


def test_join_default_generated_suffix_avoids_existing_left_names(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    left = users.with_columns(id_right=col('id'))
    j = left.join(orders, on={'id': 'user_id'}, how='left')

    assert 'id_right' in left.columns
    assert 'id_right2' in j.columns


def test_join_suffix_applies_to_all_right_columns_by_default(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    j = users.join(orders, on={'id': 'user_id'}, how='left', suffix='_order')

    assert j.columns == ['id', 'name', 'age', 'id_order', 'user_id_order', 'product_id_order', 'amount_order']


def test_join_suffix_can_apply_only_to_duplicate_right_columns(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    j = users.join(orders, on={'id': 'user_id'}, how='left', suffix='_order', apply_to_all=False)

    assert j.columns == ['id', 'name', 'age', 'id_order', 'user_id', 'product_id', 'amount']


def test_join_prefix_applies_to_all_right_columns_by_default(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    j = users.join(orders, on={'id': 'user_id'}, how='left', prefix='order_')

    assert j.columns == ['id', 'name', 'age', 'order_id', 'order_user_id', 'order_product_id', 'order_amount']


def test_join_prefix_can_apply_only_to_duplicate_right_columns(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    j = users.join(orders, on={'id': 'user_id'}, how='left', prefix='order_', apply_to_all=False)

    assert j.columns == ['id', 'name', 'age', 'order_id', 'user_id', 'product_id', 'amount']


def test_join_prefix_takes_precedence_over_suffix(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    j = users.join(
        orders,
        on={'id': 'user_id'},
        how='left',
        prefix='order_',
        suffix='_order',
    )

    assert j.columns == ['id', 'name', 'age', 'order_id', 'order_user_id', 'order_product_id', 'order_amount']
    assert 'id_order' not in j.columns
    assert 'user_id_order' not in j.columns
    assert 'amount_order' not in j.columns


def test_join_prefix_takes_precedence_over_suffixes(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    with pytest.warns(DeprecationWarning):
        j = users.join(
            orders,
            on={'id': 'user_id'},
            how='left',
            prefix='order_',
            suffixes=('_old_left', '_old_right'),
        )

    assert j.columns == ['id', 'name', 'age', 'order_id', 'order_user_id', 'order_product_id', 'order_amount']
    assert 'id_old_right' not in j.columns


def test_join_suffix_takes_precedence_over_suffixes(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    with pytest.warns(DeprecationWarning):
        j = users.join(
            orders,
            on={'id': 'user_id'},
            how='left',
            suffix='_order',
            suffixes=('_old_left', '_old_right'),
        )

    assert j.columns == ['id', 'name', 'age', 'id_order', 'user_id_order', 'product_id_order', 'amount_order']
    assert 'id_old_right' not in j.columns


def test_join_suffixes_still_supported_with_deprecation_warning(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    with pytest.warns(DeprecationWarning):
        j = users.join(orders, on={'id': 'user_id'}, how='left', suffixes=('_x', '_y'))

    assert j.columns == ['id', 'name', 'age', 'id_y', 'user_id_y', 'product_id_y', 'amount_y']


def test_join_suffixes_apply_to_duplicates_only_when_apply_to_all_false(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    with pytest.warns(DeprecationWarning):
        j = users.join(
            orders,
            on={'id': 'user_id'},
            how='left',
            suffixes=('_x', '_y'),
            apply_to_all=False,
        )

    assert j.columns == ['id', 'name', 'age', 'id_y', 'user_id', 'product_id', 'amount']


def test_join_duplicate_columns_drop_removes_overlapping_right_columns(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    j = users.join(
        orders,
        on={'id': 'user_id'},
        how='left',
        duplicate_columns='drop',
    )

    assert j.columns == ['id', 'name', 'age', 'user_id', 'product_id', 'amount']
    assert 'id_right' not in j.columns


def test_join_duplicate_columns_drop_with_apply_to_all_drops_all_right_columns_when_no_prefix_suffix(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    j = users.join(
        orders,
        on={'id': 'user_id'},
        how='left',
        duplicate_columns='drop',
        apply_to_all=True,
    )

    assert j.columns == ['id', 'name', 'age', 'user_id', 'product_id', 'amount']


def test_join_duplicate_columns_drop_with_prefix_drops_all_right_columns(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    j = users.join(
        orders,
        on={'id': 'user_id'},
        how='left',
        prefix='order_',
        duplicate_columns='drop',
    )

    assert j.columns == ['id', 'name', 'age']


def test_join_duplicate_columns_drop_with_suffix_drops_all_right_columns(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    j = users.join(
        orders,
        on={'id': 'user_id'},
        how='left',
        suffix='_order',
        duplicate_columns='drop',
    )

    assert j.columns == ['id', 'name', 'age']


def test_join_duplicate_columns_drop_with_prefix_and_apply_to_all_false_drops_only_overlaps(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    j = users.join(
        orders,
        on={'id': 'user_id'},
        how='left',
        prefix='order_',
        apply_to_all=False,
        duplicate_columns='drop',
    )

    assert j.columns == ['id', 'name', 'age', 'user_id', 'product_id', 'amount']


def test_join_duplicate_columns_drop_with_suffix_and_apply_to_all_false_drops_only_overlaps(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    j = users.join(
        orders,
        on={'id': 'user_id'},
        how='left',
        suffix='_order',
        apply_to_all=False,
        duplicate_columns='drop',
    )

    assert j.columns == ['id', 'name', 'age', 'user_id', 'product_id', 'amount']


@pytest.mark.parametrize('how', ['inner', 'left', 'right', 'full'])
def test_join_supported_how_values(sqlite_engine, how):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    out = (
        users
        .join(orders, on={'id': 'user_id'}, how=how, apply_to_all=False)
        .select('id', 'amount')
        .collect()
    )

    assert isinstance(out, pl.DataFrame)


@pytest.mark.parametrize('how', ['outer', 'cross', 'semi', 'anti', 'bad'])
def test_join_unsupported_how_values_raise(sqlite_engine, how):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    with pytest.raises(ValueError, match='how must be one of'):
        users.join(orders, on={'id': 'user_id'}, how=how).collect()


def test_join_rejects_non_lazybearframe(sqlite_engine):
    users = scan_table('users', sqlite_engine)

    with pytest.raises(TypeError, match='other must be a LazyBearFrame'):
        users.join(object(), on='id').collect()


def test_join_requires_left_and_right_on_when_on_omitted(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    with pytest.raises(ValueError, match='must provide both left_on and right_on'):
        users.join(orders, left_on='id').collect()

    with pytest.raises(ValueError, match='must provide both left_on and right_on'):
        users.join(orders, right_on='user_id').collect()


def test_join_rejects_on_with_left_or_right_on(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    with pytest.raises(ValueError, match='specify either on= or left_on=/right_on='):
        users.join(orders, on='id', left_on='id', right_on='user_id').collect()


def test_join_rejects_mismatched_left_and_right_key_counts(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    with pytest.raises(ValueError, match='left_on and right_on must have the same number of keys'):
        users.join(orders, left_on=['id', 'age'], right_on=['user_id']).collect()


def test_join_result_can_select_renamed_prefixed_columns(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    out = (
        users
        .join(orders, on={'id': 'user_id'}, how='left', prefix='order_')
        .select('id', 'order_id', 'order_amount')
        .order_by('id', 'order_amount')
        .collect()
    )

    assert out.columns == ['id', 'order_id', 'order_amount']
    assert out.filter(pl.col('id') == 1).height == 2


def test_join_result_can_select_renamed_suffixed_columns(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    out = (
        users
        .join(orders, on={'id': 'user_id'}, how='left', suffix='_order')
        .select('id', 'id_order', 'amount_order')
        .order_by('id', 'amount_order')
        .collect()
    )

    assert out.columns == ['id', 'id_order', 'amount_order']
    assert out.filter(pl.col('id') == 1).height == 2


def test_three_way_join_users_orders_products(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)
    products = scan_table('products', sqlite_engine)

    out = (
        users
        .join(orders, on={'id': 'user_id'}, how='left', prefix='order_')
        .join(products, on={'order_product_id': 'id'}, how='left', prefix='product_')
        .select(
            'id',
            'name',
            'order_id',
            'order_product_id',
            'order_amount',
            'product_name',
            'product_category',
        )
        .order_by('id', 'order_id')
        .collect()
    )

    assert isinstance(out, pl.DataFrame)
    assert out.columns == [
        'id',
        'name',
        'order_id',
        'order_product_id',
        'order_amount',
        'product_name',
        'product_category',
    ]

    assert out['id'].to_list() == [1, 1, 2, 3, 4]
    assert out['name'].to_list() == ['Ahti', 'Ahti', 'Kalma', 'Tellervo', 'Ukko']
    assert out['order_id'].to_list() == [10, 11, 12, None, None]
    assert out['order_product_id'].to_list() == [100, 101, 102, None, None]
    assert out['order_amount'].to_list() == [12.5, 7.5, 99.0, None, None]
    assert out['product_name'].to_list() == ['sampo', 'kantele', 'hiisi', None, None]
    assert out['product_category'].to_list() == ['artifact', 'instrument', 'myth', None, None]
