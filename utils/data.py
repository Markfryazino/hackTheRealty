import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(PATH='data/P/'):
    base = pd.read_csv(PATH + 'price_housebase.tsv', sep='\t')
    train = pd.read_csv(PATH + 'price_train.tsv', sep='\t')
    test = pd.read_csv(PATH + 'price_test.tsv', sep='\t')
    sample = pd.read_csv(PATH + 'price_sample_submission.tsv', sep='\t')
    return base, train, test, sample


def build_mapping(train, base):
    keys = set(base['city_quadkey'])
    kd = pd.DataFrame(pd.Series(list(keys), name='quadkeys'))

    district_cols = ['beauty_cnt', 'shopping_cnt', 'cafe_restaurant_eating_out_cnt', 'entertainment_cnt',
                     'sport_cnt', 'chain_cnt', 'groceries_and_everyday_items_cnt', 'art_cnt', 'healthcare_cnt',
                     'laundry_and_repair_services_cnt']

    house_cols = ['building_series_id', 'flats_count', 'building_type', 'unified_address',
                  'expect_demolition', 'latitude', 'longitude', 'ceiling_height', 'has_elevator',
                  'build_year']

    bd = base[district_cols + ['city_quadkey']].set_index('city_quadkey').drop_duplicates()

    joint = kd.join(bd, on='quadkeys')

    to_rm = ['avg_price_sqm', 'month', 'city_quadkey', 'median_price_sqm']
    feats = list(train.columns)
    for el in to_rm:
        feats.remove(el)

    mean = train.groupby('city_quadkey')[feats].mean()
    joint = joint.join(mean, on='quadkeys').set_index('quadkeys')

    return joint


def preprocess(data, mapping):
    data = data[['month', 'city_quadkey']]
    data = data.join(mapping, on='city_quadkey')
    data['month_id'] = data['month'].apply(pd.to_datetime).apply(lambda x: x.month)
    data.drop(['month', 'city_quadkey'], axis=1, inplace=True)

    return data
