import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .preproc import dist_to_center, closest_tube
from sklearn.linear_model import LinearRegression
from tqdm import tqdm_notebook as tqdm


def load_data(PATH='data/P/'):
    base = pd.read_csv(PATH + 'price_housebase.tsv', sep='\t')
    train = pd.read_csv(PATH + 'price_train.tsv', sep='\t')
    test = pd.read_csv(PATH + 'price_test.tsv', sep='\t')
    sample = pd.read_csv(PATH + 'price_sample_submission.tsv', sep='\t')
    return base, train, test, sample


def build_mapping(train, base, path='../utils/'):
    keys = set(base['city_quadkey'])
    kd = pd.DataFrame(pd.Series(list(keys), name='quadkeys'))
    base = pd.concat([base, pd.get_dummies(base['building_type'])], axis=1)

    latlon = pd.Series(list(zip(base.latitude.values, base.longitude.values)))
    d2c = dist_to_center(latlon)
    ct = closest_tube(latlon, path=path)
    base = pd.concat([d2c, ct, base], axis=1)

    district_cols = ['beauty_cnt', 'shopping_cnt', 'cafe_restaurant_eating_out_cnt', 'entertainment_cnt',
                     'sport_cnt', 'chain_cnt', 'groceries_and_everyday_items_cnt', 'art_cnt', 'healthcare_cnt',
                     'laundry_and_repair_services_cnt']

    float_cols = ['flats_count', 'ceiling_height', 'build_year', 'angle', 'dist',
                  'dist_to_closest_tube']
    bin_cols = ['expect_demolition', 'has_elevator', 'BLOCK', 'BRICK', 'MONOLIT',
                'MONOLIT_BRICK', 'PANEL', 'UNKNOWN', 'WOOD']

    bd = base[district_cols + ['city_quadkey']].set_index('city_quadkey').drop_duplicates()

    joint = kd.join(bd, on='quadkeys')

    aggs_count = base.groupby('city_quadkey')['longitude'].count()
    aggs_float = base.groupby('city_quadkey')[float_cols].agg(['mean', 'median'])
    aggs_bin = base.groupby('city_quadkey')[bin_cols].agg('mean')

    aggs = pd.DataFrame(pd.Series(aggs_count, name='count'))
    for fun in ['mean', 'median']:
        for col in float_cols:
            aggs[col + '_' + fun] = aggs_float[col][fun]

    for col in bin_cols:
        aggs[col + '_mean'] = aggs_bin[col]

    aggs['line_color'] = base.groupby('city_quadkey')['line_color'].agg(lambda x: x.value_counts().index[0])

    joint = joint.join(aggs, on='quadkeys')

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
    data['month'] = data['month'].apply(pd.to_datetime)
    data['month_id'] = data['month'].apply(lambda x: x.month)
    #  data.drop(['month', 'city_quadkey'], axis=1, inplace=True)

    return data


def month_to_days(x):
    mm = pd.Timestamp(year=2017, month=1, day=1)
    return (x - mm).days


def for_reg(train, deg):
    train['days_1'] = train['month'].apply(month_to_days)
    for i in range(2, deg + 1):
        train[f'deg_{i}'] = train['days_1'].apply(lambda x: x ** i)
    y = train['y'].values
    X = train.drop(['month', 'y'], axis=1).values
    return X, y


def to_mat(test, deg):
    arr = pd.DataFrame()
    arr['deg_1'] = test
    for i in range(2, deg + 1):
        arr['deg_' + str(i)] = arr[f'deg_{i - 1}'] * arr['deg_1']
    return arr.values


def fit_general(train_x, train_y):
    temp = train_x[['month']]
    temp['y'] = train_y

    X, y = for_reg(temp, 2)
    model = LinearRegression()
    model.fit(X, y)
    return {'bias': model.intercept_, 'c1': model.coef_[0], 'c2': model.coef_[1]}, model


def plot_model(coef):
    #  plt.figure(figsize=(11, 6))

    to_pred = np.array(list(range(0, 1200, 30)))
    mm = pd.Timestamp(year=2017, month=1, day=1)
    to_draw = [mm + pd.Timedelta(days=d) for d in to_pred]

    plt.plot(to_draw, to_pred * coef['c1'] + to_pred * to_pred * coef['c2'] + coef['bias'])


def build_ans(train_x, train_y, general_coef, min_num=10):
    qk = train_x['city_quadkey'].unique()

    temp = train_x[['city_quadkey', 'month']]
    temp['y'] = train_y

    ans = pd.DataFrame(columns=['city_quadkey', 'bias', 'c1', 'c2'])
    for key in tqdm(qk):
        here = temp[temp['city_quadkey'] == key]
        X, y = for_reg(here.drop('city_quadkey', axis=1), 2)
        model = LinearRegression()
        model.fit(X, y)
        ans = ans.append({'city_quadkey': key, 'bias': model.intercept_, 'c1': model.coef_[0],
                          'c2': model.coef_[1]}, ignore_index=True)

    ans = ans.join(train_x.groupby('city_quadkey').count()['month'], on='city_quadkey')
    ans = ans.set_index('city_quadkey')

    for key in ans.index:
        if ans.loc[key]['month'] < min_num:
            ans.loc[key, 'bias'] = general_coef['bias']
            ans.loc[key, 'c1'] = general_coef['c1']
            ans.loc[key, 'c2'] = general_coef['c2']

    ans.drop('month', axis=1, inplace=True)
    return ans


def plot_random_quadkey(ans, train_x, train_y):
    plt.figure(figsize=(11, 6))
    key = random.choice(ans.index)
    ind = train_x['city_quadkey'] == key
    plt.scatter(train_x[ind]['month'], train_y[ind])

    plot_model({'bias': ans.loc[key, 'bias'], 'c1': ans.loc[key, 'c1'],
                'c2': ans.loc[key, 'c2']})


def proc(dset, general_coef):
    dset = dset.join(ans, on='city_quadkey')
    dset['days'] = dset['month'].apply(month_to_days)
    dset['c1'].fillna(general_coef['c1'], inplace=True)
    dset['c2'].fillna(general_coef['c2'], inplace=True)
    dset['bias'].fillna(general_coef['bias'], inplace=True)
    pred = dset['days'] * dset['c1'] + dset['days'] * dset['days'] * dset['c2'] + dset['bias']
    dset.drop(['days', 'month', 'city_quadkey', 'c1', 'c2', 'bias'], axis=1, inplace=True)
    dset['line_color'] = dset['line_color'].apply(float)

    return dset, pred


def get_all(data):
    general_coef, general_model = fit_general(data[0][0], data[0][1])
    ans = build_ans(data[0][0], data[0][1], general_coef, 10)

    X_train, reg_train = proc(data[0][0], general_coef)
    X_val, reg_val = proc(data[1][0], general_coef)
    X_test, reg_test = proc(data[2], general_coef)

    return (X_train, data[0][1], reg_train), (X_val, data[1][1], reg_val), (X_test, reg_test)