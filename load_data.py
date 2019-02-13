import numpy as np
import pandas as pd
from tqdm import tqdm


def load():

    def get_dictionary(s):
        try:
            d = eval(s)
        except:
            d = {}
        return d

    def get_json_dict(df):
        # global json_cols
        result = dict()
        for e_col in json_cols:
            d = dict()
            rows = df[e_col].values
            for row in rows:
                if row is None: continue
                for i in row:
                    if i['name'] not in d:
                        d[i['name']] = 0
                    d[i['name']] += 1
            result[e_col] = d
        return result

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    test['revenue'] = np.nan

    train = pd.merge(train, pd.read_csv('data/trainRatingTotalVotes.csv'), how='left', on=['imdb_id'])
    test = pd.merge(test, pd.read_csv('data/testRatingTotalVotes.csv'), how='left', on=['imdb_id'])

    trainV2 = pd.read_csv('data/trainV2.csv')

    train = pd.concat([train, trainV2])

    train['revenue'] = np.log1p(train['revenue'])
    y = train['revenue'].values

    json_cols = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast',
                 'crew']

    for col in tqdm(json_cols + ['belongs_to_collection']):
        train[col] = train[col].apply(lambda x: get_dictionary(x))
        test[col] = test[col].apply(lambda x: get_dictionary(x))

    train_dict = get_json_dict(train)
    test_dict = get_json_dict(test)

    for col in json_cols:

        remove = []
        train_id = set(list(train_dict[col].keys()))
        test_id = set(list(test_dict[col].keys()))

        remove += list(train_id - test_id) + list(test_id - train_id)
        for i in train_id.union(test_id) - set(remove):
            if train_dict[col][i] < 10 or i == '':
                remove += [i]

        for i in remove:
            if i in train_dict[col]:
                del train_dict[col][i]
            if i in test_dict[col]:
                del test_dict[col][i]

    return train, test, y, train_dict
