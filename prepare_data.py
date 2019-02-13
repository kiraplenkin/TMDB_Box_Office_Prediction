import gc
import numpy as np
import pandas as pd


def prepare(df, train_dict):
    global json_cols
    # global train_dict
    df['originalBudjet'] = np.log1p(df['budget'])
    df['budget'] = np.log1p(df['budget'])
    # df['rating'] = df['rating'].fillna(1.5)
    # df['totalVotes'] = df['totalVotes'].fillna(6)
    # df['weightRaiting'] = (df['rating'] * df['totalVotes'] + 6.367 * 1000) / (df['totalVotes'] + 1000)

    df[['release_month', 'release_day', 'release_year']] = df['release_date'].str.split('/', expand=True).replace(
        np.nan, 0).astype(int)
    df['release_year'] = df['release_year']  # ?
    df.loc[(df['release_year'] <= 19) & (df['release_year'] < 100), 'release_year'] += 2000
    df.loc[(df['release_year'] > 19) & (df['release_year'] < 100), 'release_year'] += 1900
    releaseDate = pd.to_datetime(df['release_date'])
    df['release_dayofweek'] = releaseDate.dt.dayofweek
    df['release_quarter'] = releaseDate.dt.quarter

    df['_budget_runtime_ratio'] = df['budget'] / df['runtime']
    df['_budget_popularity_ratio'] = df['budget'] / df['popularity']
    df['_budget_year_ratio'] = df['budget'] / (df['release_year'] * df['release_year'])  # ?
    df['_releaseYear_popularity_ratio'] = df['release_year'] / df['popularity']
    df['_releaseYear_popularity_ratio2'] = df['popularity'] / df['release_year']
    # df['_popularity_totalVotes_ratio'] = df['totalVotes'] / df['popularity']
    # df['_rating_popularity_ratio'] = df['rating'] / df['popularity']
    # df['_rating_totalVotes_ratio'] = df['totalVotes'] / df['rating']
    # df['_totalVotes_releaseYear_ratio'] = df['totalVotes'] / df['release_year']
    # df['_budget_rating_ratio'] = df['budget'] / df['rating']
    # df['_runtime_rating_ratio'] = df['runtime'] / df['rating']
    # df['_budget_totalVotes_ratio'] = df['budget'] / df['totalVotes']

    df['has_homepage'] = 0
    df.loc[pd.isnull(df['homepage']), 'has_homepage'] = 1
    df['isBelongs_to_collectionNA'] = 0
    df.loc[pd.isnull(df['belongs_to_collection']), 'isBelongs_to_collectionNA'] = 1
    # df['isTaglineNA'] = 0
    # df.loc[df['tagline'] == 0, "isTaglineNA"] = 1
    df['isOriginalLanguageEng'] = 0
    df.loc[df['original_language'] == 'en', 'isOriginalLanguageEng'] = 1
    df['isTitleDifferent'] = 1
    df.loc[df['original_title'] == df['title'], 'isTitleDifferent'] = 0
    df['isMovieReleased'] = 1
    df.loc[df['status'] != 'Released', 'isMovieReleased'] = 0

    df['collection_id'] = df['belongs_to_collection'].apply(lambda x: np.nan if len(x) == 0 else x[0]['id'])
    df['original_title_letter_count'] = df['original_title'].str.len()
    df['original_title_word_count'] = df['original_title'].str.split().str.len()
    df['title_word_count'] = df['title'].str.split().str.len()
    df['overview_word_count'] = df['overview'].str.split().str.len()
    # df['tagline_word_count'] = df['tagline'].str.split().str.len()
    df['production_countries_count'] = df['production_countries'].apply(lambda x: len(x))
    df['production_companies_count'] = df['production_companies'].apply(lambda x: len(x))
    df['cast_count'] = df['cast'].apply(lambda x: len(x))
    df['crew_count'] = df['crew'].apply(lambda x: len(x))

    df['meanRuntimeByYear'] = df.groupby('release_year')['runtime'].aggregate('mean')
    df['meanPopularityByYear'] = df.groupby('release_year')['popularity'].aggregate('mean')
    df['meanBudgetByYear'] = df.groupby('release_year')['budget'].aggregate('mean')
    # df['meanTotalVotesByYear'] = df.groupby('release_year')['totalVotes'].aggregate('mean')
    # df['meanTotalVotesByRating'] = df.gropupby('rating')['totalVotes'].aggregate('mean')

    for col in ['genres', 'production_countries', 'spoken_languages', 'production_companies']:
        df[col] = df[col].map(lambda x: sorted(list(set([n if n in train_dict[col] else col + '_etc' for n in [d['name']
                                                                                                               for d in
                                                                                                               x]])))).map(
            lambda x: ','.join(map(str, x)))
        temp = df[col].str.get_dummies(sep=',')
        df = pd.concat([df, temp], axis=1)
    df.drop(['genres_etc'], axis=1, inplace=True)

    df = df.drop(['id', 'revenue', 'belongs_to_collection', 'genres', 'homepage', 'imdb_id', 'overview', 'runtime',
                  'poster_path', 'production_companies', 'production_countries', 'release_date', 'spoken_languages',
                  'status', 'title', 'Keywords', 'cast', 'crew', 'original_language', 'original_title', 'tagline',
                  'collection_id'], axis=1)

    df.fillna(value=0.0, inplace=True)

    gc.collect()

    return df
