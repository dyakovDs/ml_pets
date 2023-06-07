import datetime

import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import dill
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector


def column_dropper(df):
    data = df.copy()
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    return data.drop(columns_to_drop, axis=1)


def outliners_dealer(df):
    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return boundaries

    data = df.copy()
    boundaries = calculate_outliers(data['year'])
    data.loc[data['year'] < boundaries[0], 'year'] = round(boundaries[0])
    data.loc[data['year'] > boundaries[1], 'year'] = round(boundaries[1])
    return data


def shorting(df):
    def short_model(x):
        import pandas as pd
        if not pd.isna(x):
            return x.lower().split(' ')[0]
        else:
            return x

    data = df.copy()
    data.loc[:, 'short_model'] = data['model'].apply(short_model)
    return data


def add_categories(df):
    data = df.copy()
    data.loc[:, 'age_category'] = data['year'].apply(
        lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return data


def main():
    print('Starting price category prediction pipeline...')
    df = pd.read_csv('model/data/df.csv')
    X = df.drop(['price_category'], axis=1)
    y = df['price_category'].apply(lambda x: 0.0 if x == 'low' else (1.0 if x == 'medium' else 2.0))

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('categorical_transformer', categorical_transformer, make_column_selector(dtype_include=object)),
        ('numerical_transformer', numerical_transformer, make_column_selector(dtype_exclude=object))
    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(column_dropper)),
        ('shorter', FunctionTransformer(shorting)),
        ('cat_adder', FunctionTransformer(add_categories)),
        ('outliners', FunctionTransformer(outliners_dealer)),
        ('column_transformer', column_transformer)
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    )

    best_score = 0.0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe
    best_pipe.fit(X, y)
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    filename = 'cars_best_pipe.pkl'
    with open(filename, 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'car price category prediction pipeline',
                'author': 'Dyakov Aleksandr',
                'version': '1.0',
                'date': datetime.datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'accuracy': best_score
            }
        }, file)


if __name__ == '__main__':
    main()
