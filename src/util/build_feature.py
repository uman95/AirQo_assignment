import numpy as np
import pandas as pd
from util.make_dataset import readChannel
from config import cfg


def getFeatures(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    df = df.drop(labels='channel_id', axis=1).groupby(pd.Grouper(
        key='TimeStamp', freq='1H')).mean().fillna(method='ffill')
    df = df.assign(hour=df.index.hour,
                   day=df.index.day,
                   month=df.index.month,
                   day_of_week=df.index.dayofweek,
                   week_of_year=df.index.week)

    return df


def generate_cyclical_features(df: pd.core.frame.DataFrame, col_name: list) -> pd.core.frame.DataFrame:

    for time_col in col_name:
        kwargs = {
            f'sin_{time_col}': lambda x: np.sin(2*np.pi*(x[time_col] - x[time_col].min()) / x[time_col].nunique()),
            f'cos_{time_col}': lambda x: np.cos(2*np.pi*(x[time_col] - x[time_col].min()) / x[time_col].nunique())
        }
        df_time = df.assign(**kwargs)[['sin_'+time_col, 'cos_'+time_col]]

        df = pd.concat([df, df_time], axis=1)

    return df


def oneHotEncoding(df: pd.core.frame.DataFrame, col_name: list) -> pd.core.frame.DataFrame:

    return pd.get_dummies(data=df, columns=col_name, drop_first=True)


def preprocess(df, col_list=['hour', 'day', 'month', 'day_of_week', 'week_of_year']):
    df = readChannel(df)
    df = getFeatures(df)
    df = generate_cyclical_features(df, col_list)
    df = oneHotEncoding(df, col_list)

    return df
