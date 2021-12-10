import pandas as pd
import os

dataType = {'channel_id': 'uint32',
            'pm2_5': 'float32',
            'pm10': 'float32',
            's2_pm2_5': 'float32',
            's2_pm10': 'float32',
            'Site': 'str',
            'TimeStamp': 'str'
            }


def load_data(dataPath, dataType: dict) -> pd.core.frame.DataFrame:

    data = pd.read_csv(dataPath, dtype=dataType, index_col=0)
    data = data.drop_duplicates(ignore_index=True)
    data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])

    return data


def create_channel() -> dict:

    data = load_data(
        dataPath='data/airquality-dataset/sample_dataset.csv', dataType=dataType)

    #channelSite = dict(zip(data['channel_id'].unique(), data['Site'].unique()))
    siteGroups = data.groupby('channel_id')
    for x in siteGroups.groups:
        if not os.path.exists('data_group'):
            os.makedirs('data_group')
        siteGroups.get_group(x).to_csv(f'data_group/{x}.csv')

    # return channelSite


def readChannel(channelId: int, dataPath='data/data_group') -> pd.core.frame.DataFrame:

    if not os.path.exists(dataPath):
        create_channel()

    path = dataPath + '/' + str(channelId) + '.csv'
    channelData = pd.read_csv(path, index_col=0, parse_dates=['TimeStamp'])

    return channelData
