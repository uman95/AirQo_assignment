import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.util.make_dataset import readChannel, create_channel, channelSite

st.set_page_config(page_title='AirQo Data Analysis', layout="wide")

page = st.sidebar.selectbox('Plot Type', ['Timeseries', 'Boxplot'])

_, row_one, _ = st.columns([15, 20, 0.5])

row_one.title("Series plot")

channel = readChannel(channelId=list(channelSite.keys())[
                      0], dataPath='src/data/airquality-dataset/data_group')

channel_copy = channel.set_index(['TimeStamp'])
target = ['pm2_5', 'pm10', 's2_pm2_5', 's2_pm10']

points = 10000

if page == 'Timeseries':
    _, row_two_col2, row_two_col3 = st.columns([1, 5, 10])

    row_two_col2.write("#")
    row_two_col2.header("Data plot by site")
    fig = plt.figure(figsize=(14, 10))
    channel_copy[:points].drop('channel_id', axis=1).plot(subplots=True, layout=(2, 2), figsize=(
        14, 10), sharex=False, rot=45, title='first 10,000 data of PM2.5 for Site Banda, Kampala')
    plt.tight_layout(pad=5)
    st.pyplot(fig=fig)


else:
    fig = plt.figure(figsize=(16, 7))

    ax = sns.boxplot(x=channel_copy.index.year, y='pm2_5',
                     data=channel_copy, orient='v', palette=sns.color_palette("deep", 5))
    ax.set(ylim=(-50, 600))

    plt.title('Boxplots of Hourly PM 2.5 by Year', fontsize=16)
    plt.xlabel('')
    plt.ylabel('ug/m^3', fontsize=12)
    st.pyplot(fig=fig)


hide_streamlit_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
            content:'Made with Streamlit by Usman'; 
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
