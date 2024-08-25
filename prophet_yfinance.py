#Adding libraries

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
np.float_ = np.float64
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import date

tickers = {
    'NVDA.MX': 'NVIDIA',
    'VNQ.MX': 'Vanguard Real Estate',
    'VEA.MX': 'Vanguard Developed Markets',
    'VOO.MX': 'Vanguard S&P 500',
    'VWO.MX': 'Vanguard Emerging Markets',
    'MCHI.MX': 'iShares China',
    'IVVPESOISHRS.MX': 'iShares S&P 500',
    'VTIP.MX': 'Vanguard Short-Term Inflation-Protected Securities',
    'TSLA.MX': 'Tesla',
    'VT.MX': 'Vanguard Total World Stock',
    'VGT.MX': 'Vanguard Information Technology',
    'BIMBOA.MX': 'Grupo Bimbo',
    'NOKN.MX': 'Nokia Oyj',
    'DAL.MX': 'Delta Air Lines',
    'HERO.MX': 'Global X Video Games & Esports'
}

tickers_list = list(tickers.keys())

def upload_data(ticker, start_date, end_date):
    df = yf.Ticker(ticker).history(start=start_date.strftime('%Y-%m-%d'),
                                   end=end_date.strftime('%Y-%m-%d')
        )
    return df

def predict_data(df, months_predict):
    df.reset_index(inplace=True)
    df = df.loc[:, ['Date', 'Close']]
    df['Date'] = df['Date'].dt.tz_localize(None)
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

    model = Prophet()
    model.fit(df)

    future_dates = model.make_future_dataframe(periods=int(months_predict) * 30)
    predictions = model.predict(future_dates)

    return model, predictions

st.set_page_config(
    page_title='Prophet Yahoo Finance',
    page_icon='📈'
)

#st.image('logo.png')
st.markdown("""
# Predictive Analysis App
This app leverages [Prophet](https://facebook.github.io/prophet/)
            and [yfinance](https://pypi.org/project/yfinance/)
            to forecast the value of stocks and ETFs from [Yahoo Finance](https://finance.yahoo.com/),
            converting them into Mexican Pesos.
""")

with st.sidebar:
    st.header('Menu')
    ticker = st.selectbox('Choose a ticker:', tickers_list)
    start_date = st.date_input('Start date:', value=date(2022, 1, 1))
    end_date = st.date_input('End date:')
    months_prediction = st.number_input('Number of months to predict:', 1, 24, value=6)

st.header('Summary Table from last date')

df_summary = pd.DataFrame(columns=['Ticker', 'Close'])

for ticker_key in tickers_list:
    df = upload_data(ticker_key, start_date, end_date)
    df = df.sort_index(ascending=False)
    df_temp = df[['Close']].iloc[0:1]
    df_temp['Ticker'] = ticker_key
    df_temp = df_temp[['Ticker', 'Close']]
    df_summary = pd.concat([df_summary, df_temp], ignore_index=True)
    
st.dataframe(df_summary)

df = upload_data(ticker, start_date, end_date)
df = df.sort_index(ascending=False)

if df.shape[0] != 0:

    st.header(f'Data from {tickers[ticker]} ({ticker})')
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    ))

    fig.update_layout(
    xaxis_title='Period',
    yaxis_title='Price',
    #title={'text': 'Candlestick Chart', 'x': 0.5},
    #xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig)

    show_data = st.toggle('Show data')

    if show_data:
        st.dataframe(df[['Open', 'Close', 'Volume']])

    st.subheader(f'Prediction for next {months_prediction} months')
    model, prediction = predict_data(df, months_prediction)
    fig = plot_plotly(model, prediction, xlabel='Period', ylabel='Price')
    st.plotly_chart(fig)

else:
    st.warning('Data not found in the selected period')