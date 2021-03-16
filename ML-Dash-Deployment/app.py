import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import numpy as np
import datetime as dt
import pandas_datareader as web
import plotly.express as px


app = dash.Dash()
server = app.server

start = dt.datetime(2000,1,1)
end = dt.datetime.now()
df = web.DataReader('AAPL','yahoo', start, end)
df=df.reset_index()

df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']

data=df.sort_index(ascending=True,axis=0)
new_data=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["Close"][i]=data["Close"][i]
    
new_data=new_data.set_index('Date')
dataset=new_data.values


tickers = ['TSLA','AAPL','FB','MSFT','SBUX']
df1 = web.DataReader(tickers, data_source='yahoo', start='2017-01-01', end=dt.datetime.now())
df=df1.stack().reset_index().rename(index=str, columns={"level_1": "Symbols"}).sort_values(['Symbols','Date'])
df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']

model_data= pd.read_csv("model_data.csv")

fig = px.line(model_data, x='Date', y=['Train', 'Validation', 'M1', 'M2', 'M3'])

fig.update_layout(hovermode='x unified',
    showlegend=True,
    plot_bgcolor="white",
    xaxis_title="Date",
    yaxis_title="Closing Rate Predicted",
    legend_title="Data:",
    margin=dict(t=50,l=50,b=50,r=50)
)

fig.update_xaxes(showline=True, linewidth=2, linecolor='white', gridcolor='lightgray')
fig.update_yaxes(showline=True, linewidth=2, linecolor='white', gridcolor='lightgray')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

D_validationData= pd.read_csv("D_validationData.csv")
D_train_data= pd.read_csv("D_train_data.csv")


fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=D_validationData["Date"], y=D_validationData["Close"],
                    mode='lines',name='Validation',line=dict(color="blue",width=4)))
fig2.add_trace(go.Scatter(x=D_validationData["Date"], y=D_validationData["Predictions"],
                    mode='lines',name='Predictions',line=dict(color="red",width=4)))
fig2.add_trace(go.Scatter(x=D_train_data["Date"], y=D_train_data["Close"],
                    mode='lines', name='Train',line=dict(color="darkgreen",width=4)))


fig2.update_layout(hovermode='x unified',
    showlegend=True,
    plot_bgcolor="white",
    paper_bgcolor = "rgba(0,0,0,0)",
    xaxis_title="Date",
    yaxis_title="Closing Rate Predicted",
    legend_title="Data:",
    margin=dict(t=50,l=50,b=50,r=50),
    
)

fig2.update_xaxes(showline=True, linewidth=2, linecolor='white', gridcolor='lightgray')
fig2.update_yaxes(showline=True, linewidth=2, linecolor='white', gridcolor='lightgray')


moving_avg= pd.read_csv("test_mov_avg.csv")
moving_avg

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=moving_avg["date"], y=moving_avg["close"],
                    mode='lines',
                    name='Test',
                    line=dict(color="blue",width=4)))
fig3.add_trace(go.Scatter(x=moving_avg["date"], y=moving_avg["est_N2"],
                    mode='lines',
                    name='Predictions using Moving Average',
                    line=dict(color="red",width=4)))

fig3.update_layout(hovermode='x unified',
    showlegend=True,
    plot_bgcolor="white",
    paper_bgcolor = "rgba(0,0,0,0)",
    xaxis_title="Date",
    yaxis_title="Closing Rate Predicted",
    legend_title="Data:",
    margin=dict(t=50,l=50,b=50,r=50),
    
)
fig3.update_xaxes(showline=True, linewidth=1, linecolor='white', gridcolor='lightgray')
fig3.update_yaxes(showline=True, linewidth=1, linecolor='white', gridcolor='lightgray')


app.layout = html.Div([
   
    html.H1("Stock Price Prediction- Machine Learning and Python", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='LSTM',children=[
			html.Div([				
			    html.H2("Long Short-Term Memory (LSTM)", 
                        style={'textAlign': 'center'}),
                dcc.Graph(id = 'GrapLTSM',
                        figure = fig2),
                     ]
                ),   		
        ]),

        dcc.Tab(label='Moving Average',children=[
			html.Div([				
			    html.H2("Moving Average to predict Apple Stock  Price", 
                        style={'textAlign': 'center'}),
                dcc.Graph(id = 'GrapMovingAvg',
                        figure = fig3),
                    ]
                ),
        ]),

        dcc.Tab(label='Stock Data for other Companies', children=[
            html.Div([
                html.H1("Stocks High vs Lows", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'SBUX'},
                                      {'label': 'Starbucks', 'value': 'SBUX'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'}, 
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'SBUX'},
                                      {'label': 'Starbucks', 'value': 'SBUX'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])
    ])
])




@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","SBUX": "Starbucks", "MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Symbols"] == stock]["Date"],
                     y=df[df["Symbols"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Symbols"] == stock]["Date"],
                     y=df[df["Symbols"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","SBUX":"Starbucks","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Symbols"] == stock]["Date"],
                     y=df[df["Symbols"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure



if __name__=='__main__':
	app.run_server(debug=True)