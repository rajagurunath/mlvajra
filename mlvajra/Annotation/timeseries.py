try :
    import dash
    from dash.dependencies import Output, Event,Input
    import dash_core_components as dcc
    import dash_html_components as html
    import plotly
    from plotly import tools
    import random
    import plotly.graph_objs as go
    from collections import deque
    import json
    import pandas as pd
    import csv
    #from datetime import datetime
    import numpy as np
    from datetime import timedelta
    from datetime import datetime as dt
except ImportError as e:
    print("some packages are not installed ",e)
#class DataGenerator:
#    def __init__(self, df,cont_names=['BerPreFecMax','PhaseCorrectionAve','PmdMin','Qmin','SoPmdAve']):
#        self.df = df
#        self.cont_names=cont_names
#
#    def __iter__(self):
#        self.n=0
#        return self
#
#    def __next__(self):
#        if self.n > self.df.shape[0]:
#            raise StopIteration
#
#        
#        return self.df.loc[self.n,'@timestamp'],self.df.loc[self.n,self.cont_names].tolist()


#def DataGenerator(meric_list):
#    for i in data[['@timestamp']+meric_list].iterrows():
#        yield i[1]
#        


##raw_data=pd.read_feather(r'D:\windstream_official\Anomaly_detection\data\processed\raw_df_feather')
#
#get_data1=((x,y) for x,y in zip(data['@timestamp'],data['BerPreFecMax']))
#get_data2=((x,y) for x,y in zip(data['@timestamp'],data['PhaseCorrectionAve']))
#get_data3=((x,y) for x,y in zip(data['@timestamp'],data['PmdMin']))
#get_data4=((x,y) for x,y in zip(data['@timestamp'],data['Qmin']))
#get_data5=((x,y) for x,y in zip(data['@timestamp'],data['SoPmdAve']))
#
#
#get_data1=iter(get_data1)
#get_data2=iter(get_data2)
#get_data3=iter(get_data3)
#get_data4=iter(get_data4)
#get_data5=iter(get_data5)

ochctp=pd.read_csv(r'D:\windstream_official\h_OCHCTP.csv')
metric=['BerPreFecMax','PhaseCorrectionAve','PmdMin','Qmin','SoPmdAve']
ochctp.columns=['@timestamp','device']+metric
ochctp['@timestamp']=pd.to_datetime(ochctp['@timestamp'])
ochctp=ochctp.sort_values(by='@timestamp')
ochctp.set_index(ochctp['@timestamp'],inplace=True)
ochctp['date']=(ochctp['@timestamp'].dt.date).astype('str')

device_list=ochctp['device'].unique()
dropoptions=[{'value':dev,'label':dev} for dev in device_list ]

global df 
df=pd.DataFrame(columns=metric)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

stop_n_click=0
app = dash.Dash(__name__)
app.config['suppress_callback_exceptions']=True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
app.layout = html.Div([
        html.Div(html.H2('Time series Annotator Tool')),
        html.Div([html.Div(dcc.DatePickerRange(id='date-picker-range',start_date=ochctp.index[-10].date(),end_date=ochctp.index[-1].date())),
                  html.Div(id='date-picker-output'),]),
        html.Div(dcc.Dropdown(id='device-list',options=dropoptions,className='two column',value='11-L1-9'),className='row'),
        html.Button('Show',id='Show-button'),
         dcc.Dropdown(id='anomly-or-not',
            options=[
                {'label': 'Anomaly', 'value': 'anomaly'},
                {'label': 'Not Anomaly', 'value': 'not_anomaly'},
                
            ],
            value='not_anomaly'
        ),
        html.Div(id='m1-div',children=dcc.Graph(id='m1',)),
        html.Div(id='m2-div',children=dcc.Graph(id='m2',)),
        html.Div(id='m3-div',children=dcc.Graph(id='m3',)),
        html.Div(id='m4-div',children=dcc.Graph(id='m4',)),
        html.Div(id='m5-div',children=dcc.Graph(id='m5',)),
        html.Div([html.Div([dcc.Graph(id='m1', animate=True,
                  style={"position": "relative", "width": "800px", "height": "800px"},className='five column'),
        dcc.Graph(id='m2', animate=True,
                  style={"position": "relative", "width": "800px", "height": "800px"},className='five column')],className='row'),
        html.Div([dcc.Graph(id='m3', animate=True,
                  style={"position": "relative", "width": "800px", "height": "800px"},className='five column'),
        dcc.Graph(id='m4', animate=True,
                  style={"position": "relative", "width": "800px", "height": "800px"},className='five column')],className='row'),
        dcc.Graph(id='m5', animate=True,
                  style={"position": "relative", "width": "800px", "height": "800px"},className='five column'),],className='column'),
                  
##        dcc.Interval(
#            id='graph-update1',
#            interval=1*1000
#        ),
#        dcc.Interval(
#            id='graph-update2',
#            interval=1*1000
#        ),
#        dcc.Interval(
#            id='graph-update3',
#            interval=1*1000
#        ),
#        dcc.Interval(
#            id='graph-update4',
#            interval=1*1000
#        ),
#        dcc.Interval(
#            id='graph-update',
#            interval=1*1000
#        ),
#        html.Button('Stop', id='Stop-button'),

       


                
    ]
)

#def line_plot(X,Y,name=None,anom=None):
#    trace = plotly.graph_objs.Scatter(
#                    x=list(X),
#                    y=list(Y1),
#                    name='Scatter',
#                    mode= 'lines+markers'
#                    )
#    if anom:
#        return {'data': [trace,anom],}
#    
#    print('range',min(X)-timedelta(hours=15),max(X)+timedelta(hours=15))    
#    return {'data': [trace],'layout' : go.Layout(xaxis=dict(range=[min(X)-timedelta(hours=15),max(X)+timedelta(hours=15)]),
#                                                        yaxis=dict(range=[range_dict[name]['min'],range_dict[name]['max']]),)}
#    
##    {'data': [trace],'layout' : go.Layout(xaxis=dict(range=[min(X)-timedelta(hours=1),max(X)+timedelta(hours=1)]),
##                                                        yaxis=dict(range=[range_dict[name]['min'],range_dict[name]['max']])}
    
#def get_graphs(X1,Y1,Y2,Y3,Y4,Y5,anom):
#    trace1 = go.Scatter(x=X1, y=Y1)
#    trace2 = go.Scatter(x=X1, y=Y2)
#    trace3 = go.Scatter(x=X1, y=Y3)
#    trace4 = go.Scatter(x=X1, y=Y4)
#    trace4 = go.Scatter(x=X1, y=Y5)
#    
#    fig = tools.make_subplots(rows=3, cols=2, subplot_titles=('Plot 1', 'Plot 2',
#                                                              'Plot 3', 'Plot 4'))
#    fig.append_trace([trace1,anom], 1, 1)
#    fig.append_trace(trace2, 1, 2)
#    fig.append_trace(trace3, 2, 1)
#    fig.append_trace(trace4, 2, 2)
#    fig.append_trace(trace4, 3, 1)
#
#    fig['layout']['xaxis1'].update(title='xaxis 1 title')
#    fig['layout']['xaxis2'].update(title='xaxis 2 title', )
#    fig['layout']['xaxis3'].update(title='xaxis 3 title', showgrid=False)
#    fig['layout']['xaxis4'].update(title='xaxis 4 title', )
#    
#    fig['layout']['yaxis1'].update(title='yaxis 1 title')
#    fig['layout']['yaxis2'].update(title='yaxis 2 title', )
#    fig['layout']['yaxis3'].update(title='yaxis 3 title', showgrid=False)
#    fig['layout']['yaxis4'].update(title='yaxis 4 title')
#    
#    fig['layout'].update(title='Customizing Subplot Axes')
#    return fig    
    
#fields=['Type','X','Y','Anom_point']
#with open(r'anomaly1.csv', 'a') as f:
#    writer = csv.writer(f)
#    writer.writerow(fields)
##@app.callback(
##    Output('selected-data', 'children'),
##    [Input('basic-interactions', 'selectedData')])
##def display_selected_data(selectedData):
##    return json.dumps(selectedData, indent=2)
#def save_data(anom,X1,Y1,x):
#    with open(r'anomaly1.csv', 'a') as f:
#        writer = csv.writer(f)
#        
#        writer.writerow([f"{anom,X1,Y1,x}"])
#
#q_to_list=lambda x:list(x)
#

#d1,d2='2018-02-07' ,'2018-02-16'
#
#start_date = dt.strptime(d1, '%Y-%m-%d')
#end_date = dt.strptime(d1, '%Y-%m-%d')
#start_date_string = start_date.strftime('%B %d, %Y')
        
        
@app.callback(Output('date-picker-output', 'children'),
              [Input('date-picker-range', 'start_date'),
               Input('date-picker-range', 'end_date')])

def update_df(start_date,end_date):
    global df
    df=ochctp[(ochctp['date']>start_date) & (ochctp['date']<=end_date)]
#    df=ochctp[ochctp["date"].isin(pd.date_range(start_date,end_date))]
    print('updated',df.shape)
    
    return html.H2('Date Range  from {} to {}'.format(start_date,end_date))



@app.callback(Output('m1-div', 'children'),
              [Input('m1','clickData'),
               Input('anomly-or-not','value'),
               Input('device-list','value'),
               Input('Show-button', 'n_clicks'),
              ])

def update_plot1(clickData,anom_or_not,device,n_clicks,):
        if n_clicks!=None:
    #        print('clicks',n_clicks)
#            print(ochctp.shape)
    #        if True:
    #            if (1+n_clicks)%2==0:
#            print('exec')
            global df
#            print(start_date,end_date)
    #        print(pd.date_range(start_date,end_date))
    #        df=ochctp[(ochctp['date']>start_date) & (ochctp['date']<=end_date)]
    #        df=ochctp[ochctp["date"].isin(pd.date_range(start_date,end_date))]
            print(df.shape)
            print('clickdata',clickData)
            df=df[df['device']==device]
            print(df.shape)
#            print(start_date,end_date)
            metric_series=df[metric[0]]  
            if clickData!=None:
        #    print(json.dumps(clickData, indent=2),anom)
                Xa=[i['x'] for i in clickData['points']]
                Ya=[i['y'] for i in clickData['points']]
                anom=plotly.graph_objs.Scatter(
                        x=Xa,
                        y=Ya,
                        name='Anomaly',
                        mode='markers',
                        marker={'size': 12}
                        )
                trace = plotly.graph_objs.Scatter(
                        x=metric_series.index,
                        y=metric_series.values,
                        name=name1,
                        mode= 'lines+markers'
                        )
                            
                return dcc.Graph(id='m1',figure={'data': [trace,anom],'layout' : go.Layout(title=name1,xaxis=dict(range=[min(metric_series.index)-timedelta(hours=5),
                                                                        max(metric_series.index)+timedelta(hours=5)]),
                                                                    yaxis=dict(range=[min(metric_series.values),max(metric_series.values)]))})
            else :
                trace = plotly.graph_objs.Scatter(
                        x=metric_series.index,
                        y=metric_series.values,
                        name=name2,
                        mode= 'lines+markers'
                        )
    #                print('guru')
    #    #            if anom:
    #    #                return {'data': [trace,anom],}
    #    #            name='SoPmdAve'
    #                print('range',min(X5)-timedelta(hours=15),max(X)+timedelta(hours=15))    
    #                print('yrange',min(Y5),max(Y5),min(list(Y5)[-10:]),max(list(Y5)[-10:]))    
    #                print('len',len(Y5))
                return dcc.Graph(id='m1',figure={'data': [trace],'layout' : go.Layout(title=name1,xaxis=dict(range=[min(metric_series.index)-timedelta(hours=5),
                                                                        max(metric_series.index)+timedelta(hours=5)]),
                                                                    yaxis=dict(range=[min(metric_series.values),max(metric_series.values)]))})
                
    

#@app.callback(Output('m1-div', 'children'),
#              [Input('m1','clickData'),
#               Input('anomly-or-not','value'),
#               Input('device-list','value'),
#               Input('date-picker-range', 'start_date'),
#               Input('date-picker-range', 'end_date')])
#
#def update_plot1(clickData,anom_or_not,device,start_date,end_date):
#        df=ochctp[(ochctp['date'] > start_date) & (ochctp['date'] <= end_date)]
#
##        df=ochctp[ochctp["date"].isin(pd.date_range(start_date,end_date))]
#        print('clickdata',clickData)
#        df=df[df['device']==device]
#        print(df.shape)
#        print(start_date,end_date)
#        metric_series=df[metric[0]]  
#        if clickData!=None:
#    #    print(json.dumps(clickData, indent=2),anom)
#            Xa=[i['x'] for i in clickData['points']]
#            Ya=[i['y'] for i in clickData['points']]
#            anom=plotly.graph_objs.Scatter(
#                    x=Xa,
#                    y=Ya,
#                    name='Anomaly',
#                    mode='markers',
#                    marker={'size': 12}
#                    )
#            trace = plotly.graph_objs.Scatter(
#                    x=metric_series.index,
#                    y=metric_series.values,
#                    name=name1,
#                    mode= 'lines+markers'
#                    )
#                        
#            return dcc.Graph(id='m1',figure={'data': [trace,anom],'layout' : go.Layout(title=name1,xaxis=dict(range=[min(metric_series.index)-timedelta(hours=5),
#                                                                    max(metric_series.index)+timedelta(hours=5)]),
#                                                                yaxis=dict(range=[min(metric_series.values),max(metric_series.values)]))})
#        else :
#            trace = plotly.graph_objs.Scatter(
#                    x=metric_series.index,
#                    y=metric_series.values,
#                    name=name2,
#                    mode= 'lines+markers'
#                    )
##                print('guru')
##    #            if anom:
##    #                return {'data': [trace,anom],}
##    #            name='SoPmdAve'
##                print('range',min(X5)-timedelta(hours=15),max(X)+timedelta(hours=15))    
##                print('yrange',min(Y5),max(Y5),min(list(Y5)[-10:]),max(list(Y5)[-10:]))    
##                print('len',len(Y5))
#            return dcc.Graph(id='m1',figure={'data': [trace],'layout' : go.Layout(title=name1,xaxis=dict(range=[min(metric_series.index)-timedelta(hours=5),
#                                                                    max(metric_series.index)+timedelta(hours=5)]),
#                                                                yaxis=dict(range=[min(metric_series.values),max(metric_series.values)]))})
#            
@app.callback(Output('m2-div', 'children'),
              [Input('m2','clickData'),
               Input('anomly-or-not','value'),
               Input('device-list','value'),
               Input('Show-button', 'n_clicks')])

def update_plot2(clickData,anom_or_not,device,n_clicks):
         if n_clicks!=None:
            global df
            df=df[df['device']==device]
            print(df.shape)
#            print(start_date,end_date)
            metric_series=df[metric[1]]  
            if clickData!=None:
        #    print(json.dumps(clickData, indent=2),anom)
                Xa=[i['x'] for i in clickData['points']]
                Ya=[i['y'] for i in clickData['points']]
                anom=plotly.graph_objs.Scatter(
                        x=Xa,
                        y=Ya,
                        name='Anomaly',
                        mode='markers',
                        marker={'size': 12}
                        )
                trace = plotly.graph_objs.Scatter(
                        x=metric_series.index,
                        y=metric_series.values,
                        name=name1,
                        mode= 'lines+markers'
                        )
                            
                return dcc.Graph(id='m2',figure={'data': [trace,anom],'layout' : go.Layout(title=name1,xaxis=dict(range=[min(metric_series.index)-timedelta(hours=5),
                                                                        max(metric_series.index)+timedelta(hours=5)]),
                                                                    yaxis=dict(range=[min(metric_series.values),max(metric_series.values)]))})
            else :
                trace = plotly.graph_objs.Scatter(
                        x=metric_series.index,
                        y=metric_series.values,
                        name=name2,
                        mode= 'lines+markers'
                        )
    #                print('guru')
    #    #            if anom:
    #    #                return {'data': [trace,anom],}
    #    #            name='SoPmdAve'
    #                print('range',min(X5)-timedelta(hours=15),max(X)+timedelta(hours=15))    
    #                print('yrange',min(Y5),max(Y5),min(list(Y5)[-10:]),max(list(Y5)[-10:]))    
    #                print('len',len(Y5))
                return dcc.Graph(id='m2',figure={'data': [trace],'layout' : go.Layout(title=name1,xaxis=dict(range=[min(metric_series.index)-timedelta(hours=5),
                                                                        max(metric_series.index)+timedelta(hours=5)]),
                                                                    yaxis=dict(range=[min(metric_series.values),max(metric_series.values)]))})
            
@app.callback(Output('m3-div', 'children'),
              [Input('m3','clickData'),
               Input('anomly-or-not','value'),
               Input('device-list','value'),
               Input('Show-button', 'n_clicks')])

def update_plot3(clickData,anom_or_not,device,n_clicks):
         if n_clicks!=None:
#            if (1+n_clicks)%2==0:
#            df=ochctp[(ochctp['date'] > start_date) & (ochctp['date'] <= end_date)]
            global df
    #        df=ochctp[ochctp["date"].isin(pd.date_range(start_date,end_date))]
            print('clickdata',clickData)
            df=df[df['device']==device]
            print(df.shape)
#            print(start_date,end_date)
            metric_series=df[metric[2]]  
            if clickData!=None:
        #    print(json.dumps(clickData, indent=2),anom)
                Xa=[i['x'] for i in clickData['points']]
                Ya=[i['y'] for i in clickData['points']]
                anom=plotly.graph_objs.Scatter(
                        x=Xa,
                        y=Ya,
                        name='Anomaly',
                        mode='markers',
                        marker={'size': 12}
                        )
                trace = plotly.graph_objs.Scatter(
                        x=metric_series.index,
                        y=metric_series.values,
                        name=name1,
                        mode= 'lines+markers'
                        )
                            
                return dcc.Graph(id='m3',figure={'data': [trace,anom],'layout' : go.Layout(title=name1,xaxis=dict(range=[min(metric_series.index)-timedelta(hours=5),
                                                                        max(metric_series.index)+timedelta(hours=5)]),
                                                                    yaxis=dict(range=[min(metric_series.values),max(metric_series.values)]))})
            else :
                trace = plotly.graph_objs.Scatter(
                        x=metric_series.index,
                        y=metric_series.values,
                        name=name2,
                        mode= 'lines+markers'
                        )
    #                print('guru')
    #    #            if anom:
    #    #                return {'data': [trace,anom],}
    #    #            name='SoPmdAve'
    #                print('range',min(X5)-timedelta(hours=15),max(X)+timedelta(hours=15))    
    #                print('yrange',min(Y5),max(Y5),min(list(Y5)[-10:]),max(list(Y5)[-10:]))    
    #                print('len',len(Y5))
                return dcc.Graph(id='m3',figure={'data': [trace],'layout' : go.Layout(title=name1,xaxis=dict(range=[min(metric_series.index)-timedelta(hours=5),
                                                                        max(metric_series.index)+timedelta(hours=5)]),
                                                                    yaxis=dict(range=[min(metric_series.values),max(metric_series.values)]))})
            
@app.callback(Output('m4-div', 'children'),
              [Input('m4','clickData'),
               Input('anomly-or-not','value'),
               Input('device-list','value'),
               Input('Show-button', 'n_clicks')])

def update_plot4(clickData,anom_or_not,device,n_clicks):
         if n_clicks!=None:
#            if (1+n_clicks)%2==0:
#            df=ochctp[(ochctp['date'] > start_date) & (ochctp['date'] <= end_date)]
            global df
    #        df=ochctp[ochctp["date"].isin(pd.date_range(start_date,end_date))]
            print('clickdata',clickData)
            df=df[df['device']==device]
            print(df.shape)
#            print(start_date,end_date)
            metric_series=df[metric[3]]  
            if clickData!=None:
        #    print(json.dumps(clickData, indent=2),anom)
                Xa=[i['x'] for i in clickData['points']]
                Ya=[i['y'] for i in clickData['points']]
                anom=plotly.graph_objs.Scatter(
                        x=Xa,
                        y=Ya,
                        name='Anomaly',
                        mode='markers',
                        marker={'size': 12}
                        )
                trace = plotly.graph_objs.Scatter(
                        x=metric_series.index,
                        y=metric_series.values,
                        name=name1,
                        mode= 'lines+markers'
                        )
                            
                return dcc.Graph(id='m4',figure={'data': [trace,anom],'layout' : go.Layout(title=name1,xaxis=dict(range=[min(metric_series.index)-timedelta(hours=5),
                                                                        max(metric_series.index)+timedelta(hours=5)]),
                                                                    yaxis=dict(range=[min(metric_series.values),max(metric_series.values)]))})
            else :
                trace = plotly.graph_objs.Scatter(
                        x=metric_series.index,
                        y=metric_series.values,
                        name=name2,
                        mode= 'lines+markers'
                        )
    #                print('guru')
    #    #            if anom:
    #    #                return {'data': [trace,anom],}
    #    #            name='SoPmdAve'
    #                print('range',min(X5)-timedelta(hours=15),max(X)+timedelta(hours=15))    
    #                print('yrange',min(Y5),max(Y5),min(list(Y5)[-10:]),max(list(Y5)[-10:]))    
    #                print('len',len(Y5))
                return dcc.Graph(id='m4',figure={'data': [trace],'layout' : go.Layout(title=name1,xaxis=dict(range=[min(metric_series.index)-timedelta(hours=5),
                                                                        max(metric_series.index)+timedelta(hours=5)]),
                                                                    yaxis=dict(range=[min(metric_series.values),max(metric_series.values)]))})
                
@app.callback(Output('m5-div', 'children'),
              [Input('m5','clickData'),
               Input('anomly-or-not','value'),
               Input('device-list','value'),
               Input('Show-button', 'n_clicks')])

def update_plot5(clickData,anom_or_not,device,n_clicks):
         if n_clicks!=None:
#            if (1+n_clicks)%2==0:
#            df=ochctp[(ochctp['date'] > start_date) & (ochctp['date'] <= end_date)]
            global df
    #        df=ochctp[ochctp["date"].isin(pd.date_range(start_date,end_date))]
            print('clickdata',clickData)
            df=df[df['device']==device]
            print(df.shape)
#            print(start_date,end_date)
            metric_series=df[metric[4]]  
            if clickData!=None:
        #    print(json.dumps(clickData, indent=2),anom)
                Xa=[i['x'] for i in clickData['points']]
                Ya=[i['y'] for i in clickData['points']]
                anom=plotly.graph_objs.Scatter(
                        x=Xa,
                        y=Ya,
                        name='Anomaly',
                        mode='markers',
                        marker={'size': 12}
                        )
                trace = plotly.graph_objs.Scatter(
                        x=metric_series.index,
                        y=metric_series.values,
                        name=name1,
                        mode= 'lines+markers'
                        )
                            
                return dcc.Graph(id='m5',figure={'data': [trace,anom],'layout' : go.Layout(title=name1,xaxis=dict(range=[min(metric_series.index)-timedelta(hours=5),
                                                                        max(metric_series.index)+timedelta(hours=5)]),
                                                                    yaxis=dict(range=[min(metric_series.values),max(metric_series.values)]))})
            else :
                trace = plotly.graph_objs.Scatter(
                        x=metric_series.index,
                        y=metric_series.values,
                        name=name2,
                        mode= 'lines+markers'
                        )
    #                print('guru')
    #    #            if anom:
    #    #                return {'data': [trace,anom],}
    #    #            name='SoPmdAve'
    #                print('range',min(X5)-timedelta(hours=15),max(X)+timedelta(hours=15))    
    #                print('yrange',min(Y5),max(Y5),min(list(Y5)[-10:]),max(list(Y5)[-10:]))    
    #                print('len',len(Y5))
                return dcc.Graph(id='m5',figure={'data': [trace],'layout' : go.Layout(title=name1,xaxis=dict(range=[min(metric_series.index)-timedelta(hours=5),
                                                                        max(metric_series.index)+timedelta(hours=5)]),
                                                                    yaxis=dict(range=[min(metric_series.values),max(metric_series.values)]))})
                
    

app.css.append_css({
    'external_url': ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                     'https://cdnjs.cloudflare.com/ajax/libs/vis/4.20.1/vis.min.css',]
})
if __name__ == '__main__':
#    data=pd.read_feather(r'D:\windstream_official\Anomaly_detection\data\processed\13L110')
    name1,name2,name3,name4,name5=metric

    app.run_server(debug=True)





















