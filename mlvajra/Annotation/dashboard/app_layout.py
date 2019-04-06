import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import dash_table
import  plotly.graph_objs as go
from flask import Flask
import flask
import os
server = Flask(__name__)

app=dash.Dash(name = __name__, server = server)
if not os.path.exists('tmp'):
    os.mkdir('tmp')
DIRECTORY_PATH=r'tmp\\'

app.scripts.config.serve_locally = True

app.config['suppress_callback_exceptions']=True

def return_datable(df):
    table=dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict("rows"),
    )
    return table

app.layout = html.Div([
#    logo,
    html.H2("Annotator"),
    html.H4("converting unsupervised data into supervised dataset using Annotaion (weekly supervsion)"),
    dcc.Tab(),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',   
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
    html.Div(dt.DataTable(rows=[{}]), style={'display': 'none'}),
    dcc.Input(id='user-input-for-similarity', 
                                       value='Enter the sentence', type='text',
                                       style={'width': '49%','align':'center'}),
    
    html.Div(id='similar-docs'),
    html.Br(),
    html.H3('Training dataset'),
    html.Div(id='output'),
#     html.Button('Train',id='train-button'),
#     html.Br(),
#     dcc.Input(id='user-input-for-prediction', 
#                                        value='Enter the sentence to predict', type='text',
#                                        style={'width': '49%','align':'center'}),
#     html.H1(id='train-output'),
# #    html.Button('Del data',id='delete-button'),
#     html.H1(id='del-output'),
#     dcc.Graph(id='predict-output'),
#     html.Br(),
#     dcc.Link('Why ML made this Prediction !!!!', href='/explain'),
             
    
#    html.Div([
#        html.Pre(id='output', className='two columns'),
#        html.Div(
#            dcc.Graph(
#                id='graph',
#                style={
#                    'overflow-x': 'wordwrap'
#                }
#            ),
#            className='ten columns'
#        )
#    ], className='row')
])
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})


if __name__ == '__main__':
    app.run_server(debug=True,port =8050)