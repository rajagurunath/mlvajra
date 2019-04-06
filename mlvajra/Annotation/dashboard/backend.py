from app_layout import app,return_datable
import base64
import datetime
import io
import pandas as pd
def parse_contents(contents, file_name, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    global df 
    global filename
    
    try:
        if 'csv' in file_name:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf8')))
        elif 'xls' in file_name:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    """
    pass similar contents df not file uploaded df 
    To be filled
    
    """
#    df.to_csv(r'{}'.format(filename),index=False)
    filename=file_name
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # Use the DataTable prototype component:
        # github.com/plotly/dash-table-experiments
        return_datable(df),
        #dt.DataTable(rows=df.to_dict('records'),id='edit-table'),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
#        html.Div('Raw Content'),
#        html.Pre(contents[0:200] + '...', style={
#            'whiteSpace': 'pre-wrap',
#            'wordBreak': 'break-all'
#        })
    ])

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),
               Input('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        
        return children

