from app_layout import *
from backend import *


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),
               Input('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        global df 
        df=children[0]

        #print(df.head())
        return return_datable(df)

@app.callback(
    Output(component_id='similar-docs', component_property='children'),
    [Input(component_id='user-input-for-similarity', component_property='value')])
def get_similar_docs(sent):
    print('similar docs called ',sent)
#    path=glob.glob(r'*.csv')
#    df=pd.read_csv(path[0],encoding='ISO-8859-1')
    similar_df=get_simialar_df(sent,df)
#    similar_df.to_csv
    return html.Div(dt.DataTable(rows=similar_df.to_dict('records'),id='edit-table-similar'),)

@app.callback(
    Output(component_id='output', component_property='children'),
    [Input(component_id='save-dataset', component_property='n_clicks')])
def save_dataset(n_clicks):
    filename="annotated.csv"
    if n_clicks!=None:
        similar_df.to_csv(DIRECTORY_PATH+'{}'.format(filename))

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})


if __name__ == '__main__':
    app.run_server(debug=True,port =8050)