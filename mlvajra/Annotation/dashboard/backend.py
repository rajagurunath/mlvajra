from app_layout import app
from app_layout import *
try:
    import base64
    import datetime
    import io
    import pandas as pd
    from sklearn.pipeline import Pipeline
    import pickle
    from lime.lime_text import LimeTextExplainer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
except ImportError as e:
    print("some packages are not installed consider installing the packages",e)

def transform_using_tfidf(text_series):
    tfidf=TfidfVectorizer(stop_words='english')
    array=tfidf.fit_transform(text_series.tolist()).toarray()
    return array,tfidf
    


def similarity_measure(inp_sent,array,tfidf,top_n):
    inp_vec=tfidf.transform([inp_sent]).toarray()
    
    cs=cosine_similarity(inp_vec,array)
    top_match_index=np.flip(np.argsort(cs,axis=1)[:,-top_n:],axis=1)
    return top_match_index
    


def get_similar_records(inp_sent,total_text,top_n=10):
    array,tfidf=transform_using_tfidf(total_text)
    top_match_index=similarity_measure(inp_sent,array,tfidf,top_n)
    return total_text.iloc[top_match_index.ravel()]

def get_simialar_df(sent,df):
    global similar_df
    similar_series=get_similar_records(sent,df[df.columns[0]])
    similar_df=pd.DataFrame(columns=['Similar_sentences','labels'])
    similar_df['Similar_sentences']=similar_series
    
    
    print('check',similar_df.head())
    return similar_df

def return_datable(df):
    #print(df.head())
    #dt.DataTable
    table=dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict("rows"),
    )
    print(table)
    return table
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
    return df
#     html.Div([
#         html.H5(filename),
#         html.H6(datetime.datetime.fromtimestamp(date)),

#         # Use the DataTable prototype component:
#         # github.com/plotly/dash-table-experiments
#         return_datable(df),
#         #dt.DataTable(rows=df.to_dict('records'),id='edit-table'),

#         html.Hr(),  # horizontal line

#         # For debugging, display the raw contents provided by the web browser
# #        html.Div('Raw Content'),
# #        html.Pre(contents[0:200] + '...', style={
# #            'whiteSpace': 'pre-wrap',
# #            'wordBreak': 'break-all'
# #        })
#     ])

