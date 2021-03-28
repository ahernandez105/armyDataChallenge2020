import pandas as pd

def get_skus_df(path):
    df = pd.read_csv(path, header=None, names=['Asset_UID'])
    df['index'] = df.index

    return df

def sort_df_on_skus(df, skus_df):
    out = skus_df.merge(df, how='left', on='Asset_UID')

    return out.sort_values(by='index', axis=0, ascending=True)