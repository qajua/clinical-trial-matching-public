import pandas as pd

def load_ds(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    text_merged = df.groupby('note_id')['text'].apply(
        lambda x: '\n\n'.join(x.dropna().astype(str).unique())
    ).reset_index()
    
    df = df.drop(columns=['text'])
    df = df.merge(text_merged, on='note_id', how='left')
    
    return df