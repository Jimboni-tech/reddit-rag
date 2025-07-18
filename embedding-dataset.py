#sentence-transformers/all-mpnet-base-v2
#FAISS
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm 

try:
    sub_df = pd.read_csv("datasets/top-subreddits.csv")
    sub_df['text'] = sub_df['name'] + ' ' + sub_df['public_description'].fillna('')
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(sub_df['text'].tolist(), show_progress_bar=True)

    sub_df['embeddings'] = list(embeddings)
    sub_df.to_csv('datasets/top-subreddits-embeddings.csv')
except Exception as e:
    print('Error: ', e)