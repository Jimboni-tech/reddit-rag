#sentence-transformers/all-mpnet-base-v2
#FAISS
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm 
import re
import numpy as np
import ast

try:
    sub_df = pd.read_csv("./datasets/top-subreddits.csv")
    sub_df['text'] = sub_df['name'] + ' ' + sub_df['public_description'].fillna('')
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(sub_df['text'].tolist(), show_progress_bar=True)

    sub_df['embeddings'] = list(embeddings)
    if 'embeddings' in sub_df.columns:
        sub_df['embeddings'] = sub_df['embeddings'].astype(str).fillna('')

        def parse_embedding_string(s):
            s = s.strip()
            if not s:
                return None
            s = s.strip('[] ').strip()
            if not s:
                return None
            number_strings = re.split(r'\s+', s)
            number_strings = [num_str for num_str in number_strings if num_str]
            if not number_strings:
                return None
            cleaned_s = '[' + ', '.join(number_strings) + ']'
            try:
                return np.array(ast.literal_eval(cleaned_s), dtype=np.float32)
            except (ValueError, SyntaxError) as e:
                print(f"Error evaluating embedding string: '{s}' -> {e}")
                return None

        sub_df['embeddings'] = sub_df['embeddings'].apply(parse_embedding_string)
        sub_df.dropna(subset=['embeddings'], inplace=True)
       
        sub_df.to_csv('./datasets/top-subreddits-embeddings.csv', index=False)

except Exception as e:
    print('Error: ', e)

