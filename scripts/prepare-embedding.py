import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ast
import re
import os

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

def generate_and_save_artifacts(embeddings_csv_path='./datasets/top-subreddits-embeddings.csv',
                                 processed_embeddings_path='./datasets/processed_embeddings.parquet',
                                 faiss_index_path='./datasets/faiss_index.bin'):

    try:
        embed_df = pd.read_csv(embeddings_csv_path)
        embed_df['embeddings'] = embed_df['embeddings'].apply(parse_embedding_string)
        embed_df.dropna(subset=['embeddings'], inplace=True)
        print("Embeddings DataFrame loaded and parsed from CSV.")
    except FileNotFoundError:
        print(f"Error: '{embeddings_csv_path}' not found. Please ensure it exists.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during CSV loading and parsing: {e}")
        return

    valid_embeddings = embed_df['embeddings'].tolist()
    if not valid_embeddings:
        print("Error: No valid embeddings found after parsing. Exiting artifact generation.")
        return

    corpus_embeddings = np.vstack(valid_embeddings)
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    dimension = corpus_embeddings.shape[1]
    df_to_save = embed_df[['name']].copy()

    try:
        df_to_save.to_parquet(processed_embeddings_path, index=False)
        print(f"Processed DataFrame saved to '{processed_embeddings_path}'.")
    except Exception as e:
        print(f"Error saving processed DataFrame to Parquet: {e}")


    faiss.normalize_L2(corpus_embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(corpus_embeddings)

    try:
        faiss.write_index(index, faiss_index_path)
        print(f"FAISS index saved to '{faiss_index_path}'.")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

    print("Artifact generation complete.")

if __name__ == "__main__":
    os.makedirs('./datasets', exist_ok=True)
    generate_and_save_artifacts()