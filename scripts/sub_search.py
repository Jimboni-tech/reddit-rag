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

def setup_search_environment(processed_embeddings_path='./datasets/processed_embeddings.parquet',
                             faiss_index_path='./datasets/faiss_index.bin'):
    embed_df = None
    model = None
    index = None

    # Load pre-processed DataFrame 
    try:
        if os.path.exists(processed_embeddings_path):
            embed_df = pd.read_parquet(processed_embeddings_path)
            print(f"Processed DataFrame loaded from '{processed_embeddings_path}'.")
        else:
            print(f"Warning: Processed embeddings file '{processed_embeddings_path}' not found. Please run pre-processing script.")
            return None, None, None
    except Exception as e:
        print(f"An error occurred while loading processed embeddings: {e}")
        return None, None, None

    # Load Sentence Transformer model 
    try:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        print("Sentence Transformer model loaded: all-mpnet-base-v2")
    except Exception as e:
        print(f"Error loading Sentence Transformer model: {e}")
        return None, None, None

    # Load FAISS index
    try:
        if os.path.exists(faiss_index_path):
            index = faiss.read_index(faiss_index_path)
            print(f"FAISS index loaded from '{faiss_index_path}'.")
            print(f"FAISS index populated with {index.ntotal} vectors.")
        else:
            print(f"Warning: FAISS index file '{faiss_index_path}' not found. Please run pre-processing script.")
            return None, None, None
    except Exception as e:
        print(f"An error occurred while loading FAISS index: {e}")
        return None, None, None

    return embed_df, model, index

def find_all_relevant_subreddits(user_query, dataframe, embedding_model, faiss_index, similarity_threshold=0.3):
    query_embedding = embedding_model.encode([user_query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_embedding)

    k_to_search = min(faiss_index.ntotal, 100000)
    distances, indices = faiss_index.search(query_embedding, k_to_search)

    filtered_results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        score = distances[0][i]
        if score >= similarity_threshold:
            filtered_results.append((idx, score))
        else:
            break

    if not filtered_results:
        return pd.DataFrame()

    result_indices = [item[0] for item in filtered_results]
    result_scores = [item[1] for item in filtered_results]

    top_subreddits_df = dataframe.iloc[result_indices].copy()
    top_subreddits_df['similarity_score'] = result_scores
    return top_subreddits_df.sort_values(by='similarity_score', ascending=False)

def main():
    embed_df, model, index = setup_search_environment()

    if embed_df is None:
        print("Setup failed. Exiting.")
    else:
        user_prompt = input("Enter your search query: ")
        try:
            RELEVANCE_THRESHOLD = 0.35
            all_relevant_results = find_all_relevant_subreddits(user_prompt, embed_df, model, index, similarity_threshold=RELEVANCE_THRESHOLD)

            if not all_relevant_results.empty:
                subreddits_output = []
                for i, row in all_relevant_results.iterrows():
                    subreddits_output.append({
                        'name': row['name'],
                        'similarity_score': row['similarity_score']
                    })
                print(f"Found {len(subreddits_output)} relevant subreddits.")
                relevantdf = pd.DataFrame(subreddits_output)
                print(relevantdf)
                return relevantdf
            else:
                print("No relevant subreddits found above the similarity threshold.")
                return None

        except Exception as e:
            print(f"An error occurred during search: {e}")

if __name__ == "__main__":
    main()