import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ast
import re

# --- 1. Load Data ---
try:
    embed_df = pd.read_csv('datasets/top-subreddits-embeddings.csv')
    if 'embeddings' in embed_df.columns:
        embed_df['embeddings'] = embed_df['embeddings'].astype(str).fillna('')

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

        embed_df['embeddings'] = embed_df['embeddings'].apply(parse_embedding_string)
        embed_df.dropna(subset=['embeddings'], inplace=True)
    else:
        print("Error: The DataFrame must contain an 'embeddings' column.")
        exit()
except FileNotFoundError:
    print("Error: 'datasets/top-subreddits-embeddings.csv' not found. Please check the path.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

# --- 2. Initialize Embedding Model ---
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print("Sentence Transformer model loaded: all-mpnet-base-v2")

# --- 3. Prepare Embeddings for FAISS ---
valid_embeddings = embed_df['embeddings'].tolist()
if not valid_embeddings:
    print("Error: No valid embeddings found in the dataset after cleaning. Exiting.")
    exit()

corpus_embeddings = np.vstack(valid_embeddings)
dimension = corpus_embeddings.shape[1]

# Normalize embeddings for cosine similarity (L2 normalization)
faiss.normalize_L2(corpus_embeddings)
print(f"Prepared {len(corpus_embeddings)} embeddings of dimension {dimension}.")

# --- 4. Create and Populate FAISS Index ---
index = faiss.IndexFlatIP(dimension)
index.add(corpus_embeddings)
print(f"FAISS index created and populated with {index.ntotal} vectors.")

# --- 5. Define Search Function ---
def find_all_relevant_subreddits(user_query, dataframe, embedding_model, faiss_index, similarity_threshold=0.5, batch_size=1000, max_iterations=50):
    """
    Finds all relevant subreddits based on a user query with a similarity score
    above a given threshold, by searching the FAISS index in batches.
    """
    query_embedding = embedding_model.encode([user_query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_embedding)

    all_found_indices = set()
    all_found_scores = {}
    total_vectors = faiss_index.ntotal
    search_count = 0
    k_initial = min(batch_size, total_vectors)
    distances, indices = faiss_index.search(query_embedding, k_initial)

    for i in range(len(indices[0])):
        idx = indices[0][i]
        score = distances[0][i]
        if score >= similarity_threshold:
            if idx not in all_found_indices:
                all_found_indices.add(idx)
                all_found_scores[idx] = score
        else:
            break

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


user_prompt = input("enter prompt: ")
subreddits = []
try:
    RELEVANCE_THRESHOLD = 0.3
    all_relevant_results = find_all_relevant_subreddits(user_prompt, embed_df, model, index, similarity_threshold=RELEVANCE_THRESHOLD)
    if not all_relevant_results.empty:
        for i, row in all_relevant_results.iterrows():
            subreddits.append({
                'name': row['name'],
                'similarity_score': row['similarity_score']
                })
except Exception as e:
    print("error occured: ", e)

print(subreddits)