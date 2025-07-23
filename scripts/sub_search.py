import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ast
import re

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

def setup_search_environment(embeddings_path='./datasets/top-subreddits-embeddings.csv'):

    try:
        embed_df = pd.read_csv(embeddings_path)
        embed_df['embeddings'] = embed_df['embeddings'].apply(parse_embedding_string)
        embed_df.dropna(subset=['embeddings'], inplace=True)
        print("Embeddings DataFrame loaded and parsed.")
    except FileNotFoundError:
        print(f"Error: '{embeddings_path}' not found. Please run 'preparing embeddings' script first.")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None, None, None

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    print("Sentence Transformer model loaded: all-mpnet-base-v2")

    valid_embeddings = embed_df['embeddings'].tolist()
    if not valid_embeddings:
        print("Error: No valid embeddings found in the dataset after cleaning. Exiting setup.")
        return None, None, None

    corpus_embeddings = np.vstack(valid_embeddings)
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    dimension = corpus_embeddings.shape[1]

    faiss.normalize_L2(corpus_embeddings)
    print(f"Prepared {len(corpus_embeddings)} embeddings of dimension {dimension}.")

    index = faiss.IndexFlatIP(dimension)
    index.add(corpus_embeddings)
    print(f"FAISS index created and populated with {index.ntotal} vectors.")

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
            RELEVANCE_THRESHOLD = 0.4
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