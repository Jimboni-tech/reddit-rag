import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ast
import re

# --- 1. Load Data ---
try:
    embed_df = pd.read_csv('datasets/sub-embed-dataset.csv')
    if 'embeddings' in embed_df.columns:
        # Convert the column to string type first and fill NaN values
        embed_df['embeddings'] = embed_df['embeddings'].astype(str).fillna('')

        # Function to clean and parse each embedding string
        def parse_embedding_string(s):
            s = s.strip()
            if not s: # Handle empty strings after stripping
                return None

            # Remove outer brackets if they exist, and then strip any remaining whitespace
            s = s.strip('[] ').strip()

            if not s: # If it was just '[]' or empty string
                return None

            # Split the string by any whitespace and filter out empty strings
            # This creates a list of number strings
            number_strings = re.split(r'\s+', s)
            number_strings = [num_str for num_str in number_strings if num_str] # Filter out any empty strings from split

            # If there are no number strings after splitting, return None
            if not number_strings:
                return None

            # Join them with commas to form a valid Python list string
            cleaned_s = '[' + ', '.join(number_strings) + ']'

            try:
                # Safely evaluate the cleaned string
                return np.array(ast.literal_eval(cleaned_s), dtype=np.float32)
            except (ValueError, SyntaxError) as e:
                print(f"Error evaluating embedding string: '{s}' -> {e}")
                return None # Return None for problematic entries

        embed_df['embeddings'] = embed_df['embeddings'].apply(parse_embedding_string)

        # Drop rows where embedding parsing failed (where parse_embedding_string returned None)
        embed_df.dropna(subset=['embeddings'], inplace=True)

    else:
        print("Error: The DataFrame must contain an 'embeddings' column.")
        exit()

except FileNotFoundError:
    print("Error: 'datasets/sub-embed-dataset.csv' not found. Please check the path.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

# --- 2. Initialize Embedding Model ---
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print("Sentence Transformer model loaded: all-mpnet-base-v2")

# --- 3. Prepare Embeddings for FAISS ---
valid_embeddings = embed_df['embeddings'].tolist() # Already dropped NaNs, so convert directly
if not valid_embeddings:
    print("Error: No valid embeddings found in the dataset after cleaning. Exiting.")
    exit()

corpus_embeddings = np.vstack(valid_embeddings)
dimension = corpus_embeddings.shape[1] # Dimension of the embeddings

# Normalize embeddings for cosine similarity (L2 normalization)
faiss.normalize_L2(corpus_embeddings)
print(f"Prepared {len(corpus_embeddings)} embeddings of dimension {dimension}.")

# --- 4. Create and Populate FAISS Index ---
index = faiss.IndexFlatIP(dimension)
index.add(corpus_embeddings)
print(f"FAISS index created and populated with {index.ntotal} vectors.")

# --- 5. Define Search Function ---
def find_top_subreddits(user_query, dataframe, embedding_model, faiss_index, k=10):
    """
    Finds the top k most relevant subreddits based on a user query using FAISS.
    """
    query_embedding = embedding_model.encode([user_query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_embedding)
    distances, indices = faiss_index.search(query_embedding, k)
    top_subreddit_indices = indices[0]
    top_subreddits_df = dataframe.iloc[top_subreddit_indices].copy()
    top_subreddits_df['similarity_score'] = distances[0]
    return top_subreddits_df.sort_values(by='similarity_score', ascending=False)

# --- 6. Get User Input and Display Results ---
print("\n--- Subreddit Search ---")
while True:
    user_prompt = input("Enter a topic to find related subreddits (or 'exit' to quit): ")
    if user_prompt.lower() == 'exit':
        print("Exiting search. Goodbye! 👋")
        break

    if not user_prompt.strip():
        print("Please enter a valid topic.")
        continue

    print(f"Searching for subreddits related to: '{user_prompt}'...")
    try:
        top_results = find_top_subreddits(user_prompt, embed_df, model, index, k=10)

        if not top_results.empty:
            print("\n✨ Top 10 Most Relevant Subreddits: ✨")
            print("-" * 40)
            for i, row in top_results.iterrows():
                print(f"Subreddit: r/{row['name']}")
                print(f"Subscribers: {row['subscribers']:,}" if 'subscribers' in row and pd.notna(row['subscribers']) else "Subscribers: N/A")
                print(f"URL: {row['url']}" if 'url' in row and pd.notna(row['url']) else "URL: N/A")
                print(f"Description: {row['public_description']}" if 'public_description' in row and pd.notna(row['public_description']) else "Description: N/A")
                print(f"Similarity Score: {row['similarity_score']:.4f}")
                print("-" * 40)
        else:
            print("No relevant subreddits found for this query. Try a different topic.")

    except Exception as e:
        print(f"An error occurred during search: {e}")