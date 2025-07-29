import pandas as pd
from sub_search import setup_search_environment, find_all_relevant_subreddits, main as sub_search_main 
from dotenv import load_dotenv
import os
import praw
from collections import defaultdict
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torch.nn.functional as F 
from transformers import AutoTokenizer 
import functools 


load_dotenv()
CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
USER_AGENT = os.environ.get("REDDIT_USER_AGENT")
USERNAME = os.environ.get("REDDIT_USERNAME")
PASSWORD = os.environ.get("REDDIT_PASSWORD")

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT,
    username=USERNAME,
    password=PASSWORD
)
print("Initializing PyTorch device...")
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"PyTorch operations will use device: {device}")

print("Loading sentence embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print(f"Sentence embedding model loaded on device: {device}")

# --- Load Llama 3 Tokenizer ---
# IMPORTANT: Replace 'meta-llama/Llama-3-8B-Instruct' with the actual Llama 3 model identifier
# you plan to use when it's publicly available and supported by Hugging Face Transformers.
# Llama 3.1 models have larger context windows (e.g., 128k tokens for 8B and 70B).
# For general Llama 3 (initial release), the context window is typically 8k tokens.
# Adjust MAX_CONTEXT_TOKENS accordingly.
try:
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    print("Llama tokenizer loaded successfully.")
    # Typical Llama 3 context window for 8B/70B base models is 8192 tokens.
    # Llama 3.1 models (newer) go up to 128k tokens. Adjust based on your specific Llama 3 version.
    # processing 8k tokens is more practical than 128k in real-time.
    MAX_CONTEXT_TOKENS = 8192 
    print(f"Set MAX_CONTEXT_TOKENS based on Llama 3.1 8B context: {MAX_CONTEXT_TOKENS} tokens.")
except Exception as e:
    print(f"Could not load Llama tokenizer (e.g., 'meta-llama/Llama-3-8B-Instruct'). Please ensure you have access and the correct model identifier: {e}")
    print("Falling back to a more generic token estimation for truncation.")
    llama_tokenizer = None
    MAX_CONTEXT_TOKENS = 4000 

@functools.lru_cache(maxsize=256)
def get_post_pool(subreddit_name: str, limit: int = 1000, time_filter: str = 'year') -> list[dict]:
    """
    Fetches posts from a specified subreddit.
    Cached to improve efficiency for repeated calls.
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    try:

        for submission in subreddit.top(time_filter=time_filter, limit=limit):
            if submission.selftext:
                posts_data.append({
                    "id": submission.id,
                    "title": submission.title,
                    "text": submission.selftext,
                    "score": submission.score,
                    "url": submission.url,
                    "subreddit": subreddit_name
                })
    except Exception as e:
        print(f"Error fetching posts from r/{subreddit_name}: {e}")
    return posts_data

@functools.lru_cache(maxsize=512) 
def get_top_comments(submission_id: str, limit: int = 10) -> list[dict]:
    """
    Fetches top comments for a given submission ID.
    Cached to improve efficiency for repeated calls.
    """
    submission = reddit.submission(id=submission_id)
    comments_data = []
    try:
    
        submission.comments.replace_more(limit=0)
        top_level_comments = [comment for comment in submission.comments.list() if comment.is_root]
        top_level_comments.sort(key=lambda c: c.score, reverse=True) # Sort by score

        for comment in top_level_comments[:limit]:
            if comment.body and comment.author:
                comments_data.append({
                    "id": comment.id,
                    "text": comment.body,
                    "score": comment.score,
                    "parent_id": submission_id,
                    "author": str(comment.author) 
                })
    except Exception as e:
        print(f"Error fetching comments for submission {submission_id}: {e}")
    return comments_data


def filter_and_select_relevant_content(query: str, all_posts_pool: list[dict], embedding_model: SentenceTransformer,
                                       min_post_score: int = 50, min_comment_score: int = 10, num_relevant_posts: int = 5) -> defaultdict:
    """
    Filters and selects the most relevant posts and their comments based on cosine similarity
    to the query and predefined score thresholds.
    Utilizes MPS for embedding and similarity calculations.
    """
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).to(device)

    relevant_posts_candidates = []

    if not all_posts_pool:
        print("No posts in the pool to filter.")
        return defaultdict(lambda: {"posts": [], "comments": []})

    post_texts = [f"{post['title']} {post['text']}" for post in all_posts_pool]
    post_embeddings = embedding_model.encode(post_texts, convert_to_tensor=True, batch_size=64).to(device)

    similarities = F.cosine_similarity(query_embedding.unsqueeze(0), post_embeddings).cpu().numpy()

    for i, post in enumerate(all_posts_pool):
        if post["score"] >= min_post_score:
            post_similarity_score = similarities[i]
            relevant_posts_candidates.append({
                **post, 
                "relevance_score": float(post_similarity_score)
            })

    relevant_posts_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
    selected_posts = relevant_posts_candidates[:num_relevant_posts]

    all_comments_for_selected_posts = []
    for post in selected_posts:
        comments = get_top_comments(post["id"])
        all_comments_for_selected_posts.extend(comments)

    relevant_comments = []
    if all_comments_for_selected_posts:
        comment_texts = [comment["text"] for comment in all_comments_for_selected_posts]
        comment_embeddings = embedding_model.encode(comment_texts, convert_to_tensor=True, batch_size=64).to(device)
        comment_similarities = F.cosine_similarity(query_embedding.unsqueeze(0), comment_embeddings).cpu().numpy()

        for i, comment in enumerate(all_comments_for_selected_posts):
            if comment["score"] >= min_comment_score:
                comment_similarity_score = comment_similarities[i]
                relevant_comments.append({
                    **comment,
                    "relevance_score": float(comment_similarity_score)
                })
    relevant_comments.sort(key=lambda x: x["relevance_score"], reverse=True)

    final_relevant_data = defaultdict(lambda: {"posts": [], "comments": []})
    for post in selected_posts:
        final_relevant_data[post["subreddit"]]["posts"].append(post)

    for comment in relevant_comments:
        parent_post_subreddit = next((post["subreddit"] for post in selected_posts if post["id"] == comment["parent_id"]), None)
        if parent_post_subreddit:
            final_relevant_data[parent_post_subreddit]["comments"].append(comment)

    return final_relevant_data

def format_for_llm(relevant_data: defaultdict) -> str:
    """
    Formats the relevant data into a structured string for LLM context.
    Prioritizes highly relevant posts and their comments.
    """
    context_parts = []
    context_parts.append("Context from Reddit:\n\n")

    sorted_subreddits = sorted(relevant_data.keys(), key=lambda sub: (
        max((p["relevance_score"] for p in relevant_data[sub]["posts"]), default=-1.0)
        if relevant_data[sub]["posts"] else -1.0, sub
    ), reverse=True)

    for subreddit_name in sorted_subreddits:
        data = relevant_data[subreddit_name]
        if not data["posts"] and not data["comments"]:
            continue 

        context_parts.append(f"--- Subreddit: r/{subreddit_name} ---\n\n")

        data["posts"].sort(key=lambda x: x["relevance_score"], reverse=True)

        for post in data["posts"]:
            post_string = []
            post_string.append(f"Post Title: {post['title']} (Score: {post['score']}, Relevance: {post['relevance_score']:.4f})\n")
            post_string.append(f"Post Content: {post['text']}\n")

            post_comments = [c for c in data["comments"] if c["parent_id"] == post["id"]]
            post_comments.sort(key=lambda x: x["relevance_score"], reverse=True) 
            post_comments_by_score = sorted(post_comments, key=lambda x: x["score"], reverse=True) 

            if post_comments:
                post_string.append("Top Relevant Comments:\n")
                for comment in post_comments[:3]:
                    post_string.append(f"- Comment (Score: {comment['score']}, Relevance: {comment['relevance_score']:.4f}): {comment['text']}\n")
            post_string.append("\n---\n\n")
            context_parts.append("".join(post_string))

    return "".join(context_parts)

def truncate_context_for_llm(context_string: str, max_tokens: int, tokenizer: AutoTokenizer) -> str:
    """
    Truncates the context string to fit within a maximum token limit using a specific tokenizer.
    Prioritizes retaining information from the beginning (more relevant content).
    """
    if tokenizer is None:
        if len(context_string) <= max_tokens * 4: 
            return context_string
        else:
            truncated_context = context_string[:max_tokens * 4]
            last_newline = truncated_context.rfind('\n\n')
            if last_newline != -1 and last_newline > max_tokens * 4 * 0.8:
                truncated_context = truncated_context[:last_newline]
            return truncated_context + "\n\n... [Context truncated by character count, tokenizer not loaded]"

    tokens = tokenizer.encode(context_string, add_special_tokens=False)

    if len(tokens) <= max_tokens:
        return context_string
    else:

        truncated_tokens = tokens[:max_tokens]

        truncated_context = tokenizer.decode(truncated_tokens, clean_up_tokenization_spaces=True)
        return truncated_context + "\n\n... [Context truncated by token count]"


if __name__ == "__main__":
    print("Starting Reddit RAG Pipeline...")
    user_query = input("Enter your search query: ")
    print(f"\nSearching for relevant subreddits for query: '{user_query}'...")
    relevant_subs_df = sub_search_main(user_query) 

    if relevant_subs_df is not None and not relevant_subs_df.empty:

        relevant_subs_list = relevant_subs_df['name'].head(3).tolist()
        print(f"Top 3 Relevant Subreddits: {relevant_subs_list}")
        all_posts_pool = []
        for sub in relevant_subs_list:
            print(f"Retrieving posts from r/{sub} (cached)...")
            all_posts_pool.extend(get_post_pool(sub, limit=500)) 

        print(f"\nFetched {len(all_posts_pool)} posts in total.")

        print(f"\nFiltering and selecting relevant content for query: '{user_query}' using embeddings on {device}...")
        final_relevant_data = filter_and_select_relevant_content(
            user_query, all_posts_pool, model,
            min_post_score=50, 
            min_comment_score=10,
            num_relevant_posts=5 
        )

        print("\nFormatting context for LLM...")
        raw_llm_context = format_for_llm(final_relevant_data)
        print(f"\nTruncating context to fit LLM's {MAX_CONTEXT_TOKENS} token window...")
        llm_context = truncate_context_for_llm(raw_llm_context, MAX_CONTEXT_TOKENS, llama_tokenizer)

        print(f"\n--- LLM Context (Approx. {len(llama_tokenizer.encode(llm_context, add_special_tokens=False)) if llama_tokenizer else 'N/A'} tokens) ---")
        print(llm_context)
        print("--- End LLM Context ---")

    else:
        print("No relevant subreddits found or sub_search setup failed to start the RAG pipeline.")