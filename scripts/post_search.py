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
import gc
import requests 
import json 

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


device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"PyTorch operations will use device: {device}")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print(f"Sentence embedding model loaded on device: {device}")


llama_tokenizer = None
llama_model_name_for_tokenizer = "meta-llama/Meta-Llama-3-8B-Instruct" 
MAX_CONTEXT_TOKENS = 8192

try:
    print(f"Loading Llama 3 tokenizer: {llama_model_name_for_tokenizer}...")
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name_for_tokenizer)
    MAX_CONTEXT_TOKENS = llama_tokenizer.model_max_length
    if MAX_CONTEXT_TOKENS > 100000: 
        MAX_CONTEXT_TOKENS = 128000
    elif MAX_CONTEXT_TOKENS < 8000: 
        MAX_CONTEXT_TOKENS = 8192
    print(f"{MAX_CONTEXT_TOKENS} tokens")

except Exception as e:
    print(f"Could not load Llama tokenizer for token estimation: {e}")
    llama_tokenizer = None
    MAX_CONTEXT_TOKENS = 4000 


@functools.lru_cache(maxsize=256)
def get_post_pool(subreddit_name: str, limit: int = 600, time_filter: str = 'year') -> list[dict]:
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

    submission = reddit.submission(id=submission_id)
    comments_data = []
    try:
        submission.comments.replace_more(limit=0)
        top_level_comments = [comment for comment in submission.comments.list() if comment.is_root]
        top_level_comments.sort(key=lambda c: c.score, reverse=True)

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

            if post_comments:
                post_string.append("Top Relevant Comments:\n")
                for comment in post_comments[:3]:
                    post_string.append(f"- Comment (Score: {comment['score']}, Relevance: {comment['relevance_score']:.4f}): {comment['text']}\n")
            post_string.append("\n---\n\n")
            context_parts.append("".join(post_string))

    return "".join(context_parts)

def truncate_context_for_llm(context_string: str, max_tokens: int, tokenizer: AutoTokenizer) -> str:

    if tokenizer is None:
        max_chars = int(max_tokens * 4)
        if len(context_string) <= max_chars:
            return context_string
        else:
            truncated_context = context_string[:max_chars]
            last_newline = truncated_context.rfind('\n\n')
            if last_newline != -1 and last_newline > max_chars * 0.8:
                truncated_context = truncated_context[:last_newline]
            return truncated_context + "\n\n... [Context truncated by character count, tokenizer not loaded]"

    tokens = tokenizer.encode(context_string, add_special_tokens=False)

    if len(tokens) <= max_tokens:
        return context_string
    else:
        truncated_tokens = tokens[:max_tokens]
        truncated_context = tokenizer.decode(truncated_tokens, clean_up_tokenization_spaces=True)
        return truncated_context + "\n\n... [Context truncated by token count]"


def generate_ollama_response(user_query: str, context: str, ollama_model_name: str = "llama3", ollama_api_url: str = "http://localhost:11434/api/chat") -> str:
    """
    Generates a response from the Llama 3 LLM via the Ollama API.
    """
    messages = [
        {"role": "system", "content": "You are a helpful and accurate assistant. Use the provided Reddit context to answer the user's question. If the context does not contain enough information, state that clearly and do not make up answers."},
        {"role": "user", "content": f"Based on the following Reddit context, answer the question:\n\nContext:\n{context}\n\nQuestion: {user_query}"}
    ]

    payload = {
        "model": ollama_model_name,
        "messages": messages,
        "stream": False, 
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 512 
        }
    }

    headers = {"Content-Type": "application/json"}

    print(f"\nSending request to Ollama with model: {ollama_model_name}...")
    try:
        response = requests.post(ollama_api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() 

        response_data = response.json()
        if "message" in response_data and "content" in response_data["message"]:
            return response_data["message"]["content"].strip()
        elif "error" in response_data:
            return f"Error from Ollama API: {response_data['error']}"
        else:
            return "Unexpected response format from Ollama API."

    except requests.exceptions.ConnectionError as e:
        return f"Could not connect to Ollama. Is it running? Error: {e}"
    except requests.exceptions.Timeout:
        return "Ollama API request timed out."
    except requests.exceptions.RequestException as e:
        return f"An error occurred during the Ollama API request: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


if __name__ == "__main__":
    user_query = input("Enter your search query: ")
    print(f"\nSearching for relevant subreddits for query: '{user_query}'...")
    relevant_subs_df = sub_search_main(user_query)

    if relevant_subs_df is not None and not relevant_subs_df.empty:
        relevant_subs_list = relevant_subs_df['name'].head(3).tolist()
        print(f"Subreddits: {relevant_subs_list}")

        all_posts_pool = []
        for sub in relevant_subs_list:
            all_posts_pool.extend(get_post_pool(sub, limit=600))

        print(f"\{len(all_posts_pool)} posts")

        final_relevant_data = filter_and_select_relevant_content(
            user_query, all_posts_pool, embedding_model,
            min_post_score=50,
            min_comment_score=10,
            num_relevant_posts=5
        )

        raw_llm_context = format_for_llm(final_relevant_data)

        llm_context = truncate_context_for_llm(raw_llm_context, MAX_CONTEXT_TOKENS, llama_tokenizer)

        approx_token_count = len(llama_tokenizer.encode(llm_context, add_special_tokens=False)) if llama_tokenizer else "N/A (tokenizer not loaded)"

        llm_response = generate_ollama_response(user_query, llm_context, ollama_model_name="llama3")
        print("\nLLM Response:")
        print(llm_response)

    else:
        print("No relevant subreddits found or sub_search setup failed to start the RAG pipeline.")