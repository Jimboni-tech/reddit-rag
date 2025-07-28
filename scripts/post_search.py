import pandas as pd
from sub_search import setup_search_environment, find_all_relevant_subreddits, main as sub_search_main # Alias main to avoid conflict
from dotenv import load_dotenv
import os 
import praw
from collections import defaultdict
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import torch

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

print("Loading sentence embedding model...")
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print(f"Model loaded on device: {device}")

def get_post_pool(subreddit_name, limit=500, time_filter='year'):
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

def get_top_comments(submission_id, limit=10):
    submission = reddit.submission(id=submission_id)
    comments_data = []
    try:
        submission.comments.replace_more(limit=0)
        top_level_comments = [comment for comment in submission.comments.list() if comment.is_root] 
        top_level_comments.sort(key=lambda c: c.score, reverse=True)
        
        for comment in top_level_comments[:limit]:
            if comment.body:
                comments_data.append({
                    "id": comment.id,
                    "text": comment.body,
                    "score": comment.score,
                    "parent_id": submission_id
                })
    except Exception as e:
        print(f"Error fetching comments for submission {submission_id}: {e}")
    return comments_data

def filter_and_select_relevant_content(query, all_posts_pool, embedding_model, min_post_score=50, min_comment_score=10, num_relevant_posts=5):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu()

    relevant_posts_candidates = []
    
    post_texts = [f"{post['title']} {post['text']}" for post in all_posts_pool]
    post_embeddings = embedding_model.encode(post_texts, convert_to_tensor=True).cpu()

    similarities = cosine_similarity(query_embedding.unsqueeze(0), post_embeddings)[0]

    for i, post in enumerate(all_posts_pool):
        if post["score"] >= min_post_score:
            post_similarity_score = similarities[i]
            relevant_posts_candidates.append({
                **post,
                "relevance_score": post_similarity_score
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
        comment_embeddings = embedding_model.encode(comment_texts, convert_to_tensor=True).cpu()
        comment_similarities = cosine_similarity(query_embedding.unsqueeze(0), comment_embeddings)[0]

        for i, comment in enumerate(all_comments_for_selected_posts):
            if comment["score"] >= min_comment_score:
                comment_similarity_score = comment_similarities[i]
                relevant_comments.append({
                    **comment,
                    "relevance_score": comment_similarity_score
                })
    relevant_comments.sort(key=lambda x: x["relevance_score"], reverse=True)


    final_relevant_data = defaultdict(lambda: {"posts": [], "comments": []})
    for post in selected_posts:
        final_relevant_data[post["subreddit"]]["posts"].append(post)
    for comment in relevant_comments:
        parent_post_subreddit = None
        for post in selected_posts:
            if post["id"] == comment["parent_id"]:
                parent_post_subreddit = post["subreddit"]
                break
        if parent_post_subreddit:
            final_relevant_data[parent_post_subreddit]["comments"].append(comment)

    return final_relevant_data

def format_for_llm(relevant_data):
    context_string = "Context from Reddit:\n\n"
    for subreddit_name, data in relevant_data.items():
        context_string += f"--- Subreddit: r/{subreddit_name} ---\n\n"
        data["posts"].sort(key=lambda x: x["relevance_score"], reverse=True)
        for post in data["posts"]:
            context_string += f"Post Title: {post['title']} (Relevance Score: {post['relevance_score']:.4f})\n"
            context_string += f"Post Content: {post['text']}\n"

            post_comments = [c for c in data["comments"] if c["parent_id"] == post["id"]]
            post_comments.sort(key=lambda x: x["relevance_score"], reverse=True)
            if post_comments:
                context_string += "Comments:\n"
                for comment in post_comments:
                    context_string += f"- {comment['text']} (Relevance Score: {comment['relevance_score']:.4f})\n"
            context_string += "\n---\n\n" 
    return context_string

if __name__ == "__main__":
    user_query = input("Enter your search query: ")
    relevant_subs_df = sub_search_main(user_query) 

    if relevant_subs_df is not None and not relevant_subs_df.empty:
        relevant_subs_list = relevant_subs_df['name'].head(3).tolist()
        print(f"Top 3 Relevant Subreddits: {relevant_subs_list}")

        all_posts_pool = []
        for sub in relevant_subs_list:
            print(f"Retrieving posts from r/{sub}...")
            all_posts_pool.extend(get_post_pool(sub, limit=500)) 

        print(f"\nFiltering and selecting relevant content for query: '{user_query}' using embeddings...")
        final_relevant_data = filter_and_select_relevant_content(user_query, all_posts_pool, model, num_relevant_posts=5)

        print("\nFormatting context for LLM...")
        llm_context = format_for_llm(final_relevant_data)
        
        print("\n--- LLM Context ---")
        print(llm_context)
        print("--- End LLM Context ---")
    else:
        print("No relevant subreddits found or setup failed to start the RAG pipeline.")