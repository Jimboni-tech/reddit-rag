import pandas as pd
from sub_search import setup_search_environment, find_all_relevant_subreddits, main
from dotenv import load_dotenv
import os 
import praw

load_dotenv()
CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID") 
CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
USER_AGENT = os.environ.get("REDDIT_USER_AGENT")
USERNAME = os.environ.get("REDDIT_USERNAME") 
PASSWORD = os.environ.get("REDDIT_PASSWORD") 



relevant_subs = main()

if relevant_subs is not None:
    reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
            username=USERNAME,
            password=PASSWORD
        )
    for sub in relevant_subs['name']:
