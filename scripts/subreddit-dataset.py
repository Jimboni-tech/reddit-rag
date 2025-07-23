import praw
from dotenv import load_dotenv
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('corpora/stopwords')
except LookupError: # Changed to LookupError
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError: # Changed to LookupError
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # Changed to LookupError
    nltk.download('punkt')


load_dotenv()
CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID") 
CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
USER_AGENT = os.environ.get("REDDIT_USER_AGENT")
USERNAME = os.environ.get("REDDIT_USERNAME") 
PASSWORD = os.environ.get("REDDIT_PASSWORD") 
print("CLIENT_ID:", CLIENT_ID)
print("CLIENT_SECRET:", CLIENT_SECRET)
print("USER_AGENT:", USER_AGENT)
print("USERNAME:", USERNAME)
print("PASSWORD:", PASSWORD)



def clean_text(text):
    
    if text is None:
        return ""
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'[#*_`]', '', text)
    text = text.lower()
    '''text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)

    words = nltk.word_tokenize(text) #tokenization

    stop_words = set(stopwords.words('english')) #stop words
    words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer() #lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    text = ' '.join(words).strip()
    text = re.sub(r'\s+', ' ', text).strip()'''


    return text


def getTopSubreddits(limit):
    reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
            username=USERNAME,
            password=PASSWORD
        )
    print('Gathering subreddits.')
    subreddits = []
    try: 
        num = 1
        for subreddit in reddit.subreddits.popular(limit=limit):
            if subreddit.over18 == False:
                subreddits.append({
                    "name": subreddit.display_name,
                    "subscribers": subreddit.subscribers,
                    "url": f"https://www.reddit.com/r/{subreddit.display_name}/",
                    "public_description": clean_text(subreddit.public_description),
                })
                print(num)
                num += 1

                if len(subreddits) >= limit:
                    break
    except Exception as e:
        print(f"Error: {e}")
    sub_df = pd.DataFrame(subreddits)
    if not sub_df.empty:
       sub_df = sub_df.sort_values(by='subscribers', ascending=False).reset_index(drop=True)

    return sub_df



if __name__ == "__main__":
    if CLIENT_ID != None and CLIENT_SECRET != None and USERNAME != None and PASSWORD != None:
        sub_df = getTopSubreddits(5000)
        print(sub_df.head(5))
        print(len(sub_df))
        sub_df.to_csv('./datasets/top-subreddits.csv')
    else:
        print("Missing access information")
    
