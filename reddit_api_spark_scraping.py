from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
import praw
import prawcore
import time
from datetime import datetime
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("RedditDataCollection") \
    .getOrCreate()

# Reddit API credentials
reddit = praw.Reddit(
    client_id='<redacted_for_privacy>',
    client_secret='<redacted_for_privacy>',
    user_agent='<redacted_for_privacy>'
)

# Define date range for the query
START_DATE = '2023-01-01'
END_DATE = '2024-11-11'
start_time = int(time.mktime(time.strptime(START_DATE, "%Y-%m-%d")))
end_time = int(time.mktime(time.strptime(END_DATE, "%Y-%m-%d")))

# Existing data file
existing_csv = "reddit_data_filtered.csv"
try:
    existing_df = pd.read_csv(existing_csv)
    processed_ids = set(existing_df["id"])
except FileNotFoundError:
    processed_ids = set()

# Subreddits and topics
political_subreddits = [
    'politics', 'news', 'worldnews', 'conservative', 'liberal', 'progressive',
    'democrats', 'Republican', 'Libertarian', 'GreenParty', 'SocialDemocracy',
    'worldpolitics', 'BlackLivesMatter', 'AbortionDebate', 'PoliticalHumor',
    'ModeratePolitics', 'PoliticalDiscussion', 'worldevents', 'business', 'economics',
    'environment', 'energy', 'law', 'history', 'worldnews2', 'politics2',
    'uspolitics', 'americangovernment', 'lgbtnews', 'worldnews', 'alltheleft',
    'labor', 'democracy', 'freethought', 'equality', 'lgbt'
]
sort_methods = ['hot', 'top', 'best', 'controversial']
topics = ['Gun Control', 'Abortion', 'Racial Inequality', 'Climate Change']

# Define schema for Spark DataFrame
schema = StructType([
    StructField("id", StringType(), True),
    StructField("title", StringType(), True),
    StructField("selftext", StringType(), True),
    StructField("score", IntegerType(), True),
    StructField("comments_count", IntegerType(), True),
    StructField("created_utc", StringType(), True),
    StructField("subreddit", StringType(), True),
    StructField("url", StringType(), True),
    StructField("topic", StringType(), True),
    StructField("comments", ArrayType(StringType()), True)
])

# Function to fetch posts and comments
def fetch_posts_and_comments(topic, subreddit_name, sort_type):
    data = []
    query = f"{topic} timestamp:{start_time}..{end_time}"
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        print(f"Fetching '{sort_type}' posts from r/{subreddit_name} for topic '{topic}'...")
        
        for submission in subreddit.search(query, sort=sort_type, limit=100):
            if submission.id in processed_ids:
                continue  # Skip already processed posts
            
            submission.comments.replace_more(limit=0)
            comments = [comment.body for comment in submission.comments.list()]
            
            data.append((
                submission.id,
                submission.title,
                submission.selftext,
                submission.score,
                submission.num_comments,
                datetime.utcfromtimestamp(submission.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                subreddit_name,
                submission.url,
                topic,
                comments
            ))
            
            processed_ids.add(submission.id)  # Add Post ID to processed set
    except prawcore.exceptions.NotFound:
        print(f"Subreddit '{subreddit_name}' not found. Skipping...")
    except Exception as e:
        print(f"An error occurred with subreddit '{subreddit_name}': {e}")
    
    return data

# Main data collection loop
final_data = []
for topic in topics:
    for subreddit in political_subreddits:
        for sort_type in sort_methods:
            final_data.extend(fetch_posts_and_comments(topic, subreddit, sort_type))
            time.sleep(10)  # Delay to avoid API limits

# Convert data to Spark DataFrame
if final_data:
    rdd = spark.sparkContext.parallelize(final_data)
    df = spark.createDataFrame(rdd, schema=schema)
    
    # Save new data to CSV
    try:
        existing_spark_df = spark.read.csv(existing_csv, header=True, schema=schema)
        combined_df = existing_spark_df.union(df)
    except Exception as e:
        print(f"File '{existing_csv}' not found or invalid. Creating a new file: {e}")
        combined_df = df

    combined_df.write.csv(existing_csv, mode='overwrite', header=True)
    print(f"Data collection complete! Saved to '{existing_csv}'.")
else:
    print("Nothing new.")

# Stop the Spark session
spark.stop()
