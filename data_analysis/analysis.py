import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# -------------------------
# Load Data Functions
# -------------------------

def load_ratings(path="ml-100k/u.data"):
    return pd.read_csv(path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

def load_movies(path="ml-100k/u.item"):
    cols = [
        "movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    return pd.read_csv(path, sep='|', encoding='latin-1', header=None, names=cols)

def load_users(path="ml-100k/u.user"):
    return pd.read_csv(path, sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])

# -------------------------
# Basic Stats
# -------------------------

def print_basic_stats(users_df, movies_df, ratings_df):
    print("✅ Basic Stats:")
    print(f"Total Users   : {users_df['user_id'].nunique()}")
    print(f"Total Movies  : {movies_df['movie_id'].nunique()}")
    print(f"Total Ratings : {len(ratings_df)}")

# -------------------------
# Visualization Functions
# -------------------------

def plot_top_10_popular_movies(ratings_df, movies_df):
    merged = pd.merge(ratings_df, movies_df[['movie_id', 'title']], on='movie_id')
    top10 = merged['title'].value_counts().head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(y=top10.index, x=top10.values, palette="viridis")
    plt.title("Top 10 Most Popular Movies")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Movie Title")
    plt.tight_layout()
    plt.show()

def plot_rating_distribution(ratings_df):
    plt.figure(figsize=(8, 5))
    sns.histplot(ratings_df['rating'], bins=5, kde=False)
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_avg_rating_by_genre(ratings_df, movies_df):
    genre_data = pd.merge(ratings_df, movies_df, on='movie_id')

    genre_cols = movies_df.columns[5:]
    genre_melted = genre_data.melt(
        id_vars=['rating'], value_vars=genre_cols,
        var_name='genre', value_name='is_genre'
    )
    genre_filtered = genre_melted[genre_melted['is_genre'] == 1]

    avg_ratings = genre_filtered.groupby('genre')['rating'].mean().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_ratings.values, y=avg_ratings.index, palette="coolwarm")
    plt.title("Average Rating by Genre")
    plt.xlabel("Average Rating")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.show()

# -------------------------
# Insight Functions
# -------------------------

def get_most_active_users(ratings_df, top_n=5):
    active_users = ratings_df['user_id'].value_counts().head(top_n)
    print("Most active users (by rating count):")
    print(active_users)
    return active_users

def get_top_rated_movies(ratings_df, movies_df, min_ratings=50, top_n=10):
    merged = pd.merge(ratings_df, movies_df[['movie_id', 'title']], on='movie_id')
    stats = merged.groupby('title').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    )
    top_movies = stats[stats['rating_count'] >= min_ratings].sort_values(by='avg_rating', ascending=False).head(top_n)
    print("\nTop highest-rated movies (min 50 ratings):")
    print(top_movies)
    return top_movies


# ratings = load_ratings()
# movies = load_movies()
# users = load_users()

# print_basic_stats(users, movies, ratings)

# plot_top_10_popular_movies(ratings, movies)
# plot_rating_distribution(ratings)
# plot_avg_rating_by_genre(ratings, movies)

# get_most_active_users(ratings)
# get_top_rated_movies(ratings, movies)