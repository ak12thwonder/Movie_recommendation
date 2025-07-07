import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------
# Load Data
# -------------------------------------
# def load_data(ratings_path="../data/ml-100k/u.data", movies_path="../data/ml-100k/u.item"):
def load_data(ratings_path="data/ml-100k/u.data", movies_path="data/ml-100k/u.item"):

    # Load ratings
    ratings_df = pd.read_csv(ratings_path, sep='\t', names=["user_id", "item_id", "rating", "timestamp"])
    ratings_df.drop("timestamp", axis=1, inplace=True)

    # Load movie titles
    movies_df = pd.read_csv(
        movies_path,
        sep='|',
        encoding='latin-1',
        header=None,
        names=["item_id", "title"] + [f"col_{i}" for i in range(22)]
    )
    movies_df = movies_df[["item_id", "title"]]

    return ratings_df, movies_df

# -------------------------------------
# User-Item Matrix
# -------------------------------------
def create_user_item_matrix(ratings_df):
    return ratings_df.pivot_table(index="user_id", columns="item_id", values="rating")

# -------------------------------------
# User Similarity Matrix
# -------------------------------------
def compute_user_similarity(user_item_matrix):
    filled_matrix = user_item_matrix.fillna(0)
    similarity_matrix = cosine_similarity(filled_matrix)
    return pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

# -------------------------------------
# Predict Ratings
# -------------------------------------
def predict_ratings(user_id, user_item_matrix, similarity_matrix, k=5):
    user_ratings = user_item_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings.isna()].index

    similar_users = similarity_matrix[user_id].drop(user_id).nlargest(k)

    predictions = {}
    for item in unrated_items:
        neighbor_ratings = user_item_matrix.loc[similar_users.index, item]
        valid_ratings = neighbor_ratings.dropna()

        if not valid_ratings.empty:
            weights = similar_users[valid_ratings.index]
            predictions[item] = np.dot(valid_ratings, weights) / weights.sum()

    return pd.Series(predictions).sort_values(ascending=False)

# -------------------------------------
# Recommend Movies with Titles
# -------------------------------------
def recommend_movies(user_id, user_item_matrix, similarity_matrix, movies_df, n=5):
    predictions = predict_ratings(user_id, user_item_matrix, similarity_matrix)
    top_n = predictions.head(n).reset_index()
    top_n.columns = ["item_id", "predicted_rating"]

    # Merge with movie titles
    top_n = top_n.merge(movies_df, on="item_id")
    return top_n[["title", "predicted_rating"]]

# -------------------------------------
# Run Full Recommender Pipeline
# -------------------------------------
def run_recommender(user_id, n=5):
    ratings_df, movies_df = load_data()
    user_item_matrix = create_user_item_matrix(ratings_df)
    similarity_matrix = compute_user_similarity(user_item_matrix)
    recommendations = recommend_movies(user_id, user_item_matrix, similarity_matrix, movies_df, n)
    return recommendations

# -------------------------------------
# Test for Sample User
# -------------------------------------
if __name__ == "__main__":
    user_id = 100  # change to any user ID between 1 and 943
    print(f"Top movie recommendations for User {user_id}:\n")
    print(run_recommender(user_id, n=5))


# import pandas as pd

# df = pd.read_csv(
#     "../data/ml-100k/u.data",
#     sep="\t",  # tab-separated
#     header=None,  # no header row in the file
#     names=["user_id", "movie_id", "rating", "timestamp"]  # assign column names
# )
# print(df.head())
# print(df.columns)