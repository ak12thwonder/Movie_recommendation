import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# -----------------------------
# Load the ratings dataset
# -----------------------------
def load_ratings(path="data/ml-100k/u.data"):
    return pd.read_csv(path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

# -----------------------------
# Build User-Item Matrix
# -----------------------------
def build_user_item_matrix(ratings_df):
    return ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# -----------------------------
# Compute User-User Similarity
# -----------------------------
def compute_user_similarity(user_item_matrix):
    similarity = cosine_similarity(user_item_matrix)
    return pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# -----------------------------
# Recommend ratings for a user
# -----------------------------
def predict_rating(user_id, movie_id, user_item_matrix, similarity_matrix, k=10):
    if movie_id not in user_item_matrix.columns:
        return None
    
    # Get similarities of other users with this user
    sim_scores = similarity_matrix[user_id].drop(user_id, errors='ignore')

    # Ratings by other users for the target movie
    movie_ratings = user_item_matrix[movie_id]

    # Only keep users who have rated the movie
    valid_users = movie_ratings[movie_ratings > 0].index
    sim_scores = sim_scores.loc[valid_users]
    movie_ratings = movie_ratings.loc[valid_users]

    if sim_scores.empty:
        return None

    # Select top-k similar users
    top_k_users = sim_scores.sort_values(ascending=False).head(k)
    top_k_ratings = movie_ratings.loc[top_k_users.index]

    # Weighted average
    weighted_sum = np.dot(top_k_ratings, top_k_users)
    sim_sum = top_k_users.sum()

    if sim_sum == 0:
        return None

    return weighted_sum / sim_sum

# -----------------------------
# Evaluate Model (RMSE + MAE)
# -----------------------------
def evaluate_model(train_df, test_df, k=10):
    user_item_matrix = build_user_item_matrix(train_df)
    similarity_matrix = compute_user_similarity(user_item_matrix)

    y_true = []
    y_pred = []

    for _, row in test_df.iterrows():
        user = row['user_id']
        movie = row['movie_id']
        actual = row['rating']

        if user in user_item_matrix.index:
            pred = predict_rating(user, movie, user_item_matrix, similarity_matrix, k)
            if pred is not None:
                y_true.append(actual)
                y_pred.append(pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return rmse, mae, len(y_true)

# -----------------------------
# Run It All
# -----------------------------
# if __name__ == "__main__":
#     ratings_df = load_ratings()
#     train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

#     rmse, mae, count = evaluate_model(train_df, test_df, k=10)

#     print("✅ Collaborative Filtering Evaluation:")
#     print(f"Evaluated on {count} test ratings")
#     print(f"RMSE: {rmse:.4f}")
#     print(f"MAE : {mae:.4f}")
