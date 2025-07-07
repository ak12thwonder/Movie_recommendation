import pandas as pd
import psycopg2
from datetime import datetime

def load_and_insert_movielens_data(db_config, data_path="ml-100k"):
    """
    Load MovieLens 100K data and insert into PostgreSQL database.
    
    Args:
        db_config (dict): Dictionary with keys host, dbname, user, password, port.
        data_path (str): Path to the MovieLens data folder (default = "ml-100k").
    """
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cursor:

            # ---------------------------- Movies ----------------------------
            movies = pd.read_csv(f"{data_path}/u.item", sep='|', encoding='latin-1', header=None, usecols=(0, 1, 2))
            movies.columns = ['movie_id', 'title', 'release_date']
            movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')

            for _, row in movies.iterrows():
                release_date = row.release_date if pd.notnull(row.release_date) else None
                cursor.execute("""
                    INSERT INTO movies (movie_id, title, release_date)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (movie_id) DO NOTHING
                """, (int(row.movie_id), row.title, release_date))
            print("✅ Movies inserted")

            # ---------------------------- Users ----------------------------
            users = pd.read_csv(f"{data_path}/u.user", sep='|', header=None)
            users.columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']

            for _, row in users.iterrows():
                cursor.execute("""
                    INSERT INTO users (user_id, age, gender, occupation, zip_code)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO NOTHING
                """, (int(row.user_id), int(row.age), row.gender, row.occupation, row.zip_code))
            print("✅ Users inserted")

            # ---------------------------- Ratings ----------------------------
            ratings = pd.read_csv(f"{data_path}/u.data", sep='\t', header=None)
            ratings.columns = ['user_id', 'movie_id', 'rating', 'timestamp']

            for _, row in ratings.iterrows():
                cursor.execute("""
                    INSERT INTO ratings (user_id, movie_id, rating, timestamp)
                    VALUES (%s, %s, %s, %s)
                """, (int(row.user_id), int(row.movie_id), int(row.rating), int(row.timestamp)))
            print("✅ Ratings inserted")

        conn.commit()
        print("🎉 All data inserted successfully.")

# -------------------------------------
# Example usage
# # -------------------------------------
# if __name__ == "__main__":
#     db_config = {
#         "host": "localhost",
#         "dbname": "12thwonder",
#         "user": "postgres",
#         "password": "admin",
#         "port": "5432"
#     }
#     load_and_insert_movielens_data(db_config)
