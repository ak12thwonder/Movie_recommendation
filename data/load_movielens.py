import pandas as pd
import psycopg2
from datetime import datetime

def load_and_insert_movielens_data(db_config, data_path="ml-100k"):
    """
    Load MovieLens 100K data and insert into PostgreSQL database.
    Drops and recreates tables for a clean start.
    Args:
        db_config (dict): Dictionary with keys host, dbname, user, password, port.
        data_path (str): Path to the MovieLens data folder (default = "ml-100k").
    """
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cursor:
            # Drop tables if they exist
            cursor.execute("DROP TABLE IF EXISTS ratings;")
            cursor.execute("DROP TABLE IF EXISTS users;")
            cursor.execute("DROP TABLE IF EXISTS movies;")
            conn.commit()

            # Create movies table with full schema
            cursor.execute("""
                CREATE TABLE movies (
                    movie_id INTEGER PRIMARY KEY,
                    title TEXT,
                    release_date TEXT,
                    video_release_date TEXT,
                    IMDb_URL TEXT,
                    unknown INTEGER,
                    Action INTEGER,
                    Adventure INTEGER,
                    Animation INTEGER,
                    Children INTEGER,
                    Comedy INTEGER,
                    Crime INTEGER,
                    Documentary INTEGER,
                    Drama INTEGER,
                    Fantasy INTEGER,
                    FilmNoir INTEGER,
                    Horror INTEGER,
                    Musical INTEGER,
                    Mystery INTEGER,
                    Romance INTEGER,
                    SciFi INTEGER,
                    Thriller INTEGER,
                    War INTEGER,
                    Western INTEGER
                )
            """)

            # Create users table
            cursor.execute("""
                CREATE TABLE users (
                    user_id INTEGER PRIMARY KEY,
                    age INTEGER,
                    gender TEXT,
                    occupation TEXT,
                    zip_code TEXT
                )
            """)

            # Create ratings table
            cursor.execute("""
                CREATE TABLE ratings (
                    user_id INTEGER,
                    movie_id INTEGER,
                    rating INTEGER,
                    timestamp INTEGER
                )
            """)
            conn.commit()

            # ---------------------------- Movies ----------------------------
            movie_columns = [
                'movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                'Documentary', 'Drama', 'Fantasy', 'FilmNoir', 'Horror', 'Musical', 'Mystery',
                'Romance', 'SciFi', 'Thriller', 'War', 'Western'
            ]
            movies = pd.read_csv(f"{data_path}/u.item", sep='|', encoding='latin-1', header=None, names=movie_columns)
            movies = movies[movie_columns]  # Ensure only these columns are used
            for _, row in movies.iterrows():
                values = tuple(row[col] for col in movie_columns)
                # Debug print
                # print(len(values), values)
                cursor.execute("""
                    INSERT INTO movies (
                        movie_id, title, release_date, video_release_date, IMDb_URL,
                        unknown, Action, Adventure, Animation, Children, Comedy, Crime,
                        Documentary, Drama, Fantasy, FilmNoir, Horror, Musical, Mystery,
                        Romance, SciFi, Thriller, War, Western
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, values)
            print("✅ Movies inserted")

            # ---------------------------- Users ----------------------------
            users = pd.read_csv(f"{data_path}/u.user", sep='|', header=None)
            users.columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
            for _, row in users.iterrows():
                cursor.execute("""
                    INSERT INTO users (user_id, age, gender, occupation, zip_code)
                    VALUES (%s, %s, %s, %s, %s)
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

# Example usage
if __name__ == "__main__":
    db_config = {
        "host": "localhost",
        "dbname": "12thwonder",
        "user": "postgres",
        "password": "admin",
        "port": "5432"
    }
    load_and_insert_movielens_data(db_config)
