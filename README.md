# Movie Recommendation System

A comprehensive movie recommendation system built with Streamlit, featuring collaborative filtering, data analysis, and interactive visualizations.

## 🚀 Features

- **📊 Dashboard**: Overview of dataset statistics and key metrics
- **📈 Data Analysis**: Interactive visualizations of ratings, genres, and user demographics
- **🎯 Movie Recommendations**: Personalized movie recommendations using collaborative filtering
- **📊 Model Evaluation**: Performance metrics (RMSE, MAE) with customizable parameters
- **⚙️ Settings**: Application configuration and information

## 📋 Prerequisites

- Python 3.8 or higher
- MovieLens 100k dataset (included in `data/ml-100k/` directory)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Movie_recommendation
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎬 Dataset Setup

Ensure the MovieLens 100k dataset is in the `data/ml-100k/` directory with the following files:
- `u.data` - Ratings data
- `u.item` - Movie information
- `u.user` - User information

## 🚀 Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## 📁 Project Structure

```
Movie_recommendation/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   ├── load_movielens.py          # Data loading utilities
│   └── ml-100k/                   # MovieLens dataset
├── data_analysis/
│   └── analysis.py                # Data analysis functions
├── logic/
│   ├── colloborative_filtering.py # Recommendation algorithm
│   ├── Calculating_accuracy_py    # Model evaluation
│   └── db_utils.py                # Database utilities
└── venv/                          # Virtual environment
```

## 🎯 Usage

### Dashboard
- View key metrics and statistics
- Quick insights into popular movies and rating distribution

### Data Analysis
- **Rating Distribution**: Analyze how users rate movies
- **Popular Movies**: See which movies are most rated
- **Genre Analysis**: Compare average ratings across genres
- **User Demographics**: Explore user age and gender distributions

### Movie Recommendations
1. Select a user ID from the dropdown
2. Choose the number of recommendations
3. Click "Generate Recommendations" to get personalized movie suggestions

### Model Evaluation
1. Adjust test size, number of neighbors, and random state
2. Click "Run Evaluation" to see model performance metrics
3. View RMSE and MAE scores with performance interpretation

## 🔧 Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Database**: PostgreSQL (optional)

## 📊 Model Details

The system uses **Collaborative Filtering** with:
- **Similarity Metric**: Cosine Similarity
- **Prediction Method**: Weighted Average
- **Evaluation Metrics**: RMSE (Root Mean Square Error) and MAE (Mean Absolute Error)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

---

**Made with ❤️ using Streamlit | MovieLens 100k Dataset | Collaborative Filtering**