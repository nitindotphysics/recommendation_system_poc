
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import time
import random
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


from surprise import Dataset, Reader, SVD
# from surprise.model_selection import train_test_split
from surprise import accuracy
# import numpy as np
# import pandas as pd
# from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')


# In[3]:
ratings = pd.read_csv('ratings_cleaned.csv')
movies = pd.read_csv('movies_cleaned.csv')
users = pd.read_csv('users_cleaned.csv')


genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 
                 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                 'Sci-Fi', 'Thriller', 'War', 'Western']


# # Build Recommender Systems 

# ## Content-Based Approach

# Description on Content-Based Recommender System
# Create feature matrix
feature_matrix = movies[genre_columns].values

# Split ratings into train and test sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Precompute user profiles and averages
user_profiles = {}
user_avg_ratings = ratings.groupby('user_id')['rating'].mean().to_dict()
global_avg = ratings['rating'].mean()

# print("Precomputing user profiles...")
start_time = time.time()

for user_id in ratings['user_id'].unique():
    user_ratings = ratings[ratings['user_id'] == user_id]
    if len(user_ratings) > 0:
        # Get indices of movies rated by this user
        movie_indices = []
        for item_id in user_ratings['item_id']:
            if item_id in movies['item_id'].values:
                item_idx = movies[movies['item_id'] == item_id].index[0]
                movie_indices.append(item_idx)

        # Calculate weighted average feature vector
        if movie_indices:
            weights = user_ratings['rating'].values
            user_profile = np.average(feature_matrix[movie_indices], axis=0, weights=weights)
            user_profiles[user_id] = user_profile

# print(f"User profiles computed in {time.time() - start_time:.2f} seconds")

# Precompute all movie vectors
movie_vectors = feature_matrix

# Prediction cache
prediction_cache = {}

cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

def predict_rating_fast(user_id, item_id):
    cache_key = f"{user_id}_{item_id}"
    if cache_key in prediction_cache:
        return prediction_cache[cache_key]

    if user_id not in user_profiles:
        result = user_avg_ratings.get(user_id, global_avg)
        prediction_cache[cache_key] = result
        return result

    if item_id not in movies['item_id'].values:
        result = user_avg_ratings.get(user_id, global_avg)
        prediction_cache[cache_key] = result
        return result

    item_idx = movies[movies['item_id'] == item_id].index[0]
    user_vec = user_profiles[user_id]
    movie_vec = movie_vectors[item_idx]

    # Calculate cosine similarity
    dot_product = np.dot(user_vec, movie_vec)
    user_norm = np.linalg.norm(user_vec)
    movie_norm = np.linalg.norm(movie_vec)

    if user_norm == 0 or movie_norm == 0:
        result = user_avg_ratings.get(user_id, global_avg)
        prediction_cache[cache_key] = result
        return result

    cosine_sim = dot_product / (user_norm * movie_norm)

    # Map cosine similarity to rating scale (1-5)
    predicted_rating = 2.5 + 2.5 * cosine_sim
    result = max(1, min(5, predicted_rating))
    prediction_cache[cache_key] = result
    return result

def evaluate_with_sampling(n_users=100, top_k=10, threshold=4):
    """
    Evaluate on a sample of users to speed up computation
    """
    # Randomly sampled users
    sampled_users = np.random.choice(test_data['user_id'].unique(), n_users, replace=False)

    precisions = []
    recalls = []

    for user_id in sampled_users:
        # Get user's test ratings (considered as relevant items)
        user_test_ratings = test_data[(test_data['user_id'] == user_id) & 
                                     (test_data['rating'] >= threshold)]
        if len(user_test_ratings) == 0:
            continue

        relevant_items = set(user_test_ratings['item_id'].values)

        # Get user's rated movies from training data
        user_rated_movies = set(train_data[train_data['user_id'] == user_id]['item_id'])

        # Get candidate movies (not rated by user)
        candidate_movies = set(movies['item_id']) - user_rated_movies

        # Predict ratings for a sample of candidate movies
        n_candidates = min(200, len(candidate_movies))  # Limit to 200 candidates
        sampled_candidates = np.random.choice(list(candidate_movies), n_candidates, replace=False)

        movie_scores = []
        for item_id in sampled_candidates:
            predicted_rating = predict_rating_fast(user_id, item_id)
            movie_scores.append((item_id, predicted_rating))

        # Sort by predicted rating and get top K
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        recommended_items = [item_id for item_id, score in movie_scores[:top_k]]

        # Calculate precision and recall
        true_positives = set(recommended_items).intersection(relevant_items)
        precision = len(true_positives) / len(recommended_items) if len(recommended_items) > 0 else 0
        recall = len(true_positives) / len(relevant_items) if len(relevant_items) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)

def evaluate_rmse_fast(n_ratings=1000):
    """
    Evaluate RMSE on a sample of ratings
    """
    # Sample ratings from test data
    sampled_ratings = test_data.sample(n=min(n_ratings, len(test_data)), random_state=42)

    predictions = []
    actuals = []

    for _, row in sampled_ratings.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        actual_rating = row['rating']

        predicted_rating = predict_rating_fast(user_id, item_id)

        predictions.append(predicted_rating)
        actuals.append(actual_rating)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return rmse

"""
# Run evaluations
print("Evaluations:\n")

# Evaluate precision and recall at different K values
for k in [5, 10, 15]:
    precision, recall = evaluate_with_sampling(top_k=k)
    print(f"Precision@{k}: {precision*100:.2f}%, Recall@{k}: {recall*100:.2f}%")

# Evaluate RMSE
rmse = evaluate_rmse_fast()
print(f"RMSE: {rmse:.2f}")
"""

def get_top_n_recommendations(user_id, n=10, exclude_rated=True):
    """
    Get top N recommendations for a user based on content similarity

    Parameters:
    user_id (int): ID of the user to get recommendations for
    n (int): Number of recommendations to return
    exclude_rated (bool): Whether to exclude movies the user has already rated

    Returns:
    list: List of recommended movie titles with predicted ratings
    """
    # Get movies the user has already rated if we need to exclude them
    rated_movies = set()
    if exclude_rated:
        user_ratings = ratings[ratings['user_id'] == user_id]
        rated_movies = set(user_ratings['item_id'].values)

    # Get all movie IDs
    all_item_ids = movies['item_id'].values

    # Calculate predicted ratings for all movies
    movie_scores = []
    for item_id in all_item_ids:
        if exclude_rated and item_id in rated_movies:
            continue

        predicted_rating = predict_rating_fast(user_id, item_id)
        movie_title = movies[movies['item_id'] == item_id]['title'].values[0]
        movie_scores.append((movie_title, predicted_rating))

    # Sort by predicted rating and get top N
    movie_scores.sort(key=lambda x: x[1], reverse=True)

    return movie_scores[:n]

# ## Collaborative Filtering
# 

# Description of this approach

# To speed up calculations, and save computation time, I will be using a subset of the original data. 70% users and 70% movies data. Given more compute resources and more time, we can definitely use the entire data for model building.
# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Sample the data for computational efficiency (use 70% of users and items)
user_sample = np.random.choice(ratings['user_id'].unique(), 
                               size=int(0.7 * len(ratings['user_id'].unique())), 
                               replace=False)
item_sample = np.random.choice(ratings['item_id'].unique(), 
                               size=int(0.7 * len(ratings['item_id'].unique())), 
                               replace=False)

ratings = ratings[ratings['user_id'].isin(user_sample) & ratings['item_id'].isin(item_sample)]


# Create user-item matrix
user_item_matrix = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Split data for evaluation (70% train, 30% test)
train_data, test_data = train_test_split(ratings, test_size=0.3, random_state=42)

# Create train and test matrices
train_matrix = train_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
test_matrix = test_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Define common evaluation function
def evaluate_predictions(predictions, actuals):
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    return rmse, mae

# Define function to get movie title
def get_movie_title(item_id):
    title = movies[movies['item_id'] == item_id]['title'].values
    return title[0] if len(title) > 0 else f"Movie {item_id}"

# Define function to generate recommendations
def generate_recommendations(user_id, model_func, n=5):
    user_ratings = user_item_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index

    predictions = []
    for item_id in unrated_movies:
        pred_rating = model_func(user_id, item_id)
        if pred_rating > 0:  # Only consider predictions that are valid
            predictions.append((item_id, pred_rating))

    # Sort by predicted rating and return top-N
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    return [(get_movie_title(item_id), rating) for item_id, rating in top_n]

# Define function to evaluate precision and recall
def evaluate_precision_recall(model_func, test_data, k=10, threshold=3.5):
    # Sample users for evaluation (for computational efficiency)
    sample_users = np.random.choice(test_data['user_id'].unique(), 
                                   size=min(50, len(test_data['user_id'].unique())), 
                                   replace=False)

    precisions = []
    recalls = []

    for user_id in sample_users:
        # Get user's test ratings that are above threshold
        user_test_ratings = test_data[(test_data['user_id'] == user_id) & (test_data['rating'] >= threshold)]
        relevant_items = set(user_test_ratings['item_id'].values)

        if len(relevant_items) == 0:
            continue  # Skip users with no relevant items in test set

        # Get user's rated items from training data
        user_train_ratings = train_data[train_data['user_id'] == user_id]
        rated_items = set(user_train_ratings['item_id'].values)

        # Generate top-k recommendations
        recommendations = []
        for item_id in user_item_matrix.columns:
            if item_id not in rated_items:  # Only consider items not rated by user
                pred_rating = model_func(user_id, item_id)
                if pred_rating > 0:  # Only consider valid predictions
                    recommendations.append((item_id, pred_rating))

        # Get top-k predicted items
        recommendations.sort(key=lambda x: x[1], reverse=True)
        top_k_recommendations = set([item_id for item_id, _ in recommendations[:k]])

        # Calculate precision and recall
        if len(top_k_recommendations) > 0:
            true_positives = len(top_k_recommendations.intersection(relevant_items))
            precision = true_positives / len(top_k_recommendations)
            recall = true_positives / len(relevant_items) if len(relevant_items) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0

    return avg_precision, avg_recall

# ## USER-BASED COLLABORATIVE FILTERING

print("\n" + "="*60)
print("USER-BASED COLLABORATIVE FILTERING")
print("="*60)

# Compute user-user similarity matrix
start_time = time.time()
print("Computing user similarities...")
user_similarity = cosine_similarity(train_matrix)
user_sim_df = pd.DataFrame(user_similarity, index=train_matrix.index, columns=train_matrix.index)
compute_time = time.time() - start_time
print(f"Similarity computation time: {compute_time:.2f} seconds")

def user_based_predict(user_id, item_id, k=10):
    if user_id not in train_matrix.index or item_id not in train_matrix.columns:
        return 0

    # Find users who rated the item
    rated_users = train_matrix.loc[train_matrix[item_id] > 0].index

    if len(rated_users) == 0:
        return 0

    # Get top-k similar users who rated the item
    sim_scores = user_sim_df.loc[user_id, rated_users].sort_values(ascending=False)[:k]

    # Remove zero or negative similarities
    sim_scores = sim_scores[sim_scores > 0]

    if len(sim_scores) == 0:
        return 0

    # Calculate weighted average
    weighted_sum = (train_matrix.loc[sim_scores.index, item_id] * sim_scores).sum()
    return weighted_sum / sim_scores.sum()

# Evaluate user-based CF
# print("Evaluating user-based collaborative filtering...")
start_time = time.time()

predictions = []
actuals = []
covered = 0

# Sample test data for evaluation (for computational efficiency)
test_sample = test_data.sample(min(1000, len(test_data)), random_state=42)

for idx, row in test_sample.iterrows():
    user_id = row['user_id']
    item_id = row['item_id']
    actual_rating = row['rating']

    pred_rating = user_based_predict(user_id, item_id)

    if pred_rating > 0:  # Only count if we could make a prediction
        predictions.append(pred_rating)
        actuals.append(actual_rating)
        covered += 1

user_based_time = time.time() - start_time
user_based_rmse, user_based_mae = evaluate_predictions(predictions, actuals)
coverage = covered / len(test_sample)

"""
print(f"RMSE: {user_based_rmse:.2f}")
print(f"MAE: {user_based_mae:.2f}")
print(f"Coverage: {coverage:.2%} ({covered}/{len(test_sample)})")
print(f"Evaluation time: {user_based_time:.2f} seconds")

# Evaluate precision and recall
print("Evaluating precision and recall...")
user_precision, user_recall = evaluate_precision_recall(user_based_predict, test_data)
print(f"Precision@10: {user_precision:.2f}")
print(f"Recall@10: {user_recall:.2f}")
"""

# Generate sample recommendations
sample_user = test_sample['user_id'].iloc[0]
# print(f"\nTop 5 recommendations for user {sample_user}:")
recommendations = generate_recommendations(sample_user, user_based_predict)
for i, (title, rating) in enumerate(recommendations, 1):
    print(f"{i}. {title} (predicted rating: {rating:.2f})")


# ## ITEM-BASED COLLABORATIVE FILTERING

# Compute item-item similarity matrix
start_time = time.time()
# print("Computing item similarities...")
item_user_matrix = train_matrix.T
item_similarity = cosine_similarity(item_user_matrix)
item_sim_df = pd.DataFrame(item_similarity, index=item_user_matrix.index, columns=item_user_matrix.index)
compute_time = time.time() - start_time
# print(f"Similarity computation time: {compute_time:.2f} seconds")

def item_based_predict(user_id, item_id, k=10):
    if user_id not in train_matrix.index or item_id not in train_matrix.columns:
        return 0

    # Get user's ratings
    user_ratings = train_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index

    if len(rated_items) == 0:
        return 0

    # Get top-k similar items rated by the user
    sim_scores = item_sim_df.loc[item_id, rated_items].sort_values(ascending=False)[:k]

    # Remove zero or negative similarities
    sim_scores = sim_scores[sim_scores > 0]

    if len(sim_scores) == 0:
        return 0

    # Calculate weighted average
    weighted_sum = (user_ratings[sim_scores.index] * sim_scores).sum()
    return weighted_sum / sim_scores.sum()


# Evaluate item-based CF
# print("Evaluating item-based collaborative filtering...")
start_time = time.time()

predictions = []
actuals = []
covered = 0

for idx, row in test_sample.iterrows():
    user_id = row['user_id']
    item_id = row['item_id']
    actual_rating = row['rating']

    pred_rating = item_based_predict(user_id, item_id)

    if pred_rating > 0:  # Only count if we could make a prediction
        predictions.append(pred_rating)
        actuals.append(actual_rating)
        covered += 1

item_based_time = time.time() - start_time
item_based_rmse, item_based_mae = evaluate_predictions(predictions, actuals)
coverage = covered / len(test_sample)
"""
print(f"RMSE: {item_based_rmse:.2f}")
print(f"MAE: {item_based_mae:.2f}")
print(f"Coverage: {coverage:.2%} ({covered}/{len(test_sample)})")
print(f"Evaluation time: {item_based_time:.2f} seconds")

# Evaluate precision and recall
print("Evaluating precision and recall...")
item_precision, item_recall = evaluate_precision_recall(item_based_predict, test_data)
print(f"Precision@10: {item_precision*100:.2f}")
print(f"Recall@10: {item_recall*100:.2f}")
"""

# Generate sample recommendations
# print(f"\nTop 5 recommendations for user {sample_user}:")
recommendations = generate_recommendations(sample_user, item_based_predict)
for i, (title, rating) in enumerate(recommendations, 1):
    print(f"{i}. {title} (predicted rating: {rating:.2f})")

# ## MODEL-BASED COLLABORATIVE FILTERING (SVD)

# Prepare the data for Surprise

# We only need user_id, item_id, and rating for the basic SVD model
ratings_data = ratings[['user_id', 'item_id', 'rating']]

# Define the rating scale
reader = Reader(rating_scale=(1, 5))

# Load data from DataFrame
data = Dataset.load_from_df(ratings_data, reader)


# Step 2: Split data into train and test sets using surprise library
from surprise.model_selection import train_test_split

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Step 3: Train the SVD model
# For better efficiency, you can reduce n_factors or n_epochs
algo = SVD(n_factors=50, n_epochs=20, random_state=42)  # Reduced factors for efficiency
algo.fit(trainset)

# Step 4: Evaluate RMSE and MAE
predictions = algo.test(testset)
svd_rmse = accuracy.rmse(predictions, verbose=False)
svd_mae = accuracy.mae(predictions, verbose=False)
print(f'RMSE: {svd_rmse:.2f}, MAE: {svd_mae:.2f}')

# Step 5: Compute Precision and Recall
threshold = 4.0  # Consider items with rating >= 4 as relevant
top_n = 10       # Top-N recommendations

# Map predictions to each user
user_est_true = defaultdict(list)
for uid, iid, true_r, est, _ in predictions:
    user_est_true[uid].append((est, true_r))

precisions = []
recalls = []

for uid, user_ratings in user_est_true.items():
    # Sort user ratings by estimated value
    user_ratings.sort(key=lambda x: x[0], reverse=True)

    # Number of relevant items (true rating >= threshold)
    n_rel = sum(true_r >= threshold for (_, true_r) in user_ratings)

    # Number of recommended items in top-N
    n_rec_k = sum(est >= threshold for (est, _) in user_ratings[:top_n])

    # Number of relevant and recommended items
    n_rel_and_rec_k = sum(
        (true_r >= threshold) and (est >= threshold)
        for (est, true_r) in user_ratings[:top_n]
    )

    # Precision and Recall for this user
    precisions.append(n_rel_and_rec_k / top_n if top_n != 0 else 0)
    recalls.append(n_rel_and_rec_k / n_rel if n_rel != 0 else 0)

# Average precision and recall over all users
svd_precision = np.mean(precisions)
svd_recall = np.mean(recalls)
# print(f'Precision@{top_n}: {svd_precision*100:.2f}%, Recall@{top_n}: {svd_recall*100:.2f}%')


# Optional: Generate recommendations for a specific user
def get_top_n_recommendations(algo, user_id, n=10):
    # Get a list of all item IDs
    all_items = ratings['item_id'].unique()

    # Get items already rated by the user
    rated_items = ratings[ratings['user_id'] == user_id]['item_id'].values

    # Get items not rated by the user
    items_to_predict = [item for item in all_items if item not in rated_items]

    # Predict ratings for all items not rated by the user
    testset = [[user_id, item_id, 4.] for item_id in items_to_predict]
    predictions = algo.test(testset)

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Return top N recommendations
    top_n = predictions[:n]

    # Get movie titles for the recommendations
    top_n_with_titles = []
    for pred in top_n:
        movie_title = movies[movies['item_id'] == pred.iid]['title'].values[0]
        top_n_with_titles.append((pred.iid, movie_title, pred.est))

    return top_n_with_titles

# Example: Get top 10 recommendations for user 1
user_id = 1
top_recommendations = get_top_n_recommendations(algo, user_id)
print(f"\nTop 10 recommendations for user {user_id}:")
for i, (item_id, title, rating) in enumerate(top_recommendations, 1):
    print(f"{i}. {title} (predicted rating: {rating:.2f})")

"""
# ## COMPARATIVE ANALYSIS

print("\n" + "="*60)
print("COMPARATIVE ANALYSIS")
print("="*60)

# Create comparison table
comparison = pd.DataFrame({
    'Approach': ['User-Based', 'Item-Based', 'SVD'],
    'RMSE': [user_based_rmse, item_based_rmse, svd_rmse],
    'MAE': [user_based_mae, item_based_mae, svd_mae],
    'Precision@10': [user_precision, item_precision, svd_precision],
    'Recall@10': [user_recall, item_recall, svd_recall]})

print(comparison)

"""
