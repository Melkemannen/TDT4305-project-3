from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def item_cf(ratings_matrix, mu_query, neigh):
    def mean_of_movie(movie):
        non_zero_ratings = movie[movie > 0]
        return np.mean(non_zero_ratings) if non_zero_ratings.size > 0 else 0

    def adjusted_movie(movie):
        mean = mean_of_movie(movie)
        return np.where(movie > 0, movie - mean, 0)

    # Adjust all movies
    adjusted_movies = np.apply_along_axis(adjusted_movie, axis=1, arr=ratings_matrix)

    # Extract user and movie indices from the query
    movie_index = mu_query[0] - 1  # Convert to 0-indexed
    user_index = mu_query[1] - 1  # Convert to 0-indexed

    # Compute cosine similarities
    similarities = cosine_similarity([adjusted_movies[movie_index]], adjusted_movies)[0]
    sorted_indices = np.argsort(similarities)[::-1]  # Sort in descending order
    nearest_indices = [idx for idx in sorted_indices if idx != movie_index][:neigh]  # Exclude the movie itself

    print("NEAREST MOVIES")
    print(nearest_indices)
    print("COSINE SIMILARITY WITH MOVIE", mu_query[0], "(ADJUSTED)")
    for idx in nearest_indices:
        print(f"Movie {mu_query[0]} vs Movie {idx + 1}: {similarities[idx]}")

    # Calculate prediction for the given user and movie
    numerator = 0
    denominator = 0
    for neighbor_index in nearest_indices:
        neighbor_rating = ratings_matrix[neighbor_index, user_index]  # User's rating for the neighbor movie
        similarity = similarities[neighbor_index]  # Similarity score with the target movie

        if neighbor_rating > 0:  # Only consider movies rated by the user
            numerator += similarity * neighbor_rating
            denominator += similarity
            print(f"Neighbor {neighbor_index + 1} rating: {neighbor_rating}, similarity: {similarity}")

    print(f"PREDICTION FOR USER {mu_query[1]}, MOVIE {mu_query[0]}")
    if denominator > 0:
        prediction = numerator / denominator
        print(f"Numerator: {numerator}")
        print(f"Denominator: {denominator}")
        print(f"Predicted rating: {prediction}")
    else:
        print("No valid neighbors to predict rating.")
        prediction = None

    return prediction


# Inputs
ratings_matrix = np.array([
    [1, 0, 3, 0, 0, 5, 0, 0, 5, 0, 4, 0],
    [0, 0, 5, 4, 0, 0, 4, 0, 0, 2, 1, 3],
    [2, 4, 0, 1, 2, 0, 3, 0, 4, 3, 5, 0],
    [0, 2, 4, 0, 5, 0, 0, 4, 0, 0, 2, 0],
    [0, 0, 4, 3, 4, 2, 0, 0, 0, 0, 2, 5],
    [1, 0, 3, 0, 3, 0, 0, 2, 0, 0, 4, 0]
])

list_mu_query = [(1, 5), (3, 3)]  # List of movie-user queries
neigh = 2  # Number of neighbors

# Process each query
for mu_query in list_mu_query:
    predicted_rating = item_cf(ratings_matrix, mu_query, neigh)
    print(f"The predicted rating of movie {mu_query[0]} by user {mu_query[1]}: {predicted_rating} (Item-Item CF)")
    print("-----------------------------------------------------------------")