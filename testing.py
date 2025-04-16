#import cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

ratings_matrix = np.array([
    [1, 0, 3, 0, 0, 5, 0, 0, 5, 0, 4, 0],
    [0, 0, 5, 4, 0, 0, 4, 0, 0, 2, 1, 3],
    [2, 4, 0, 1, 2, 0, 3, 0, 4, 3, 5, 0],
    [0, 2, 4, 0, 5, 0, 0, 4, 0, 0, 2, 0],
    [0, 0, 4, 3, 4, 2, 0, 0, 0, 0, 2, 5],
    [1, 0, 3, 0, 3, 0, 0, 2, 0, 0, 4, 0]
])


def mean_of_movie(movie):
    non_zero_ratings = movie[movie > 0]
    return np.mean(non_zero_ratings) if non_zero_ratings.size > 0 else 0

def adjusted_movie(movie):
    mean = mean_of_movie(movie)
    return np.where(movie > 0, movie - mean, 0)

# adjust all
adjusted_movies = np.apply_along_axis(adjusted_movie, axis=1, arr=ratings_matrix)


#print("ADJUSTED VECTOR OF MOVIE 1")
#print(adjusted_movies[0])
#print("ADJUSTED VECTOR OF MOVIE 2")
#print(adjusted_movies[1])

# print og cosine similarity between two movies
#print("ORIGINAL COSINE SIMILARITY")
#print(cosine_similarity([ratings_matrix[0]], [ratings_matrix[1]])[0][0])
#print("ADJUSTED COSINE SIMILARITY")
#print(cosine_similarity([adjusted_movies[0]], [adjusted_movies[1]])[0][0])
# HAHAHAHAHA IT WORKS!!!

# select k nearest, remember to ignore itself
k = 2
user_index = 4  # User 5 (0-indexed)
movie_index = 0  # Movie 1 (0-indexed)

similarities = cosine_similarity([adjusted_movies[movie_index]], adjusted_movies)[0]
sorted_indices = np.argsort(similarities)[::-1]  # Sort in descending order
nearest_indices = [idx for idx in sorted_indices if idx != movie_index][:k]  # Exclude the movie itself
print("NEAREST MOVIES")
print(nearest_indices)
print("COSINE SIMILARITY WITH MOVIE 1 (ADJUSTED)")
for idx in nearest_indices:
    print(f"Movie 1 vs Movie {idx + 1}: {similarities[idx]}")

    
# calculate prediction for user 5, movie 1


# Prediction = (rating * weight + rating * weight + ... ) / (weight + weight + ... )
# where weight is the similarity score
# and rating is the rating of the user for that movie

numerator = 0
denominator = 0
for neighbor_index in nearest_indices:
    neighbor_rating = ratings_matrix[neighbor_index, user_index]  # User's rating for the neighbor movie
    similarity = similarities[neighbor_index]  # Similarity score with the target movie
    # print rating of movie 6, user 5
    print(f"ALERT Movie {neighbor_index + 1} rating for user {user_index + 1}: {neighbor_rating}")
    print(ratings_matrix[neighbor_index, user_index])

    if neighbor_rating > 0:  # Only consider movies rated by the user
        numerator += similarity * neighbor_rating
        denominator += similarity
        print(f"Neighbor {neighbor_index + 1} rating: {neighbor_rating}, similarity: {similarity}")

print("PREDICTION FOR USER 5, MOVIE 1")
if denominator > 0:
    prediction = numerator / denominator
    print(f"Numerator: {numerator}")
    print(f"Denominator: {denominator}")
    print(f"Predicted rating: {prediction}")
else:
    print("No valid neighbors to predict rating.")
    
    
