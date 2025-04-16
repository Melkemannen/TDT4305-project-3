import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def user_cf(rate_m, tup_mu, neigh):
    
    #transpose rate_m, i prefer working on the rows
    rate_m = rate_m.T
    # Get the user index
    user_index = tup_mu[1] -1
    movie_index = tup_mu[0] -1
    print("USER INDEX", user_index)
    print("MOVIE INDEX", movie_index)
    
    def mean_of_user(user):
        non_zero_ratings = user[user > 0]
        return np.mean(non_zero_ratings) if non_zero_ratings.size > 0 else 0
    
    def adjust_user(user):
        mean = mean_of_user(user)
        return user - mean
    
    # adjust all users
    adjusted_users = np.apply_along_axis(adjust_user, 1, rate_m)
    print("ADJUSTED USERS")
    print(adjusted_users)
    
    similarities = cosine_similarity([adjusted_users[user_index]], adjusted_users)[0]
    sorted_indices = np.argsort(similarities)[::-1]  # Sort indices by similarity in descending order
    nearest_indices = [idx for idx in sorted_indices if idx != user_index][:neigh]  # Exclude the movie itself
    print("NEAREST USERS")
    print(nearest_indices)
    print("COSINE SIMILARITY WITH USER", tup_mu[1], "(ADJUSTED)")
    for idx in nearest_indices:
        print(f"User {tup_mu[1]} vs User {idx + 1}: {similarities[idx]}")
    #calculation time
    numerator = 0
    denominator = 0
    for neighbor_index in nearest_indices:
        neighbor_rating = rate_m[neighbor_index][movie_index]
        print("NEIGHBOR RATING", neighbor_rating)
        print("NEIGHBOR INDEX", neighbor_index)
        similarity = similarities[neighbor_index]
        print("SIMILARITY", similarity)
        print("PROCEED")
        if neighbor_rating > 0:
            numerator += similarity * neighbor_rating
            denominator += similarity
            
    if denominator > 0:
        prediction = numerator / denominator
    else:
        prediction = 505050505050505005       
    
    return prediction   
# Inputs
# users on columns, movies on rows
ratings_matrix = np.array([
    [1, 0, 3, 0, 0, 5, 0, 0, 5, 0, 4, 0],
    [0, 0, 5, 4, 0, 0, 4, 0, 0, 2, 1, 3],
    [2, 4, 0, 1, 2, 0, 3, 0, 4, 3, 5, 0],
    [0, 2, 4, 0, 5, 0, 0, 4, 0, 0, 2, 0],
    [0, 0, 4, 3, 4, 2, 0, 0, 0, 0, 2, 5],
    [1, 0, 3, 0, 3, 0, 0, 2, 0, 0, 4, 0]
])

list_mu_query = [(1, 5), (3, 3)]  # movie, user
neigh = 2  # Number of neighbors

# Process each query
for mu_query in list_mu_query:
    predicted_rating = user_cf(ratings_matrix, mu_query, neigh)
    print(f"The predicted rating of movie {mu_query[0]} by user {mu_query[1]}: {predicted_rating} (User-User CF)")
    print("-----------------------------------------------------------------")