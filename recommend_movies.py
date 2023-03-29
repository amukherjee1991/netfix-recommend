'''
Netflix Movie recommendation based on features
AUTHOR: Arghya Mukherjee
Date: December 17,2021
'''

import pandas as pd
import numpy as np
import copy
import re
import math
from scipy import spatial
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

'''
Load netflix data
Dataset includes titles,country,cast,description,director etc
'''

df = pd.read_csv('netflix_titles.csv')
df.fillna('missing', inplace = True)

# listed_in is basically genre
df = df.rename(columns={"listed_in":"Genre"})
df['Genre'] = df['Genre'].apply(lambda x: x.split(",")[0])

# recommendation_cols = ['country', 'release_year', 'rating', 'duration', 'listed_in','description']

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
# This is important as stop words repeated and affects any statistical analysis
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df['description'] = df['description'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['description'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape

'''
Compute the cosine similarity matrix
Mathematically compute similarity
'''
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Returns 10 movies based on description

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]
'''
Upon running this function I see that using just description is not a good idea
There are other features like cast, director which might give us better accuracy
'''
print("Recommendation based on title")
print(get_recommendations('PK'))

'''
So I have added all the relevant features like director,cast,description,
Add the relevant features to a list
'''

features=['director','cast','description','title','country','Genre']
filters = df[features]

#Cleaning the data by making all the words in lower case.
def clean_data(x):
        return str.lower(x.replace(" ", ""))

# Apply clean data function to list of features
for feature in features:
    filters[feature] = filters[feature].apply(clean_data)

# function to
def create_soup(x):
    return x['director'] + ' ' + x['cast'] + ' ' +x['country']+' '+ x['description']+' '+ x['Genre']

filters['soup'] = filters.apply(create_soup, axis=1)


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(filters['soup'])

# Compute the Cosine Similarity matrix based on the count_matrix
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
filters=filters.reset_index()
indices = pd.Series(filters.index, index=filters['title'])

'''
Building a function to create recommendation
Based on features list
'''

def get_recommendations_new(title, cosine_sim=cosine_sim):
    title=title.replace(' ','').lower()
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]


print("Recommendation based on features")
print(get_recommendations_new('PK', cosine_sim2))
