import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
import numpy as np

#Load Movies Metadata
metadata = pd.read_csv('/Users/mac/Documents/GitHub/CBMRS/archive/movies_metadata.csv', low_memory=False)
#Load credits
credits = pd.read_csv('/Users/mac/Documents/GitHub/CBMRS/archive/credits.csv')
#Load keywords
keywords = pd.read_csv('/Users/mac/Documents/GitHub/CBMRS/archive/keywords.csv')

#Drop the rows containing bad IDs
metadata = metadata.drop([19730, 29503, 35587])

#Convert IDs to int. Required for later merging
metadata['id'] = metadata['id'].astype('int')
credits['id'] = credits['id'].astype('int')
keywords['id'] = keywords['id'].astype('int')

#Now we merge credits & keywords into our main dataframe metadata
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

#Convert our data into usable form
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

#Retrives the directors name of the searched movie from crew feature
#Returns NaN if not listed
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


#Return top 3 elements or the whole list
def get_list(x):
    if isinstance(x, list):








#Removes stop words like "the", "an", etc. They give no useful info
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN values with an empty string
metadata['overview'] = metadata['overview'].fillna('')

#Contruct the required TF-IDF matrix, (45466=movies, 75827=different vocab)
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#Computes the cosince similarity matrix
#Cosine similarity: Independent of magnitude, easy, fast to calculate (espacially with TF-IDF)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
#mechanism to identify the index of a movie in our data, given its title
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim):
    
    #Retrive index of given movie title
    index = indices[title]

    #Retrives a list of cosine similarity scores for our movie
    #list of tuples, [position, similarity score]
    sim_scores = list(enumerate(cosine_sim[index]))

    #Sort the list based on similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    #Get top 10 movies in the list, ignoring first element since its our searched movie
    #get the scores
    sim_scores = sim_scores[1:11]
    #get the indices
    movie_indices = [i[0] for i in sim_scores]

    return metadata['title'].iloc[movie_indices]



print(get_recommendations('The Dark Knight Rises'))

