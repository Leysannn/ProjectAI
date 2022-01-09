import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
import numpy as np

path_elliot = '/Users/elliotmartin/Documents/KTH/Cinte/ID1214/Project/ProjectAI/archive/'

#Load Movies Metadata
metadata = pd.read_csv('/Users/mac/Documents/GitHub/ProjectAI/archive/movies_metadata.csv', low_memory=False)
#Load credits
credits = pd.read_csv('/Users/mac/Documents/GitHub/ProjectAI/archive/credits.csv')


#Drop the rows containing bad IDs
metadata = metadata.drop([19730, 29503, 35587])

#Convert IDs to integers. We will need this later when merging
metadata['id'] = metadata['id'].astype('int')
credits['id'] = credits['id'].astype('int')


#Now we merge credits into our main dataframe metadata
metadata = metadata.merge(credits, on='id')


#Convert our data into usable form
features = ['cast', 'crew', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

#Retrives the directors name of the searched movie from crew feature
#Returns NaN if not listed
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


#Retrives the 3 top actors names of the searched movie from cast feature
#If there is less then 3 actors listed, return them instead
#Returns empty list if not listed
def get_top_actors(x):
    try:
        names = [i['name'] for i in x]

        if len(names) > 3:
            names = names[:3]
        return names

    except: 
        return []

#Retrives the 3 top genres of the searched movie from genres feature
#If there is less then 3 genres listed, return them instead
#Returns empty list if not listed
def get_top_genres(x):
    try:
        names = [i['name'] for i in x]

        if len(names) > 3:
            names = names[:3]
        return names

    except: 
        return [] 

#Define director, cast and genres features again in suitable form
metadata['director'] = metadata['crew'].apply(get_director)
metadata['cast'] = metadata['cast'].apply(get_top_actors)
metadata['genres'] = metadata['genres'].apply(get_top_genres)

#Prints the features for the first 3 films
#print(metadata[['title', 'cast', 'director', 'genres']].head(3))


#Converting the names into lowercase and remove the spaces between them
#Converting all names into compound names
def convert_names(x):
    #Checks names in cast and genres
    if isinstance(x, list):
        new_names = [str.lower(i.replace(" ", "")) for i in x]
        return new_names
    #Checks name in director
    else:
        if isinstance(x, str):
            new_name = str.lower(x.replace(" ", ""))
            return new_name
        else:
            return ''



#Apply convert_names function to your features.
features = ['cast', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(convert_names)

#Join our features into a string so that we can insert it to out vectorizer
def join_features(x):
    string = ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
    return string

#Create a new feature for join_features function
metadata['join'] = metadata.apply(join_features, axis=1)

#Removes stop words like "the", "an", etc. They give no useful info
count = CountVectorizer(stop_words='english')
#Contruct the required CountVectorizer matrix (46628=movies, 73881=diff. vocab)
count_matrix = count.fit_transform(metadata['join'])

#Computes the cosince similarity matrix
#Cosine similarity: Independent of magnitude, easy, fast to calculate
cosine_sim = cosine_similarity(count_matrix, count_matrix)

#Construct a reverse map of indices and movie titles
#mechanism to identify the index of a movie in our data, given its title
indices = pd.Series(metadata.index, index=metadata['title'])

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