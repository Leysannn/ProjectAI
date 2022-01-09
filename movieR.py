import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#Load Movies Metadata
metadata = pd.read_csv('/Users/mac/Documents/GitHub/ProjectAI/archive/movies_metadata.csv', low_memory=False)

#Removes stop words like "the", "an", etc. They give no useful info
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN values with an empty string
metadata['overview'] = metadata['overview'].fillna('')

#Contruct the required TF-IDF matrix, (45466=movies, 75827=diff. vocab)
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
