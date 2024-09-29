import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

submission_files= [ file for file in os.listdir() if file.endswith('.txt')]
submission_data=[open(file,encoding='utf-8').read() for file in submission_files]


def tfdif(text): 
    return TfidfVectorizer().fit_transform(text).toarray()

def compare(file1,file2):

    return cosine_similarity([file1,file2])


vectors= tfdif(submission_data)

name_vectors=list(zip(submission_files,vectors))

result=set()

def checker():

    global vectors
    for submission_a, text_a in name_vectors:
        copy_vectors=name_vectors.copy()
        current =name_vectors.index((submission_a,text_a))
        del copy_vectors[current]
        for submission_b, text_b in copy_vectors:
            similarity = compare(text_a,text_b)[0][1]
            pair=sorted((submission_a,submission_b))

            final=(pair[0],pair[1],similarity)
            result.add(final)
    return result        

for data in checker():
    print ("Similarity:\n", data)