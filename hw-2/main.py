# import
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np



def calcPrecisionRecall(input, index):
    f = open("./r/"+str(index)+".txt")
    result = f.read().split()
    tp = 0 # true positive
    fn = 0 # false negative

    for value in input:
        if str(value) in result:
            tp += 1

    for value in result:
        if int(value) not in input:
            fn += 1

    precision = float(tp) / len(input)
    recall = float(tp) / (tp + fn)
    f_measure = 0
    if recall != 0 and precision != 0:
        f_measure = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f_measure

# prepare corpus
corpus = []

tfidfcos = []
tfcos = []
bincos = []

tfidfeu = []
tfeu = []
bineu = []

tfidfCosinePrecision = 0.0
tfCosinePrecision = 0.0
tfBinaryCosinePrecision = 0.0
tfidfEuclideanPrecision  = 0.0
tfEuclideanPrecision  = 0.0
tfBinaryEuclideanPrecision = 0.0

tfidfCosineRecall = 0.0
tfCosineRecall = 0.0
tfBinaryCosineRecall = 0.0
tfidfEuclideanRecall  = 0.0
tfEuclideanRecall  = 0.0
tfBinaryEuclideanRecall = 0.0

tfidfCosineFmeasure = 0.0
tfCosineFmeasure = 0.0
tfBinaryCosineFmeasure = 0.0
tfidfEuclideanFmeasure  = 0.0
tfEuclideanFmeasure  = 0.0
tfBinaryEuclideanFmeasure = 0.0

for d in range(1400):
    f = open("./d/"+str(d+1)+".txt")
    corpus.append(f.read())

# init vectorizer
tfidf_vectorizer = TfidfVectorizer()

# add query to corpus
for q in range(224):
    f = open("./q/"+str(q + 1)+".txt")
    corpus.append(f.read())

    # enable IDF reweighting. Disable binary output. Normalize term vectors
    tfidf_vectorizer.use_idf = True
    tfidf_vectorizer.binary = False
    tfidf_vectorizer.norm = u'l2'

    # prepare tf-idf matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # compute similarity between query and all docs (tf-idf) and get top 10 relevant
    c_sim = np.array(cosine_similarity(tfidf_matrix[len(corpus)-1], tfidf_matrix[0:(len(corpus)-1)])[0])
    topRelevantTfidf = c_sim.argsort()[-10:][::-1]+1
    # compute distances between query and all docs (tf-idf) and get top 10 relevant
    distance =  np.array(euclidean_distances(tfidf_matrix[len(corpus)-1], tfidf_matrix[0:(len(corpus)-1)])[0])
    topRelevantTfidfDistance = distance.argsort()[::-1][-10:]+1

    # disable IDF reweighting.
    tfidf_vectorizer.use_idf = False
    #prepare tf matrix
    tf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # compute similarity between query and all docs (tf) and get top 10 relevant
    c_sim = np.array(cosine_similarity(tf_matrix[len(corpus)-1], tf_matrix[0:(len(corpus)-1)])[0])
    topRelevantTf = c_sim.argsort()[-10:][::-1]+1
        # compute distances between query and all docs (tf) and get top 10 relevant
    distance =  np.array(euclidean_distances(tf_matrix[len(corpus)-1], tf_matrix[0:(len(corpus)-1)])[0])
    topRelevantTfDistance = distance.argsort()[::-1][-10:]+1

    # set tf term in tf-idf to binary. set idf (previous step) and normalization to False to get 0/1 outputs.
    tfidf_vectorizer.binary = True
    tfidf_vectorizer.norm = False
    #prepare binary matrix
    tf_binary_matrix = tfidf_vectorizer.fit_transform(corpus)

    # compute similarity between query and all docs (binary) and get top 10 relevant
    c_sim = np.array(cosine_similarity(tf_binary_matrix[len(corpus)-1], tf_binary_matrix[0:(len(corpus)-1)])[0])
    topRelevantTfBinary = c_sim.argsort()[-10:][::-1]+1
    # compute distances between query and all docs (tf-idf) and get top 10 relevant
    distance =  np.array(euclidean_distances(tf_binary_matrix[len(corpus)-1], tf_binary_matrix[0:(len(corpus)-1)])[0])
    topRelevantTfBinaryfDistance = distance.argsort()[::-1][-10:]+1

    precision, recall, f_measure = calcPrecisionRecall(topRelevantTfidf, q + 1)
    tfidfCosinePrecision += precision
    tfidfCosineRecall += recall
    tfidfCosineFmeasure += f_measure
    tfidfcos.append([precision, recall, f_measure])

    precision, recall, f_measure = calcPrecisionRecall(topRelevantTf, q + 1)
    tfCosinePrecision += precision
    tfCosineRecall += recall
    tfCosineFmeasure += f_measure
    tfcos.append([precision, recall, f_measure])

    precision, recall, f_measure = calcPrecisionRecall(topRelevantTfBinary, q + 1)
    tfBinaryCosinePrecision += precision
    tfBinaryCosineRecall += recall
    tfBinaryCosineFmeasure += f_measure
    bincos.append([precision, recall, f_measure])

    precision, recall, f_measure = calcPrecisionRecall(topRelevantTfidfDistance, q + 1)
    tfidfEuclideanPrecision += precision
    tfidfEuclideanRecall += recall
    tfidfEuclideanFmeasure += f_measure
    tfidfeu.append([precision, recall, f_measure])

    precision, recall, f_measure = calcPrecisionRecall(topRelevantTfDistance, q + 1)
    tfEuclideanPrecision += precision
    tfEuclideanRecall += recall
    tfEuclideanFmeasure += f_measure
    tfeu.append([precision, recall, f_measure])

    precision, recall, f_measure = calcPrecisionRecall(topRelevantTfBinaryfDistance, q + 1)
    tfBinaryEuclideanPrecision += precision
    tfBinaryEuclideanRecall += recall
    tfBinaryEuclideanFmeasure += f_measure
    bineu.append([precision, recall, f_measure])

    corpus.pop();


print("The quality and difference of both scores and different weighting schemas:")
print("")
print("TF-IDF, Cosine similarity measure: Precision - " + str(tfidfCosinePrecision / 225) + ", Recall - " + str(tfidfCosineRecall / 225) + ", F-Measure - " + str(tfidfCosineFmeasure / 225))
print("Pure Term Frequency, Cosine similarity measure: Precision - " + str(tfCosinePrecision / 225) + ", Recall - " + str(tfCosineRecall / 225) + ", F-Measure - " + str(tfCosineFmeasure / 225) )
print("Binary representation, Cosine similarity measure: Precision - " + str(tfBinaryCosinePrecision / 225) + ", Recall - " + str(tfBinaryCosineRecall / 225) + ", F-Measure - " + str(tfBinaryCosineFmeasure / 225))
print("")
print("TF-IDF, Euclidean distance: Precision - " + str(tfidfEuclideanPrecision / 225) + ", Recall - " + str(tfidfEuclideanRecall / 225) + ", F-Measure - " + str(tfidfEuclideanFmeasure / 225))
print("Pure Term Frequency, Euclidean distance: Precision - " + str(tfEuclideanPrecision / 225) + ", Recall - " + str(tfEuclideanRecall / 225) + ", F-Measure - " + str(tfEuclideanFmeasure / 225))
print("Binary representation, Euclidean distance: Precision - " + str(tfBinaryEuclideanPrecision / 225) + ", Recall - " + str(tfBinaryEuclideanRecall / 225) + ", F-Measure - " + str(tfBinaryEuclideanFmeasure / 225))
print("")


np.savetxt("tfidfcos.csv", tfidfcos, delimiter=",")
np.savetxt("tfcos.csv", tfcos, delimiter=",")
np.savetxt("bincos.csv", bincos, delimiter=",")

np.savetxt("tfidfeu.csv", tfidfeu, delimiter=",")
np.savetxt("tfeu.csv", tfeu, delimiter=",")
np.savetxt("bineu.csv", bineu, delimiter=",")
