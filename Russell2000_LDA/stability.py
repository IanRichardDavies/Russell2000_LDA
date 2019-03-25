# Import required packages
import re
import numpy as np
import pandas as pd
from pprint import pprint
import os
import nltk
nltk.download('stopwords')
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from os import listdir
from os.path import isfile, join
from gensim.parsing.preprocessing import preprocess_string

def create_set_companies(list_of_years):
	# initialize set
	first_year = list_of_years[0]
	path = f'{first_year}'
	companies_set = {f for f in listdir(path) if isfile(join(path, f))}
	# iterate over the rest
	for i in range(1,len(list_of_years)):
		path = f'{list_of_years[i]}'
		companies_temp = {f for f in listdir(path) if isfile(join(path, f))}
		companies_set = companies_set.intersection(companies_temp)
	return companies_set
	
def iter_hist_documents(year, companies_set):
	doc_dict = {}
	for company in companies_set:
		for root, dirs, files in os.walk(f'C:\\Users\\ian_d\\OneDrive\\Desktop\\Capstone_files\\scrape_and_clean_test\\{year}'):
			for file in files:
				if str(file) == str(company):
					document = open(os.path.join(root, file)).read() # read the entire document, as one big string
					document = preprocess_string(document)
					doc_dict[f'{file}'] = document
	return doc_dict

def produce_reference_corpus(dictionary):
	'''
	Dictionary should be the output from iter_hist_documents
	The reference corpus is a list of lists
	reference_corpus[i][0] is a company name
	reference_corpus[i][0] is a bag of words
	Used later on when we need to match a company's name with its cluster
	'''
	reference_corpus = [[key, values] for key, values in dictionary.items()]
	return reference_corpus

def produce_corpus_list(dictionary):
	'''
	Dictionary should be the ouput from iter_documents
	The corpus_list is a list of lists
	corpus_list[i] is a list of words
	'''
	corpus_list = [values for key, values in dictionary.items()]
	return corpus_list

def stem_function(corpus_list): 
	'''
	Uses nltk stemmer to stem words in corpus_list
	corpus_list is a list of lists (nested lists are lists of words)
	'''
	stemmer = nltk.stem.PorterStemmer()
	stemmed_list = []
	for group in corpus_list:
		nested_list = []
		for word in group:
			nested_list.append(stemmer.stem(word))
		stemmed_list.append(nested_list)
	return stemmed_list

def make_bigrams(prestemmed_list, stemmed_list):
	# Build the bigram models
	bigram = gensim.models.Phrases(prestemmed_list, min_count=5, threshold=10) 
	# This may seem redundant but will speed up computation
	bigram_mod = gensim.models.phrases.Phraser(bigram)
	return [bigram_mod[doc] for doc in stemmed_list]

def make_trigrams(prestemmed_list, bigrammed_list):
	'''
	First argument should be the same unstemmed list that was passed into make_bigrams function
	Second argument should be the output list from the make_bigrams function
	'''
	bigram = gensim.models.Phrases(prestemmed_list, min_count=5, threshold=10) 
	trigram = gensim.models.Phrases(bigram[prestemmed_list], min_count=5, threshold=10) 
	# This may seem redundant but will speed up computation
	trigram_mod = gensim.models.phrases.Phraser(trigram)
	return [trigram_mod[doc] for doc in bigrammed_list]

def build_lda(corpus_trigrams, num_topics, random_state = 100, update_every = 1, chunksize = 100, passes = 10, alpha = 'auto'):
	'''
	corpus_trigrams (first arg) should be the output from make_trigrams
	num_topics: the desired number of clusters
	Other default arguments are hyperparameters of Gensim's LDA model
	'''
	# Create Dictionary
	id2word = corpora.Dictionary(corpus_trigrams)
	# Term Document Frequency
	corpus = [id2word.doc2bow(text) for text in corpus_trigrams]
	# Build LDA model
	lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
										   id2word=id2word,
										   num_topics=num_topics, 
										   random_state=random_state,
										   update_every=update_every,
										   chunksize=chunksize,
										   passes=passes,
										   alpha=alpha,
										   per_word_topics=True)
	return (lda_model, id2word, corpus)

def label_companies(lda_model, corpus):
	'''
	arg: lda_model: if result = build_lda(), then lda_model = result[0]
	arg: corpus: if result = build_lda(), then corpus = result[2]
	Returns a list containing industry labels for each company
	'''
	# initialize list
	labels = []
	# Get main topic in each document
	for i, row in enumerate(lda_model[corpus]):
		row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
		# Get the Dominant topic, Perc Contribution and Keywords for each document
		for j, (topic_num, topic_perc) in enumerate(row):
			# Because row is sorted, if j == 0, then j is the dominant topic,
			if j == 0:  
				labels.append(topic_num)
			else:
				# We only want the dominant industry (topic) for each company (document)
				break
	return labels

def stability_pipeline_function(year, num_topics, companies_set):
	'''
	Pipeline function that runs the full LDA process
	'''
	init_corpus = iter_hist_documents(year, companies_set)
	reference_corpus = produce_reference_corpus(init_corpus)
	corpus_list = produce_corpus_list(init_corpus)	
	stemmed_list = stem_function(corpus_list)
	bigrammed_list = make_bigrams(corpus_list, stemmed_list)
	corpus_trigrams = make_trigrams(corpus_list, bigrammed_list)
	tup = build_lda(corpus_trigrams, num_topics, random_state = 100, update_every = 1, chunksize = 100, passes = 10, alpha = 'auto')
	labels = label_companies(tup[0],tup[2])
	return labels

def stability(years_list, num_topics):
	'''
	Function that returns adjusted rand score for each sequential pairs of years in years_list
	arg: years_list: list of targeted years, should only include years entered into nmf() call
	arg: num_topics: should be the same argument as entered into nmf() call 
	arg: results: is the full results from nmf() call 
	len(results) should equal number of years
	results[i][0] = lda_model
	results[i][1] = corpus_list
	results[i][2] = reference_list
	results[i][3] = corpus
	results[i][4] = id2word
	results[i][5] = classification_df
	'''
	companies_set = create_set_companies(years_list)
	from sklearn.metrics import adjusted_rand_score
	labels_list = []
	for year in years_list:
		labels = stability_pipeline_function(year, num_topics, companies_set)
		print(f'Completed {year}')
		labels_list.append(labels)
	scores = []
	for i in range(len(labels_list)-1):
		true = labels_list[i]
		pred = labels_list[i+1]
		score = adjusted_rand_score(true, pred)
		scores.append(score)
	return scores

if (__name__ == '__main__'):
	import sys
	stability((sys.argv[1]),(int(sys.argv[2])))
	

