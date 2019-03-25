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
from gensim.parsing.preprocessing import preprocess_string

def iter_documents(top_directory):
	"""Iterate over all documents within the directory, yielding a dictionary
	The keys of the output dictionary are company names and the values is a preprocessing bags of words
	"""
	doc_dict = {}
	for root, dirs, files in os.walk(top_directory):
		for file in files:
			document = open(os.path.join(root, file)).read() # read the entire document, as one big string
			document = preprocess_string(document)
			doc_dict[f'{file}'] = document
	return doc_dict

def produce_reference_corpus(dictionary):
	'''
	Dictionary should be the output from iter_documents
	The reference corpus is a list of lists
	reference_corpus[i][0] is a company name
	reference_corpus[i][0] is a bag of words
	Used later on when we need to match a company's name with its cluster
	'''
	reference_corpus = [[key, values] for key, values in dictionary.items()]
	return reference_corpus

def produce_corpus_list(dictionary):
	'''
	Dictioanry should be the ouput from iter_documents
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

	
def vis_lda(model, id2word, corpus):
	'''
	The arguments should be the output from the build_lda function
	build_lda returns a tuple (let's call this 'tup')
	vis_lda's arguments should be:
	model = tup[0]
	id2word = tup[1]
	corpus = tup[2]
	'''
	pyLDAvis.enable_notebook()
	visual = pyLDAvis.gensim.prepare(model, corpus, id2word)
	visual

def make_lda_dataframe(lda_model, corpus, corpus_list, reference_corpus):
	'''
	Returns a database containing all relevant LDA outputs (topic number and keywords for each document)
	If tup is the output from build_lda(), then lda_model = tup[0] and corpus = tup[2]
	corpus_list is output from produce_corpus_list()
	reference_corpus is output from produce_reference_corpus()
	'''
	# initialize dataframe
	doc_topic_df = pd.DataFrame()
	# Get main topic in each document
	for i, row in enumerate(lda_model[corpus]):
		row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
		# Get the Dominant topic, Perc Contribution and Keywords for each document
		for j, (topic_num, topic_perc) in enumerate(row):
			# if j == 0, then j is the dominant topic
			if j == 0:  
				# built-in Gensim method that shows keywords associated with each topic
				words = lda_model.show_topic(topic_num)
				# words will be a tuple where words[0] is a keyword and words[i] is a proportion of contrinbution of word to topic 
				topic_keywords = ", ".join([word for word, prop in words])
				doc_topic_df = doc_topic_df.append(pd.Series([int(topic_num), round(topic_perc,3), topic_keywords]), ignore_index=True)
			else:
				# We only want the dominant industry (topic) for each company (document)
				break
	doc_topic_df.columns = ['Industry', 'Percent_Contribution', 'Keywords']
	# create list of company names:
	# Need to trim the ".text" suffix for each company name
	company_names = [i[0][:-5] for i in reference_corpus]
	# Add company name to dataframe
	doc_topic_df['Company'] = company_names
	return doc_topic_df

def lda_pipeline_function(year, num_topics, random_state, update_every, chunksize, passes, alpha):
	'''
	Function that runs a LDA clustering model for the desired years
	arg: years_list: is a list of target years
	arg: num_topics: number of industries/topics/clusters
	default args: hyperparameters of Gensim LDA model
	results will be a list of tuples
	one list for each year
	results[i][0] = lda_model
	results[i][1] = corpus_list
	results[i][2] = reference_list
	results[i][3] = corpus
	results[i][4] = id2word
	results[i][5] = classification_df
	'''
	print(f'Year: {year}')
	directory_path = f'{year}'
	init_corpus = iter_documents(directory_path)
	reference_corpus = produce_reference_corpus(init_corpus)
	corpus_list = produce_corpus_list(init_corpus)	
	stemmed_list = stem_function(corpus_list)
	bigrammed_list = make_bigrams(corpus_list, stemmed_list)
	corpus_trigrams = make_trigrams(corpus_list, bigrammed_list)
	tup = build_lda(corpus_trigrams, num_topics, random_state, update_every, chunksize, passes, alpha)
	classification_df = make_lda_dataframe(tup[0],tup[2],corpus_list,reference_corpus)
	return (tup[0], corpus_list, reference_corpus, tup[2], tup[1], classification_df)

if (__name__ == '__main__'):
	import sys
	lda_pipeline_function((sys.argv[1]),(int(sys.argv[2])))
	

