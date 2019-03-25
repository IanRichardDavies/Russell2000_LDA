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
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim 
import warnings
warnings.filterwarnings("ignore")
from gensim.parsing.preprocessing import preprocess_string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from os import listdir
from os.path import isfile, join
import datetime
import requests
import csv
import Russell2000_LDA.run_lda as run_lda
import Russell2000_LDA.sentiment as sentiment
import Russell2000_LDA.stability as stability	

def lda(year, num_topics, random_state = 100, update_every = 1, chunksize = 100, passes = 10, alpha = 'auto'):
	'''
	Function that runs a LDA clustering model for the desired years
	arg: year: target year 
	arg: num_topics: number of industries/topics/clusters
	default args: hyperparameters of Gensim LDA model
	results will be a list of tuples
	one list for each year
	results[0] = lda_model
	results[1] = corpus_list
	results[2] = reference_list
	results[3] = corpus
	results[4] = id2word
	results[5] = classification_df
	'''
	results = run_lda.lda_pipeline_function(year, num_topics, random_state, update_every, chunksize, passes, alpha)
	print('Output tuple: ')
	print('results[0] = Gensim LDA model instance')
	print('results[1] = corpus_list')
	print('results[2] = reference_list')
	print('results[3] = corpus')
	print('results[4] = id2word')
	print('results[5] = classification_df')
	return results

def display(results):
	'''
	Function that display results of the clustering
	arg: results: should a flat tuple of a single year's output from lda() 
	'''
	return results[5]

def industry_sentiment(year, num_topics, results):
	'''
	Function that return the overall sentiment for each industry
	arg: year: should only be a year included in the nmf() call
	arg: num_topics: should be the the same argument entered for nmf() call
	'''	
	result = sentiment.industry(year, num_topics, results)
	return result

def company_sentiment(year, num_topics, results):
	'''
	Function that return the overall sentiment for each industry
	arg: year: should only be a year included in the nmf() call
	arg: num_topics: should be the the same argument entered for nmf() call
	arg: results: should be a flat tuple of a single year's output from nmf()
	'''	
	result = sentiment.company(year, num_topics, results)
	return result

if (__name__ == '__main__'):
	import sys
	lda((sys.argv[1]),(int(sys.argv[2])))