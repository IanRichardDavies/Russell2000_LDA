# import required packages
import datetime
import requests
import csv
import numpy as np
import pandas as pd
import re
import os

def get_master_idx(year):
	'''
	The master_idx is a document that acts as a map of the EDGAR database.
	Each calendar has its own master_idx
	This function acquires the urls that link to the master_idx files associated with our target year
	'''
	quarters = ['QTR1', 'QTR2', 'QTR3', 'QTR4']
	history = [(year, q) for q in quarters]
	urls = ['https://www.sec.gov/Archives/edgar/full-index/%d/%s/master.idx' % (x[0], x[1]) for x in history]
	urls.sort()
	return urls

def create_accession_nums_dataframe(urls):
	'''
	Takes in a list of urls that link to the target year master_idx files
	Outputs a pandas dataframe that contains the revelent information for all 10-K filings within our target year
	Accession numbers are the identification numbers of each individual financial filing
	'''
	records = []
	# pull all the individual lines from the four master.idx files into one list
	for url in urls:
		lines = requests.get(url).content.decode("utf-8", "ignore").splitlines()
		# '|' is used as a delimiter in these files
		record = [tuple(line.split('|')) for line in lines[11:]]
		records.append(record)
	# flatten list
	records = [line for sublist in records for line in sublist]
	# create dataframe
	records_df = pd.DataFrame(records, columns = ['CIK', 'Company', 'Filing', 'Date', 'Accession Number'])
	# Eliminate all non-10-K entries from our records dataframe
	records_df = records_df[records_df['Filing']=='10-K']
	return records_df

def get_russell_accession_nums_dataframe(dataframe):
	'''
	This function takes in a dataframe that contains CIK, names, filings, dates and accession numbers of all companies
	which filed a 10-K in the target year.
	Output is a dataframe which contains the same information but only for companies in Russell 2000 Index
	The weakness of this function is the accuracy of the reference file which contains the CIK
	numbers of Russell 2000 companies - this had to be cobbled together as generally this is expensive information to collect.
	'''
	# load file which contains all CIK numbers of Russell 2000 companies
	with open('russell_ciks.csv', 'r') as f:
		reader = csv.reader(f)
		russell_ciks = list(reader)
	# flatten and remove empty lists
	russell_ciks = [num for num in russell_ciks if len(num) != 0]
	# convert russell_cik document to strings
	russell_ciks_str = [str(i[0]) for i in russell_ciks]
	# Select only Russell 2000 entries
	russell_ref_df = dataframe[dataframe['CIK'].isin(russell_ciks_str)]
	# need to remove escape characters from companies' names
	russell_ref_df['Company'] = russell_ref_df['Company'].map(lambda x: x.replace('\\', '').replace('/', ''))
	russell_ref_df = russell_ref_df.reset_index()
	return russell_ref_df

def get_raw_files(dataframe, start_num, end_num):
	'''
	Dataframe is the russell_ref_df, which contains CIKs, company names and accession numbers for Russell 2000 companies.
	The output is a dictionary whose keys are company names and values are raw html text  
	'''
	rawfiles = {}
	for i in range(start_num, end_num):
		cik = dataframe.loc[i]['CIK']
		name = dataframe.loc[i]['Company']
		# Fifth column is where accession number is located - this is used to access target file
		url = 'https://www.sec.gov/Archives/' + dataframe.iloc[i,5]
		lines = requests.get(url).content.decode("utf-8", "ignore").splitlines()
		rawfiles[name] = lines
	return rawfiles

def clean_raw_file(text):       
	'''
	Function that uses regex to clean raw html text
	Does not perform traditional NLP preprocessing
	'''
	text.lower()
	clean = re.compile('<.*?>')
	text = re.sub(clean, ' ', text)
	clean = re.compile('\t')
	text = re.sub(clean, ' ', text)
	clean = re.compile('&nbsp;')
	text = re.sub(clean, ' ', text)
	clean = re.compile('&#[0-9]{2,4};')
	text = re.sub(clean, ' ', text)
	clean = re.compile('&ldquo;')
	text = re.sub(clean, ' ', text)
	clean = re.compile('&rdquo;')
	text = re.sub(clean, ' ', text)
	clean = re.compile('&ndash;')
	text = re.sub(clean, ' ', text)
	clean = re.compile('&lt;')
	text = re.sub(clean, ' ', text)
	clean = re.compile('&amp;')
	text = re.sub(clean, ' ', text)
	clean = re.compile('&apos;')
	text = re.sub(clean, ' ', text)
	clean = re.compile('&apo;')
	text = re.sub(clean, ' ', text)
	clean = re.compile('&quot;')
	text = re.sub(clean, ' ', text)
	clean = re.compile('\v')
	text = re.sub(clean, ' ', text)
	clean = re.compile('&rsquo;')
	text = re.sub(clean, ' ', text)
	clean = re.compile('&lsquo;')
	text = re.sub(clean, ' ', text)
	clean = re.compile('&sbquo;')
	text = re.sub(clean, ' ', text)
	clean = re.compile('&bdquo;')
	text = re.sub(clean, ' ', text)
	clean = re.compile('&#\w{2,6};')
	text = re.sub(clean, ' ', text)
	text = ' '.join(text.split())
	return text.lower()

def get_clean_files(dictionary):
	'''
	Argument is a dictionary whose keys are company names and values are raw html text
	Applies to clean_raw_file function to the argument dictionary's values (html text)
	Returns a ditionary whose keys are company names and values are cleaned text
	'''
	cleanfiles = {}
	for name, file in dictionary.items():
		# Turn the list of strings into one string
		stringfile = ' '.join(file)
		cleanfile = clean_raw_file(stringfile)
		cleanfiles[name] = cleanfile
	return cleanfiles

def target_sections(dictionary):
	'''
	The function takes, as its argument, as dictionary whose keys are company names and values are clean 10-K text
	The output is dictionary whose keys are company names and values are clean 10-K text of only the relevant 10-K items
	'''
	clean_targets = {k:dictionary[k][5000:50000] for k in dictionary.keys()}
	return clean_targets

def save_to_file(dictionary, year):
	'''
	The function takes, as its argument, a dictionary whose keys are company names and values are targeted
	sections of a 10-K
	The function saves the text to a text file in the destination folder
	'''
	current_directory = os.getcwd()
	new = os.path.join(current_directory, f'{year}')
	if not os.path.exists(new):
		os.makedirs(new)
	for name, file in dictionary.items():
		with open(f'{new}\\{name}.text', 'w') as filename:
			filename.write(file)
	return

def get(startnum, stopnum, year = 2018):
	'''
	Pipeline function that sequentially calls all helper functions
	starnum/stopnum are used to chunk the data acquisition
	Default year is set to 2018
	'''
	urls = get_master_idx(year)
	df = create_accession_nums_dataframe(urls)
	print('Master dataframe created')
	russell_df = get_russell_accession_nums_dataframe(df)
	print('Russell CIK nums dataframe created')
	rawfiles = get_raw_files(russell_df, startnum, stopnum)
	print('Raw 10-Ks collected')
	cleanfiles = get_clean_files(rawfiles)
	print ('Raw files cleaned')
	targets = target_sections(cleanfiles)
	print ('Acquired target sections of 10-Ks')
	save_to_file(targets, year)
	print ('Saved to disk')
	return

if (__name__ == '__main__'):
	import sys
	get((int(sys.argv[1])),(int(sys.argv[2])),(int(sys.argv[3])))
	


