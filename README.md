# MachineLearning
This directory contains all my ML projects codes.

COMP135 proj1 Version 2.0 Feb/27/2018
Created by Xiaofei(Elfie) Guo
# pre-request:
	Python (>= 2.7 or >= 3.3),
	NumPy (>= 1.8.2),
	SciPy (>= 0.13.3).
	Sklearn //for preprocessing the txt files.
	matplotlib //for plot
	
	-To install sklearn packakges, type in terminal using pip:
	
	pip install -U scikit-learn
	
	For more infomation, please check the website below:
	http://scikit-learn.org/stable/install.html
 
	-To install matplotlib, type in terminal:
	
	python -mpip install -U pip
	python -mpip install -U matplotlib
	
	For more infomation, please check the website below:
	https://matplotlib.org/users/installing.html
# How to complie this program:
	chmod +x COMP135proj1.py
	python3 COMP135proj1.py
# Note:
	a)Please include the test data files in the same directory as this program.
	b)There will be some pictures generated in the same directory after running this program.
	c)It may take several minutes for this program to run.
  
  
  
# For dataset:
	This dataset was created for the Paper 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015
	Please cite the paper if you want to use it :)

	It contains sentences labelled with positive or negative sentiment, extracted from reviews of products, movies, and restaurants

	-Format:
	sentence \t score \n

	-Details:
	Score is either 1 (for positive) or 0 (for negative)	
	The sentences come from three different websites/fields:

	imdb.com
	amazon.com
	yelp.com

	For each website, there exist 500 positive and 500 negative sentences. Those were selected randomly for larger datasets of reviews. 
	We attempted to select sentences that have a clearly positive or negative connotaton, the goal was for no neutral sentences to be selected.

	For the full datasets look:

	imdb: Maas et. al., 2011 'Learning word vectors for sentiment analysis'
	amazon: McAuley et. al., 2013 'Hidden factors and hidden topics: Understanding rating dimensions with review text'
	yelp: Yelp dataset challenge http://www.yelp.com/dataset_challenge
