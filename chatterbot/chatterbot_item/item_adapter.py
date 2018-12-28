# -*- coding: utf-8 -*-
import re
from chatterbot.logic import LogicAdapter
from recsys import create_interaction_matrix, create_item_dict, create_item_emdedding_distance_matrix, item_item_recommendation, runMF
import pandas as pd  # For DataFrame operation
import numpy as np  # Numerical python for matrix operations
from chatterbot.conversation import Statement
import time
import random
import nltk
# nltk.download() # for downloading packages
import string # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")


class MyLogicAdapter(LogicAdapter):
    def __init__(self, **kwargs):
        super(MyLogicAdapter, self).__init__(**kwargs)

    # def LemTokens(tokens):
    #     lemmer = nltk.stem.WordNetLemmatizer()
    #     return [lemmer.lemmatize(token) for token in tokens]

    # def LemNormalize(text):
    #     remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    #     return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

    # Generating response
    @staticmethod
    def responseLanguage(user_response):
        f=open('./database/chatbot.txt','r',errors = 'ignore')
        raw=f.read()
        raw=raw.lower()# converts to lowercase
        #nltk.download('punkt') # first-time use only
        #nltk.download('wordnet') # first-time use only
        sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
        word_tokens = nltk.word_tokenize(raw)# converts to list of words
        sent_tokens[:2]
        word_tokens[:5]
        robo_response=''
        sent_tokens.append(user_response)
        TfidfVec = TfidfVectorizer()
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx=vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if(req_tfidf==0):
            robo_response=robo_response+"I am sorry! I don't understand you"
            return robo_response
        else:
            robo_response = robo_response+sent_tokens[idx]
            return robo_response
        
    # Checking for greetings
    def greeting(self, sentence):
        GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey", "how")
        GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]
        """If user's input is a greeting, return a greeting response"""
        for word in sentence.split():
            if word.lower() in GREETING_INPUTS:
                return random.choice(GREETING_RESPONSES)


    def getItemCodeByName(self, name):
        items = pd.read_csv('./database/item.csv')
        result = items.loc[items['Description'].str.contains(name.upper()) == True]
        try:
            return result.values.tolist()[0][0]
        except IndexError:
            return False

    def checkExistItem(self, item_id):
        items = pd.read_csv('./database/item.csv')
        result = items.loc[items['StockCode'] == item_id]
        try:
            if len(result.values.tolist()[0][1]):
                return True
        except IndexError:
            return False

    def process(self, statement):
        user_input = statement.text
        if(self.greeting(user_input)!=None):
            response = self.greeting(user_input)
        else:
            check_exist = False
            if not re.match("^[0-9 -]+$", user_input):
                item_id = self.getItemCodeByName(user_input)
                check_exist = self.checkExistItem(item_id)
            else:
                item_id = int(user_input)
                check_exist = self.checkExistItem(item_id)
            if check_exist != False:
                start_time_product = time.time()
                products = pd.read_csv('./database/product.csv')
                print("--- Product %s seconds ---" % (time.time() - start_time_product))
                start_time_item = time.time()
                items = pd.read_csv('./database/item.csv')
                print("--- Item %s seconds ---" % (time.time() - start_time_item))

                # CREATE INTERACTION MATRIX
                time_interactions = time.time()
                interactions = create_interaction_matrix(df=products,
                                                        user_col='CustomerID',
                                                        item_col='StockCode',
                                                        quantity_col='Quantity',
                                                        threshold='3')
                print("--- interactions %s seconds ---" % (time.time() - time_interactions))

                time_item_dict = time.time()
                items_dict = create_item_dict(df=items,
                                            id_col='StockCode',
                                            name_col='Description',
                                            price='UnitPrice')
                print("--- items_dict %s seconds ---" % (time.time() - time_item_dict))

                time_mf_model = time.time()
                # Building Matrix Factorization model
                mf_model = runMF(interactions=interactions,
                                n_components=30,
                                loss='warp',
                                k=15,
                                epoch=30,
                                n_jobs=4)
                print("--- mf_model %s seconds ---" % (time.time() - time_mf_model))
                user_input = statement.text
                time_item_item_dist  = time.time()
                # Item - Item Recommender
                item_item_dist = create_item_emdedding_distance_matrix(model=mf_model,
                                                                    interactions=interactions)

                print("--- item_item_dist %s seconds ---" % (time.time() - time_item_item_dist))
                
                time_rec_list = time.time()
                response = item_item_recommendation(item_emdedding_distance_matrix=item_item_dist,
                                                    item_id=item_id,
                                                    item_dict=items_dict,
                                                    n_items=10)
                print("--- rec_list %s seconds ---" % (time.time() - time_rec_list))
            else:
                response = MyLogicAdapter.responseLanguage(user_input)
        # Response statement
        confidence = 1
        response_statement = Statement(response)
        response_statement.confidence = confidence
        return response_statement