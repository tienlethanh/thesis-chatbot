from recsys import create_interaction_matrix, create_item_dict, create_item_emdedding_distance_matrix, item_item_recommendation, runMF
import pandas as pd ## For DataFrame operation
import numpy as np ## Numerical python for matrix operations
import time


products = pd.read_csv('./database/product.csv')
items = pd.read_csv('./database/item.csv')
## CREATE INTERACTION MATRIX
time_interactions = time.time()
interactions = create_interaction_matrix(df = products,
                                         user_col = 'CustomerID',
                                         item_col = 'StockCode',
                                         quantity_col = 'Quantity',
                                         threshold = '3')
print("--- interactions %s seconds ---" % (time.time() - time_interactions))                                         
#interactions.shape
#interactions.head()

## Create User Dict
# user_dict = create_user_dict(interactions=interactions)
## Create Item dict
time_item_dict = time.time()
items_dict = create_item_dict(df=items,
                              id_col='StockCode',
                              name_col='Description',
                              price='UnitPrice')
print("--- items_dict %s seconds ---" % (time.time() - time_item_dict))                            
## Building Matrix Factorization model
time_mf_model = time.time()
mf_model = runMF(interactions = interactions,
                 n_components = 30,
                 loss = 'warp',
                 k = 15,
                 epoch = 30,
                 n_jobs = 4)
print("--- mf_model %s seconds ---" % (time.time() - time_mf_model))
## User Recommender
# rec_list = sample_recommendation_user(model = mf_model, 
#                                          interactions = interactions, 
#                                       user_id = 10, 
#                                       user_dict = user_dict,
#                                       item_dict = movies_dict, 
#                                       threshold = 4,
#                                       nrec_items = 10)

## Item-User Recommender
# sample_recommendation_item(model = mf_model,
#                            interactions = interactions,
#                            item_id = 1,
#                            user_dict = user_dict,
#                            item_dict = movies_dict,
#                            number_of_user = 15)

## Item - Item Recommender
time_item_item_dist  = time.time()
item_item_dist = create_item_emdedding_distance_matrix(model = mf_model,
                                                       interactions = interactions)
print("--- item_item_dist %s seconds ---" % (time.time() - time_item_item_dist))
time_rec_list = time.time()
rec_list = item_item_recommendation(item_emdedding_distance_matrix = item_item_dist,
                                    item_id = 71270,
                                    item_dict = items_dict,
                                    n_items = 10)
print("--- rec_list %s seconds ---" % (time.time() - time_rec_list))                                