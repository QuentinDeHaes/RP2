
# a secondary main file to acquire itemsets that can be turned into  actual items for human interpretation
# (uses the dataset from the kaggle H&M contest)


from eclat.Data_Loader import DataLoader
from eclat.Eclat import new_generate_tidlists, True_Eclat, clean_maximal,True_Eclat_maximal, apriori
from eclat.main import generate_rules
from eclat.rule_rec import get_rec_csv
from knn import get_knn, get_hr_knn
import pandas as pd

from distance_funcs import cosine_distance, jaccard_distance, pointwise_mutual_info, conditional_probability

from similarity_generation import expand_items_redone, expand_items2_redone, get_similarity_items, expand_items_void
import pickle
from eclat.popularity import get_popularity_rec






dl = DataLoader('./data/test.csv')
df = dl.get_df()
dl_train = DataLoader('./transactions_train.csv')
df_train = dl_train.get_df()
df_train["user_session"] = df_train["customer_id"] + df_train["t_dat"]
df_train["product_id"] = df_train["article_id"]
df_train = df_train[["user_session", "product_id"]]
#

df_train = df_train.head(int(len(df_train.index)/8))


df_train.dropna(inplace=True)

import random




# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances

from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import json



def write_to_file(dict, filename):
    """
    store a dictionary to a file
    :param dict: the dictionary we want to store in a file
    :param filename: the location where we store the dictionary
    :return:
    """
    f = open(filename, 'w')
    f.write(json.dumps(dict))
    f.close()


def get_tids_dict(df, eclat_minsup=500, filename=None):
    """
    Enact Eclat to get the maximal productsets.
    :param df: data to base maximal itemsets on
    :param eclat_minsup: the minimal occurences of a productset before it is allowed to be stored
    :param filename: location to store the maximal itemsets
    :return: the maximal productsets.
    """
    tidlist = new_generate_tidlists(df, ["product_id"], "user_session")
    full_tids = True_Eclat_maximal(tidlist, eclat_minsup)
    full_tids = clean_maximal(full_tids)


    dict_full = dict()
    for key, d in full_tids.items():
        dict_full.update(d)

    # turn everything into strings so it can be stored
    new_d = {str(key): str(len(value)) for key, value in dict_full.items()}

    if filename is not None:
        f = open('./temp/HM/' +filename, 'w')
        f.write(json.dumps(new_d))
        f.close()

    return new_d


def alter_Codes(dct, file):
    """
    stringify a set containing item_ids
    :param dct: alter keys using this dict based on get_idtoname_dict()
    :param file: the dict to be strinified
    :return: return stringified dict
    """
    if type(file)==str:
        f = open('./temp/HM/' + file ,  'r')
        file = json.loads(f.read())
        
    new_dct = dict()
    for key in file:
        tup = []
        key2 = eval(key)
        for eid in key2:
            
            tup.append(dct.get(eid,eid))
        new_dct[str(tuple(tup))] = file[key]
            
        
        
        
    return new_dct
        
def get_idtoname_dict():
    """
    generate a dictionary that goes from product_id -> product_name
    :return: a dictionary product_id -> product_name
    """
    dl = DataLoader('./articles.csv')
    df = dl.get_df()
    # print(df.head())
    
    dct = df.groupby("article_id").first()[["product_type_name", "index_name", "detail_desc"]].to_dict()
    
    new_dct = dict()
    for key in dct["product_type_name"]:
        new_dct[key] = (dct["product_type_name"][key] , dct["index_name"][key], dct["detail_desc"][key] )
                                                                                        
    print(new_dct)
    return new_dct

def generate_rules_file(df, eclat_minsup, min_conf, output):
    """
    generate recommendation rules
    :param df: the data which is used to generate the rules
    :param eclat_minsup: the minimal amount of times the rule must occur in the data
    :param min_conf: a rule x->y can only be considered if for all instances of x in the data at least min_conf (*100)% also contain y
    :param output: location to store the rules
    :return: the recommendation rules
    """
    f = open('./temp/'+output+'.txt',"w")
    res = generate_rules(df, eclat_minsup, min_conf, f, ["product_id"], clean=(False, [0], True))
    f.close()
    res.to_pickle('./temp/HM/'+output+'.pkl')

    return res

def get_hr(df, rules, pop_rec,df_gt ,simm):
    """
    calculate the hitrate based on the rules we generated
    :param df:  the known data for which we generate recommendations
    :param rules: the recommmendation rules
    :param pop_rec: the 10 most popular recommendations
    :param df_gt: the desired recommendation for every user whose purchases are described in :df:
    :param simm: a string used to specify were to store information + how to specify this HR in the console
    :return:
    """
    csv = open('./temp/HM/rules_{}.csv'.format(simm),'w')
    hr10 = get_rec_csv(df, csv,rules, pop_rec, False, False, None)


    count = 0
    # count the HR@10 score to check if the result is goor
    for index, row in df_gt.iterrows():
        if row['product_id'] in hr10[row['user_session']]:
            count+=1
    print(count)
    print('hr@10 eclat {}: {}'.format(simm, count/df_gt.count()[0]))
    return count/df_gt.count()[0]


def main_HM():
    print("start cosine")

    sim_dict = get_similarity_items(df_train, 5, cosine_distance)
    f = open("/project_antwerp/RP2/temp/HM/cosine_simdict.pkl", "wb")
    pickle.dump(sim_dict, f)
    f.close()
# #     # f = open("./temp/HM/cosine_simdict.pkl", "rb")
# #     # sim_dict = pickle.load(f)
# #     # f.close()
#     df2 = expand_items_redone(df_train, sim_dict, 5)
#     # rules_base = generate_rules_file(df_train, 200, 0.05, "result_base2")
#     get_tids_dict(df2, filename="result_replace_cosine.json", eclat_minsup=800)
#     df2 = expand_items2_redone(df_train, sim_dict, 5)
#     get_tids_dict(df2, filename="result_add_cosine.json", eclat_minsup=2400)
    dct = get_idtoname_dict()
    result = alter_Codes(dct,"result_replace_cosine.json")
    write_to_file(result, "./temp/HM/result_replace_cosine_word_desc.json" )
    result = alter_Codes(dct,"result_add_cosine.json")
    write_to_file(result, "./temp/HM/result_add_cosine_word_desc.json" )
#     print("start jaccard")
#     sim_dict = get_similarity_items(df_train, 5, jaccard_distance)
#     f = open("./temp/HM/jaccard_simdict.pkl", "wb")
#     pickle.dump(sim_dict, f)
#     f.close()

#     # f = open("./temp/HM/jaccard_simdict.pkl", "rb")
#     # sim_dict = pickle.load(f)
#     # f.close()

#     df2 = expand_items_redone(df_train, sim_dict, 5)
#     get_tids_dict(df2, filename="result_replace_jaccard.json", eclat_minsup=800)
#     df2 = expand_items2_redone(df_train, sim_dict, 5)
#     get_tids_dict(df2, filename="result_add_jaccard.json", eclat_minsup=2400)

    # result = alter_Codes(dct,"result_replace_jaccard.json")
    # write_to_file(result, "./temp/HM/result_replace_jaccard_word_desc.json" )
    # result = alter_Codes(dct,"result_add_jaccard.json")
    # write_to_file(result, "./temp/HM/result_add_jaccard_word_desc.json" )

    
#     print("start conditional")

#     sim_dict = get_similarity_items(df_train, 5, conditional_probability)
#     f = open("./temp/HM/conditional_simdict.pkl", "wb")
#     pickle.dump(sim_dict, f)
#     f.close()

#     # f = open("./temp/HM/conditional_simdict.pkl", "rb")
#     # sim_dict = pickle.load(f)
#     # f.close()

#     df2 = expand_items_redone(df_train, sim_dict, 5)
#     get_tids_dict(df2, filename="result_replace_conditional.json", eclat_minsup=800)
#     df2 = expand_items2_redone(df_train, sim_dict, 5)
#     get_tids_dict(df2, filename="result_add_conditional.json", eclat_minsup=2400)
#     # conditional_replace = get_tids_dict(df2, filename="result_replace_conditional.json", eclat_minsup=500)
#     # conditional_add = get_tids_dict(df3, filename="result_add_conditional.json", eclat_minsup=1000)


    # result = alter_Codes(dct,"result_replace_conditional.json")
    # write_to_file(result, "./temp/HM/result_replace_conditional_word_desc.json" )
    # result = alter_Codes(dct,"result_add_conditional.json")
    # write_to_file(result, "./temp/HM/result_add_conditional_word_desc.json" )


if __name__ == "__main__":
    main_HM()