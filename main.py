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
#
dl = DataLoader('./data/test.csv')
df = dl.get_df()
dl_train = DataLoader('./data/train.csv')
df_train = dl_train.get_df()
df_train = df_train[["user_session", "product_id", "event_type"]]
#

df_train = df_train.head(int(len(df_train.index)/4))




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
        f = open(filename, 'w')
        f.write(json.dumps(new_d))
        f.close()

    return new_d


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
    res.to_pickle('./temp/'+output+'.pkl')

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
    
    csv = open('./temp/rules_{}.csv'.format(simm),'w')
    hr10 = get_rec_csv(df, csv,rules, pop_rec, False, False, None)


    count = 0
    # count the HR@10 score to check if the result is goor
    for index, row in df_gt.iterrows():
        if row['product_id'] in hr10[row['user_session']]:
            count+=1
    print(count)
    print('hr@10 eclat {}: {}'.format(simm, count/df_gt.count()[0]))
    return count/df_gt.count()[0]


def run_main():
    """
    the main function to run
    acquiring data for all different distance measures tested
    :return:
    """
    pop_rec = get_popularity_rec(df_train)
    pop_rec = pop_rec['event_type'].index.tolist()
    dl_gt = DataLoader('.//data/test_gt.csv')
    df_gt = dl_gt.get_df()
    print("start cosine")

    sim_dict = get_similarity_items(df_train, 5, cosine_distance)
    f = open("./temp/cosine_simdict.pkl", "wb")
    pickle.dump(sim_dict, f)
    f.close()
    # f = open("./temp/cosine_simdict.pkl", "rb")
    # sim_dict = pickle.load(f)
    # f.close()
# # # # #     #
    rules_base = generate_rules_file(df_train, 25, 0.05, "result_base2")
    df2 = expand_items_redone(df_train, sim_dict, 15)
    hr = get_knn(df2, conditional_probability, df, pop_rec)
    get_hr_knn(hr, df_gt, "knn conditional_replace")
    generate_rules_file(df2, 25, 0.05, "result_replace_cosine2")
    df2 = expand_items2_redone(df_train, sim_dict, 10)
    generate_rules_file(df2, 25, 0.05, "result_add_cosine2")
    hr = get_knn(df_train, conditional_probability, df, pop_rec)
    get_hr_knn(hr, df_gt, "knn conditional_add")


    # print("start jaccard")
    # sim_dict = get_similarity_items(df_train, 50, jaccard_distance)
    # f = open("./temp/jaccard_simdict.pkl", "wb")
    # pickle.dump(sim_dict, f)
    # f.close()

    # f = open("./temp/jaccard_simdict.pkl", "rb")
    # sim_dict = pickle.load(f)
    # f.close()

    # df2 = expand_items_redone(df_train, sim_dict, 2)
    # generate_rules_file(df2, 30, 0.001, "result_replace_jaccard2")
    # df2 = expand_items2_redone(df_train, sim_dict, 2)
    # generate_rules_file(df2, 300, 0.001, "result_add_jaccard2")
# # #     # jaccard_replace = get_tids_dict(df2, filename="result_replace_jaccard.json", eclat_minsup=500)
# # #     # jaccard_add = get_tids_dict(df3, filename="result_add_jaccard.json", eclat_minsup=1000)
# # #     # # #
#     print("start conditional")

#     sim_dict = get_similarity_items(df_train, 10, conditional_probability)
#     f = open("./temp/conditional_simdict.pkl", "wb")
#     pickle.dump(sim_dict, f)
    # f.close()

    # f = open("./temp/conditional_simdict.pkl", "rb")
    # sim_dict = pickle.load(f)
    # f.close()

#     df2 = expand_items_redone(df_train, sim_dict, 10)  
#     generate_rules_file(df2,120, 0.001, "result_replace_conditional2")
#     df2 = expand_items2_redone(df_train, sim_dict, 10)
#     generate_rules_file(df2, 600, 0.001, "result_add_conditional2")
    # conditional_replace = get_tids_dict(df2, filename="result_replace_conditional.json", eclat_minsup=500)
    # conditional_add = get_tids_dict(df3, filename="result_add_conditional.json", eclat_minsup=1000)



#     print("start pointwise")
#     sim_dict = get_similarity_items(df_train, 5, pointwise_mutual_info)
#     f = open("./temp/pointwise_simdict.pkl", "wb")
#     pickle.dump(sim_dict, f)
#     f.close()

    # f = open("./temp/pointwise_simdict.pkl", "rb")
    # sim_dict = pickle.load(f)
    # f.close()

#     df2 = expand_items_redone(df_train, sim_dict, 20)
#     generate_rules_file(df2, 400, 0.05, "result_replace_pointwise2")
#     # #
#     # base, others, in_all = help_funcs.remove_duplicates(base_dict,
#     #                                                     other_dicts=[conditional_replace, conditional_add, cosine_replace,
#     #                                                                  cosine_add, jaccard_replace, jaccard_add])
#                                                          # pointwise_replace, pointwise_add
#     # write_to_file(base_dict, "result_base.json")
#     # write_to_file(cosine_replace, "result_replace_cosine.json")
#     # write_to_file(cosine_add, "result_add_cosine.json")
#     # write_to_file(jaccard_replace, "result_replace_jaccard.json")
#     # write_to_file(jaccard_add, "result_add_jaccard.json")
#     # write_to_file(conditional_replace, "result_replace_conditional.json")
#     # write_to_file(conditional_add, "result_add_conditional.json")
#     # write_to_file(in_all, "results_base_everywhere.json")
#     # print(in_all)


    pop_rec = get_popularity_rec(df_train)
    pop_rec = pop_rec['event_type'].index.tolist()

#     dl = DataLoader('./data/test.csv')
#     df = dl.get_df()
    dl_gt = DataLoader('./data/test_gt.csv')
    df_gt = dl_gt.get_df()
#     total_count = 0
#     for row in pop_rec.index:
#         total_count+= df_gt.loc[df_gt['product_id'] == row].count()[0]


#     csv = open('./temp/test.csv', 'w')
    

#     print('hr@10 popularity: {}'.format(total_count/df_gt.count()[0]))
#     # et a dictionary for top 10 recommendations per session(_id)


  

    # rules = pd.read_pickle('./temp/result_replace_cosine2.pkl')
    # get_hr(df, rules, pop_rec,df_gt, "cosine replace")
    # rules = pd.read_pickle('./temp/result_add_cosine2.pkl')
    # get_hr(df, rules, pop_rec, df_gt,"cosine add")


    # rules = pd.read_pickle('./temp/result_replace_jaccard2.pkl')
    # get_hr(df, rules, pop_rec,df_gt, "jaccard replace")
    # rules = pd.read_pickle('./temp/result_add_jaccard2.pkl')
    # get_hr(df, rules, pop_rec,df_gt, "jaccard add")
    
    
    # rules = pd.read_pickle('./temp/result_replace_conditional2.pkl')
    # get_hr(df, rules, pop_rec,df_gt, "conditional replace")
    # rules = pd.read_pickle('./temp/result_add_conditional2.pkl')
    # get_hr(df, rules, pop_rec,df_gt, "conditional add")
#     #
#     #
#     #
    # rules = pd.read_pickle('./temp/result_replace_pointwise2.pkl')
    # get_hr(df, rules, pop_rec, df_gt, "pointwise replace")



#     df2 = expand_items2_redone(df_train, sim_dict, 20)
#     generate_rules_file(df2, 1000, 0.1, "result_add_pointwise2")
# #     # #
# #     # # pointwise_replace = get_tids_dict(df2, filename="result_replace_pointwise.json", eclat_minsup=2000)
# #     # # pointwise_add = get_tids_dict(df3, filename="result_add_pointwise.json", eclat_minsup=3000)
# #     # #
# #     # #
#     rules = pd.read_pickle('./temp/result_replace_pointwise2.pkl')
#     get_hr(df, rules, pop_rec, df_gt, "pointwise replace")
    
#     df2 = expand_items_redone(df_train, sim_dict, 20)
#     generate_rules_file(df2, 800, 0.05, "result_replace_pointwise2")
#     rules = pd.read_pickle('./temp/result_replace_pointwise2.pkl')
#     get_hr(df, rules, pop_rec,df_gt, "pointwise replace")
#     rules = pd.read_pickle('./temp/result_add_pointwise2.pkl')
    # get_hr(df, rules, pop_rec, "pointwise add")
    # hr = get_knn(df_train, cosine_distance, df, pop_rec)
    # get_hr_knn(hr, df_gt, "knn cosine")
    # hr = get_knn(df_train, conditional_probability, df, pop_rec)
    # get_hr_knn(hr, df_gt, "knn conditional")
    # hr = get_knn(df_train, jaccard_distance, df, pop_rec)
    # get_hr_knn(hr, df_gt, "knn jaccard")
    # hr = get_knn(df_train, pointwise_mutual_info, df, pop_rec)
    # get_hr_knn(hr, df_gt, "knn pointwise")

#     csv.close()


if __name__ == "__main__":
    run_main()
