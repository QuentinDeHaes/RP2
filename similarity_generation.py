import random
import numpy as np
from pandas import DataFrame
def get_similarity_items(df, k, distance_measure):
    """
    generate the similarity matrix for all items, and turn it into a dict containing the k most similar items for each item
    :param k: the number of similar items to keep
    :param distance_measure: the method used to calculate the similarity of a pair of items
    :return: a dict containing the k most similar items and their similarity score for each item
    """

    all_items = np.array(df['user_session'].unique())
    # print(all_items)
    # all_items2 = all_items.reshape(-1, 1)
    # print(all_items2)
    df2 = df.copy()
    df_session_items = df.groupby("user_session")['product_id'].apply(list).to_dict()
    df_items_count = df.groupby("product_id")['user_session'].nunique().to_dict()

    df2 = df2[["user_session", "product_id"]]
    df2["product_id2"] = df2["user_session"].apply(lambda x: df_session_items[x])
    df2 = df2.explode("product_id2")
    df2['product_pair'] = list(zip(df2.product_id, df2.product_id2))
    temp = df2.groupby(["product_id", "product_id2"])['user_session'].nunique()
    df2 = DataFrame({'count': temp}).reset_index()

    df2['similarity'] = df2.apply(
        lambda x: distance_measure(x['count'], df_items_count[x['product_id']],
                                   df_items_count[x['product_id2']],
                                   all_items),
        axis=1)
    df2 = df2[df2.similarity > 0]
    
    df_cp = df2.groupby("product_id")['similarity'].apply(max).to_dict()
    df2['similarity'] = df2.apply(lambda x: normalize_similarity(df_cp, x['product_id'], x['similarity']), axis=1)
    df2['sim_pair'] = list(zip(df2.product_id2, df2.similarity))
    temp = df2.groupby("product_id")['sim_pair'].apply(list)
    df2 = DataFrame({'sim_pair': temp}).reset_index()



    def x_sort(l):
        return sorted(l, key=lambda a: a[1], reverse=True)

    df2['sim_pair'] = df2['sim_pair'].apply(x_sort)
    df2['sim_pair'] = df2['sim_pair'].apply(lambda x: x[:k])
    df2['sim_pair'] = df2['sim_pair'].apply(lambda x: zip(*x)).apply(tuple)
    # print(df2)
    new_dict = df2.groupby('product_id')['sim_pair'].apply(tuple).apply(lambda x: x[0]).to_dict()
    #
    print("generated similarities")
    return new_dict


def get_replacement(item, sim_dict):
    """
    a function that is used in the apply method of the  replace_items function, it chooses a random item from the dict to replace the given item with
    :param item: the item to be replaced
    :param sim_dict: the dictionary gained from the get_similarity_items function
    :return: the replacement for item
    """
    return random.choices(sim_dict[item][0], weights=sim_dict[item][1])[0]


def expand_items(df, sim_dict, n):
    """
    generate a dataframe that replaces each item randomly with one of it's similar items
    :param df: the original dataframe
    :param sim_dict: the dictionary gained from the get_similarity_items function
    :param n: the amount of rows generated per row
    :return:
    """
    df2 = df.copy()
    df2['product_id'] = df2['product_id'].apply(get_list, args=(sim_dict, n))
    df2 = df2.explode("product_id")

    return df2

def expand_items_void(df, n ):
    df2 = df.copy()
    df2['user_session'] = df2['user_session'].apply(lambda x: [x + "_{}".format(i) for i in range(n)])
    df2 = df2.explode("user_session")
    return df2

def expand_items_redone(df, sim_dict, n):
    """
    generate a dataframe that replaces each item randomly with one of it's similar items
    :param df: the original dataframe
    :param sim_dict: the dictionary gained from the get_similarity_items function
    :param n: the amount of rows generated per row
    :return:
    """

    df2 = df.copy()
    df2['user_session'] = df2['user_session'].apply(lambda x: [x + "_{}".format(i) for i in range(n)])
    df2 = df2.explode("user_session")
    df2['product_id'] = df2['product_id'].apply(get_replacement, args=(sim_dict,))
    df2.reset_index()
    return df2


def expand_items2(df, sim_dict, n):
    """
        generate a dataframe that adds each one of it's similar items witch prob of similarity
        :param df: the original dataframe
        :param sim_dict: the dictionary gained from the get_similarity_items function
        :param n: the amount of rows generated per row
        :return:
        """

    df2 = df.copy()
    df2['product_id'] = df2['product_id'].apply(lambda x: [x] * n)
    df2 = df2.explode("product_id")
    df2['product_id'] = df2['product_id'].apply(expand_add, args=(sim_dict,))
    df2 = df2.explode("product_id")

    return df2


def expand_items2_redone(df, sim_dict, n):
    """
        generate a dataframe that adds each one of it's similar items witch prob of similarity
        :param df: the original dataframe
        :param sim_dict: the dictionary gained from the get_similarity_items function
        :param n: the amount of rows generated per row
        :return:
        """

    df2 = df.copy()
    df2['user_session'] = df2['user_session'].apply(lambda x: [x + "_{}".format(i) for i in range(n)])
    df2 = df2.explode("user_session")
    df2['product_id'] = df2['product_id'].apply(expand_add, args=(sim_dict,))
    df2 = df2.explode("product_id")
    df2.reset_index()

    return df2


def expand_add(product_id, sim_dict):
    """
    a support function for within an apply of a dataframe to add items based on the similarities for a singular item
    :param product_id: the singular item we want to (possibly) add its similarities to
    :param sim_dict: the dictionary containing similarities for all items
    :return: a list containing product_id and all simmilar items that ended up getting added
    """
    values = sim_dict[product_id]
    maxval = max(values[1])
    probs = [random.random() for _ in range(len(values[0]))]
    l = []
    for i in range(len(values[0])):
        if probs[i] <= values[1][i] / maxval:
            l.append(values[0][i])

    return l


def get_list(product_id, sim_dict, n):
    """
    generate n copies where the product_id is replaced by one of its similar items
    :param product_id:the singular item we want to copy and replace its similarities to
    :param sim_dict: the dictionary containing similarities for all items
    :param n: the amount of copies
    :return: a list of the replacements for product_id
    """
    values = sim_dict[product_id]
    try:
        result = random.choices(values[0], weights=values[1], k=n)
        return result
    except:
        print(values[0], "and", values[1], "for id:", product_id)

def normalize_similarity(dct_max, product_id, sim_score):
    """
    normalize the similarity score to be between 0 and 1 ( used for :expand_add:) (mainly necessary for the :pointwise_mutual_info: metric
    :param dct_max: dictionary containing the highest value achieved as pairwise distance for each product_id
    :param product_id: the product_id of the score we want to normalize
    :param sim_score:the score we want to normalize
    :return: the normalized score
    """

    max_sim_score = dct_max[product_id]
    return sim_score / max_sim_score
    
                                 
                                 

                                 

                                 
                                 
                     
