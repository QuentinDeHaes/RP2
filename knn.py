from similarity_generation import get_similarity_items
from pandas import DataFrame
import math

def get_knn_single_session(product_set, simdict, poprec):
    """
    generate 10 recommendations for a singular product set using k-nearest neighbours
    :param product_set: the singular product set for which to get 10 recommendations
    :param simdict: a dictionary giving a similarity score for each pair of products
    :param poprec: the top 10 most popular items
    :return:10 recommendations
    """
    similarity_values = dict()
    for product in product_set:
        
        sims, sim_scores = simdict.get(product, ([],[]))
        for i in range(len(sims)):
            if sims[i] not in similarity_values:
                similarity_values[sims[i]] = 0
            similarity_values[sims[i]] += math.log(sim_scores[i])
    for product in product_set:
        similarity_values.pop(product, None)
        # del similarity_values[product]

    l = []
    for key in similarity_values:
        l.append((key, similarity_values[key]))
    l.sort(key=lambda x: x[1], reverse=True)
    for x in poprec:
        if x not in similarity_values and x not in product_set:
            l.append((x, 0))
    hr = []
    for i in range(10):
        hr.append(l[i][0])
    return hr


def get_knn(df_train, distance_measure, df, pop_rec):
    """
    generate 10 recommendations for for each session in df. these recommendations are trained based on df_train
    :param df_train: the training data for our recommender
    :param distance_measure: the distance metric used for knn
    :param df: the data for which to generate recommendations
    :param pop_rec:the top 10 most popular items
    :return: a dict containing 10 recommendations for each session in df
    """
    sim_dict = get_similarity_items(df_train, 200, distance_measure)

    df_dict = df.groupby("user_session")["product_id"].apply(set)

    df = DataFrame({'products': df_dict}).reset_index()

    df["hr"] = df["products"].apply(lambda x: get_knn_single_session(x, sim_dict, pop_rec))

    hr = df.groupby("user_session")["hr"].first().to_dict()
    return hr


def get_hr_knn(hr, df_gt, simm=""):
    """
    calculate the hr@10 based on the recommendation
    :param hr:  a dict containing 10 recommendations for each session in df
    :param df_gt: the items, session combinations we want to recommend
    :param simm: text to print to specify extra info like distance metric
    :return: HR@10
    """
    count = 0
    # count the HR@10 score to check if the result is good
    for index, row in df_gt.iterrows():
        if row['product_id'] in hr[row['user_session']]:
            count += 1
    print(count)
    print('hr@10 knn {}: {}'.format(simm, count / df_gt.count()[0]))
    return count / df_gt.count()[0]
