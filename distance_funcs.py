import math

def pointwise_mutual_info(sup12, sup1, sup2, all_sessions):
    """
    :param sup12: support of the intersection of item 1 and item2
    :param sup1: support of item1
    :param sup2: support of item2
    :param all_sessions: the total amount of sessions in our training data
    :return: pointwise_mutual_info of item1 and item2
    """
    all_sessions = len(all_sessions)
    return math.log((sup12 * all_sessions) / (sup1 * sup2))


def cosine_distance(sup12, sup1, sup2, all_sessions=None):
    """
    :param sup12: support of the intersection of item 1 and item2
    :param sup1: support of item1
    :param sup2: support of item2
    :param all_sessions: unused variable nescessary for backward compatiblity with pointwise_mutual_info
    :return: cosine distance of item1 and item2
    """
    return sup12 / (math.sqrt(sup1 * sup2))


def jaccard_distance(sup12, sup1, sup2, all_sessions=None):
    """
    :param sup12: support of the intersection of item 1 and item2
    :param sup1: support of item1
    :param sup2: support of item2
    :param all_sessions: unused variable nescessary for backward compatiblity with pointwise_mutual_info
    :return: jaccard distanceof item1 and item2
    """
    return sup12 / (sup1 + sup2 - sup12)


def conditional_probability(sup12, sup1, sup2=None, all_sessions=None):
    """
    :param sup12: support of the intersection of item 1 and item2
    :param sup1: support of item1
    :param sup2: support of item2 (unnecessary but used for backward compatiblity with other distance measures
    :param all_sessions: unused variable nescessary for backward compatiblity with pointwise_mutual_info
    :return: conditional probability of item1 and item2
    """
    return sup12 / sup1
