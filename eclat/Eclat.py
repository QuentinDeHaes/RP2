import pandas as pd
from typing import List


def new_generate_tidlists(df: pd.DataFrame, itemtypes: List, tid="user_session"):
    """
    generate a list of all tidlists for the diferent itemtype values
    :param df: the data we will check
    :param itemtypes: the value(s) we generate tidlists for
    :param tid: the value(s) we will be seeing as the tid of the data
    :return: a dict of value to it's tids
    """
    all_tidlists = dict()  # generate the returning dict
    for itemtype in itemtypes:

        if isinstance(itemtype, str):
            itemtype = [itemtype]
        else:
            itemtype = list(itemtype)
        df2 = df[itemtype + [tid]]
        all_tidlists[tuple(itemtype)] = df2.groupby(itemtype)[tid].apply(set).to_dict()

    return all_tidlists


def generate_tidlists(df: pd.DataFrame, itemtypes: List, tid="user_session"):
    """
    DEPRECATED because extremely slow in comparison
    :param df: the data we will check
    :param itemtypes: the value(s) we generate tidlists for
    :param tid: the value(s) we will be seeing as the tid of the data
    :return: a dict of value to it's tids
    """
    all_tidlists = dict()
    for itemtype in itemtypes:
        all_tidlists[itemtype] = dict()
        if isinstance(itemtype, str):
            for distinct_value in df[itemtype].unique():
                all_tidlists[itemtype][distinct_value] = set(df.loc[df[itemtype] == distinct_value][tid])
        else:
            if len(itemtype) == 2:
                for distinct_value in df[itemtype[0]].unique():
                    df2 = df.loc[df[itemtype[0]] == distinct_value]
                    for dist_2 in df2[itemtype[1]].unique():
                        t1 = set(df2.loc[df2[itemtype[1]] == dist_2][tid])
                        all_tidlists[itemtype][(distinct_value, dist_2)] = t1

            else:
                df2 = df[list(itemtype)]
                df2.drop_duplicates()
                for _, distinct_value in df2.iterrows():
                    currentset = None
                    for column in itemtype:
                        fullset = set(df.loc[df[column] == distinct_value[column]].index)
                        if currentset is None:
                            currentset = fullset
                        else:
                            currentset = currentset.intersection(fullset)
                    all_tidlists[itemtype][tuple(distinct_value.values)] = set(df.iloc[list(currentset)][tid])

        print("{} is finished".format(itemtype))

    return all_tidlists


def flatten_dict(dic: dict, only_first: bool):
    """
    flatten the dict of dicts into a single dict
    :param dic: the full original dict
    :param only_first: bool: make the key Only the key of the second dict of make it in the form key1:key2
    :return: flattened dict
    """
    new_dic = dict()
    for key in dic:
        for key2 in dic[key]:
            if only_first:
                new_dic[(key2,)] = dic[key][key2]
            else:
                new_dic[((key, key2),)] = dic[key][key2]

    return new_dic


def Eclat(tidlists, min_sup: int, only_first: bool = True):
    """
    get all tidlists of intersections of bigger sets with a support of atleast min_sup, uses BFS instead of DFS
    :param tidlists: the dict given by (new_)generate_tidlists
    :param min_sup: the minimal required support needed for testing
    :param only_first: a boolean used to state using only the value (True) or both value and the itemtype of the values
    :return: a dict of all (min_sup) values per size
    """
    tidlists = flatten_dict(tidlists, only_first)
    to_delete = set()
    for key in tidlists:
        if len(tidlists[key]) < min_sup:
            to_delete.add(key)
    for key in to_delete:
        del tidlists[key]

    maxlength = 1

    all_tids = {1: tidlists}

    while len(all_tids[maxlength]) != 0:

        maxlength += 1
        all_tids[maxlength] = dict()
        for x in all_tids[maxlength - 1]:
            for y in all_tids[1]:
                if y[0] < x[0]:
                    newtup = y + x
                    inter = all_tids[1][y].intersection(all_tids[maxlength - 1][x])
                    if len(inter) >= min_sup:
                        all_tids[maxlength][newtup] = inter
        print("intersections of {} items is done".format(maxlength))
    return all_tids


def apriori(df: pd.DataFrame, itemtype: str, min_sup: int, tid="user_session"):
    df2 = pd.DataFrame({tid: df.groupby(itemtype)[tid].apply(set)}).reset_index()
    df2['len'] = df2[tid].apply(len)
    df2 = df2[df2.len > min_sup][[tid, itemtype]]
    df2 = df2.explode(tid)
    result_per_size = dict()
    
    df1 = df2.copy()
    df1[itemtype] = df1[itemtype].apply(lambda x: (x,))
    new_dict = df1.groupby(itemtype)[tid].apply(set).to_dict()
    result_per_size[1] = new_dict
    i = 2
    print("oldkeys:", len(new_dict.keys()))

    tempdict = df2.groupby(tid)[itemtype].apply(set).to_dict()
    df2[itemtype] = df2[itemtype].apply(lambda x: (x,))
    while bool(new_dict):
        df2['extra'] = df2[tid].apply(lambda x: tempdict[x])
        # df2 = df2[df2['extra'] !=-1] #remove unoccuri
        df2 = df2.explode('extra')
        
        # df2 = df2[df2['extra'] not in df2[itemtype]] #remove double product_ids
        # df2 = df2.query(("extra not in {}".format(itemtype)))
        df2['tmp'] = df2[itemtype].apply(lambda x : x[-1])
        
        
        df2 = df2[df2['tmp'] < df2['extra']] #ensure no ab AND ba exist simultaneously
        

        df2['extra'] = df2['extra'].apply(lambda x: (x,))
        df2[itemtype] = df2[itemtype] + df2['extra']
        
        df2 = pd.DataFrame({tid: df2.groupby(itemtype)[tid].apply(set)}).reset_index()
        df2['len'] = df2[tid].apply(len)
        df2 = df2[df2.len > min_sup][[tid, itemtype]]
        df2 = df2.explode(tid)
        new_dict = df2.groupby(itemtype)[tid].apply(set).to_dict()
        
        

        result_per_size[i] = new_dict
        i += 1

    del result_per_size[i-1]
    return result_per_size

def True_Eclat(tidlists, min_sup: int, only_first: bool = True):
    """
    The eclat using DFS so the correct implementation
    :param tidlists: the dict given by (new_)generate_tidlists
    :param min_sup: the minimal required support needed for testing
    :param only_first: a boolean used to state using only the value (True) or both value and the itemtype of the values
    :return: a dict of all (min_sup) values per size
    """
    tidlists = flatten_dict(tidlists, only_first)
    to_delete = set()
    for key in tidlists:
        if len(tidlists[key]) < min_sup:
            to_delete.add(key)
    for key in to_delete:
        del tidlists[key]

    all_supported = []

    all_tids = list(tidlists.keys())

    for i in range(len(all_tids)):
        all_supported += _eclat_rec((all_tids[i], tidlists[all_tids[i]]), all_tids[i + 1:], min_sup, tidlists)

    # r = max(map(len, all_supported))
    r = max([len(t[0]) for t in all_supported])
    all_tids = {1: tidlists}
    for i in range(2, r + 1):
        all_tids[i] = dict()
    for val in all_supported:
        all_tids[len(val[0])][val[0]] = val[1]

    return all_tids


def True_Eclat_maximal(tidlists, min_sup: int, only_first: bool = True):
    """
    The eclat using DFS so the correct implementation
    :param tidlists: the dict given by (new_)generate_tidlists
    :param min_sup: the minimal required support needed for testing
    :param only_first: a boolean used to state using only the value (True) or both value and the itemtype of the values
    :return: a dict of all (min_sup) values per size
    """
    tidlists = flatten_dict(tidlists, only_first)
    to_delete = set()
    for key in tidlists:
        if len(tidlists[key]) < min_sup:
            to_delete.add(key)
    for key in to_delete:
        del tidlists[key]

    all_supported = []

    all_tids = list(tidlists.keys())
    to_delete = set()
    for i in range(len(all_tids)):
        y= _eclat_rec((all_tids[i], tidlists[all_tids[i]]), all_tids[i + 1:], min_sup, tidlists)
        if len(y)!= 0:
            to_delete.add(all_tids[i])
        all_supported += y


    for key in to_delete:
        del tidlists[key]
    # r = max(map(len, all_supported))
    r = max([len(t[0]) for t in all_supported])
    all_tids = {1: tidlists}
    for i in range(2, r + 1):
        all_tids[i] = dict()
    for val in all_supported:
        all_tids[len(val[0])][val[0]] = val[1]

    return all_tids

def _eclat_rec(current_thing: tuple, remaining: list, min_sup: int, length1_tidlists: dict):
    """
    the recursive function used within the eclat function
    :param current_thing: a tuple of a tuple (combination of values) and a set of it's intersections of tids
    :param remaining: all values that should still be attempted to be intersected with the current_thing values
    :param min_sup: the minimal support
    :param length1_tidlists:  a dict of tidlists for singular values (of greater than min_sup)
    :return: a list of all current_thing + remaining combinations that have a support greater than min_sup
    """
    x = []
    for i in range(len(remaining)):
        inter = current_thing[1].intersection(length1_tidlists[remaining[i]])

        if len(inter) >= min_sup :
            new_thing = (current_thing[0] + (remaining[i][0],), inter)

            y = _eclat_rec(new_thing, remaining[i + 1:], min_sup, length1_tidlists)
            if len(y) == 0:
                x.append(new_thing)
            else:
                x += y

    return x


def clean_maximal(full_tids):
    max_key = max(full_tids.keys())
    for key in range(max_key,0,-1):
        # print (key)
        for value in full_tids[key]:
            clean_recursive(full_tids, value, key)

    return full_tids

def clean_recursive(full_tids, value, depth):
    if depth ==1:
        return
    # if depth == 2:
    #     print(value[1:])
    res =full_tids[depth-1].pop(value[1:], "not")
    # print(res)
    # print("test")
    # if res != 'not':
    clean_recursive(full_tids,value[1:], depth-1)





def get_confidence(full_tid: dict, conf: float = 0.85):
    """
    use the full_tid to get a list of all confidences
    :param full_tid: dict gotten from the Eclat function
    :param conf: the minimal confidence needed to accept a rule
    :return: a set of (antecedent, consequent, confidence) tuples
    """
    all_confidences = []
    for i in range(len(full_tid) - 1, 0, -1):
        for j in range(i - 1, 0, -1):
            for key1 in full_tid[i]:
                for key2 in full_tid[j]:
                    if set(key2) == set(key1).intersection(set(key2)):
                        thisconf = len(full_tid[i][key1]) / len(full_tid[j][key2])
                        if thisconf > conf:
                            rule_to = tuple(set(key1).difference(set(key2)))
                            all_confidences.append((key2, rule_to, thisconf, len(full_tid[i][key1])))

    return all_confidences


def clean_confidences(all_confidences: set, item_i: list):
    """
    clean out all confidences where exists view, a -> cart, a etc
    :param all_confidences: the set of all confidences given by the get_confidence function
    :param item_i: the item in the antecedent and consequent tuples that need to be compared with eachother (so don't do eventtype but product_id for example)
    :return: a cleaned set of the confidences
    """
    cleaned_confs = set()
    for conf in all_confidences:
        newset = set()
        newset2 = set()
        for i in conf[0]:
            val = i
            for r in range(len(item_i)):
                val = val[item_i[r]]
            newset.add(val)
        for i in conf[1]:
            val = i
            for r in range(len(item_i)):
                val = val[item_i[r]]
            newset2.add(val)

        if len(newset.intersection(newset2)) == 0:
            cleaned_confs.add(conf)
    return cleaned_confs

def clean_confidences2(all_confidences: set):
    cleaned_confs = set()
    for conf in all_confidences:
        if conf[2] < 0.999:
            cleaned_confs.add(conf)

    return cleaned_confs