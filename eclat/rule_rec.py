import pandas as pd
import numpy as np


def generate_hr_1session(items_in_session, rules, top_10_pop):
    """
    generate the hr@10 for a single session
    :param items_in_session: the product_ids in the single session
    :param rules: the generated rules of product_id (and event) based on session_id
    :param top_10_pop: the top 10 most popular list to extend the hr to ensure 10 (or empty if user based recommendation is done afterwards)
    :return: the hr@10 for that single session
    """
    new_rules = rules.loc[lambda rules: rules['X'].apply(is_subset, args=(items_in_session,))]
    new_rules = new_rules[["Y", 'confidence', 'support']]
    new_rules = new_rules.explode('Y')

    items_only = items_in_session
    new_rules = new_rules.loc[lambda new_rules: new_rules['Y'].apply(not_in, args=(items_only,))]

    hitrates = list(new_rules['Y'].unique())
    hitrates = hitrates + top_10_pop
    hitrates = unique(hitrates)
    return hitrates[:10]


def generate_hr_add_user(user_row, hitrates, session_user_dict, user_item_dict, user_rules, top_10_pop):
    """
    generate "extra" hr@10 rules based on the user_id to extend the other rules
    :param user_row: the session_id
    :param hitrates: the list of hitrates already achieved by the generate_hr_1session function
    :param session_user_dict: a dict to map session_ids to user_ids
    :param user_item_dict:a dict to map user_ids to the user_items
    :param user_rules:the generated rules of product_id (and event) based on user_id
    :param top_10_pop:the top 10 most popular list to extend the hr to ensure 10
    :return: 10 hitrates for the session_id
    """

    user_item = user_item_dict[session_user_dict[user_row]]
    if len(hitrates) < 10 and user_item is not None and user_rules is not None:
        new_rules = user_rules.loc[lambda user_rules: user_rules['X'].apply(is_subset, args=(user_item,))]
        new_rules = new_rules[["Y", 'confidence', 'support']]
        new_rules = new_rules.explode('Y')

        items_only = user_item
        new_rules = new_rules.loc[lambda new_rules: new_rules['Y'].apply(not_in, args=(items_only,))]
        new_rules = new_rules.loc[lambda new_rules: new_rules['Y'].apply(not_in, args=(hitrates,))]
        hr2 = list(new_rules['Y'].unique())
        hitrates += hr2
        hitrates = hitrates + top_10_pop
        hitrates = unique(hitrates)
        hitrates = hitrates[:10]

    if len(hitrates) <10:
        print(hitrates)
        print("wack")
    return hitrates


def get_rec_csv(df, file, rules, pop, include_event=True, Include_id=False, id_rules=None):
    """
    return a dictionary for the hr@10 of each session_id
    :param df: the logfile for which we have to get the hr@10
    :param file: the location where we would write the solution
    :param rules:the rules bases on sessionid for generating the hr@10
    :param pop: the list of the most popular items
    :param include_event: a bool that says whether or not (event_type ,product_id) or just (product_id) is used
    :param Include_id:a bool that says whether a secondary training is done using user_id after session_id
    :param id_rules: the rules based on user_id for generating the hr@10
    :return: a dictionary session_id -> list(product_id)[:10)
    """

    # print('=================initialising recomender system=================================')
    # print(pop)
    if include_event:
        df['event/product'] = list(zip(df.event_type, df.product_id))
    else:
        df['event/product'] = df['product_id']
    df = df[["event/product", "user_id", "user_session"]]

    df_dict = df.groupby("user_session")["event/product"].apply(set)
    df_new = pd.DataFrame()
    df_new['user_session'] = df_dict.index
    df_new['product_set'] = df_new['user_session'].apply(lambda x: df_dict[x])

    rules['X'] = rules['X'].apply(set)

    rules['Y'] = rules['Y'].apply(list)
    # print("================generate_hitrates================================")
    if Include_id:
        top_pop = []
    else:
        top_pop = pop

    df_new['hr10'] = df_new['product_set'].apply(generate_hr_1session, args=(rules, top_pop))

    # print("=============== finished generating session based rates ===========")

    hr_dict = df_new.set_index('user_session')

    if Include_id:
        df__dict = df.groupby("user_id")["event/product"].apply(set)
        # df__new = pd.DataFrame()
        # df__new['user_id'] = df__dict.index
        # df__new['product_set'] = df__new['user_id'].apply(lambda x: df__dict[x])

        # df_ses_to_id = df.groupby("user_session")["user_id"]
        df_ses_to_id = df[['user_session', 'user_id']].set_index('user_session').to_dict()['user_id']
        id_rules['X'] = id_rules['X'].apply(set)

        id_rules['Y'] = id_rules['Y'].apply(list)
        # print(df_ses_to_id)
        df_new['hr10'] = df_new.apply(
            lambda row: generate_hr_add_user(row['user_session'], row['hr10'], df_ses_to_id, df__dict, id_rules, pop),
            axis=1)
        hr_dict = df_new.set_index('user_session')

    # print('=================finished generating hitrates====================================')

    temp = hr_dict.explode('hr10')
    temp['product_id'] = temp['hr10']
    temp['product_id'].to_csv(file, line_terminator='\n')

    # print(hr_dict['hr10']['0024f39d-376b-440a-a987-ddf46eb9da09'])
    return hr_dict['hr10']


def is_subset(a, b):
    """
    use the set.issubset method as a function
    :param a: set that should be the subsection
    :param b: set that should be the supersection
    :return: bool a is subsection of b
    """
    return a.issubset(b)


def not_in(a, b):
    """
    do a not in b as a function
    :param a: the object that should be in b
    :param b: the list where a should be in
    :return: bool (a not in b)
    """
    return a not in b


def unique(l):
    # x = np.array(l)
    # x = np.unique(x).tolist()
    # return x # Order preserving
    ''' Modified version of Dave Kirby solution '''
    seen = set()
    return [x for x in l if x not in seen and not seen.add(x)]
