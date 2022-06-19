def get_popularity_rec(df):
    """
    get the 10 most popular items
    :param df: the training file
    :return: the 10 most popular items
    """

    df = df[['event_type', 'product_id']]
    df['event_type'] =df['event_type'].apply(__supfunc)
    df = df.groupby(['product_id']).sum()
    return df.nlargest(10, 'event_type')



def __supfunc(x):
    """
    the function that takes differing scores based on event_type
    :param x:
    :return:
    """
    if x == 'view':
        return 1
    elif x == 'cart':
        return 4
    elif x == 'purchase':
        return 9

    return x