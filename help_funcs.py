def remove_duplicates(base_dict:dict, other_dicts):
    """
    compare other dicts to base_dict, if a key is present in base_dict AND all dicts in other_dicts it is removed from all dicts and added to duplicates
    :param base_dict:
    :param other_dicts:
    :return:  base_dict -duplicates , other_dicts - duplicates , duplicates
    """
    duplicates = {}

    for item in base_dict.keys():
        is_in = True
        for other in other_dicts:
            if item not in other:
                is_in = False
                break

        if is_in:
            duplicates[item] = base_dict[item]

            #
    for key in duplicates.keys():
        base_dict.pop(key)
        for other in other_dicts:
            other.pop(key)
    return base_dict, other_dicts, duplicates