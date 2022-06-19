from RP2.eclat.Data_Loader import DataLoader, pd, load_full_data, load_small_data
from RP2.eclat.Eclat import generate_tidlists, Eclat, new_generate_tidlists, get_confidence, clean_confidences, True_Eclat, clean_confidences2, apriori
import time
import pandas


def generate_rules(df: pd.DataFrame, min_sup: int, min_conf: float, result_file, types: list, tid="user_session",
                   clean=(True, [1],False), only_key=True):
    """
    the function that generates the rules from the dataframe and writes them into a file
    :param df: the dataframe
    :param min_sup: the minimal support needed for a rule to be relevant
    :param min_conf: the minimal confidence needed for a rule to be relevant
    :param result_file: the file to which we will write our relevant rules
    :param types:a list of all the types we will use to generate tidlists from
    :param tid: the column(s) we will use as tid used to generate the tidlists
    :param clean: a tuple of boolean, list, boolean stating whether the rules a->a should be cleaned the list is used to state which values should be compared to eachother and the final bool is to check wether different pairs are from the same rows
    :param only_key: a bool stating whether the typing should be added in the along with the various values
    :return: None
    """
#     starttime = time.time()
#     print("---start generating tidlists for {}---".format(types))

    tidlist = new_generate_tidlists(df, types, tid)  # generate all tidlists based on the typing

#     print("---finished generating tidlists---")
#     print("---starting eclat---")
#     # full_tids = Eclat(tidlist, min_sup)
    full_tids = True_Eclat(tidlist, min_sup,
                           only_first=only_key)  # get the complete tidlists that have a support > min_sup
    # full_tids = apriori(df, types[0],min_sup)
    # for j in range(1,5):
    #     print(list(full_tids[j].items())[0:5])
    print("--- generating rules")

    all_confidences = get_confidence(full_tids, min_conf)  # get all rulesets with a confidence above min_conf

    confidences = all_confidences

    if clean[0]:
        confidences = clean_confidences(all_confidences,
                                        clean[1])  # clean up the confidence rules doing (cart, a) -> (view, a)
    if clean[2]:
        confidences = clean_confidences2(all_confidences)


    print("we have  {} useful confident rulesets".format(len(confidences)))
    # write the rules to a file
    res = pandas.DataFrame(confidences, columns=['X', 'Y', 'confidence', 'support'])
    result_file.write("\n==============================Rules for {}=================================\n".format(types))
    for conf in confidences:
        result_file.write("{} -> {} ;confidence :{} ; support: {}\n".format(conf[0], conf[1], conf[2], conf[3]))

    # print("===finished in {} ===".format(time.time() - starttime))
    return res


# if __name__== "__main__":
#     df = load_full_data()

#     print("---loaded in all data---")
#     f = open("./results.txt", "w")
#     # get all rules for eventtype, product_id -> eventtype, product_id and for event_type, category_id-> event_type, category_id
#     # and for event_type, brand -> event_type, brand but NOT for combinations of these things
#     # These values will be stored in results.txt


#     generate_rules(df, 750, 0.75, f, [("event_type", "product_id")])
#     generate_rules(df, 1000, 0.75, f, [("event_type", "category_id")])
#     generate_rules(df, 1000, 0.75, f, [("event_type", "brand")])

#     f.close()
#     f = open("./results2.txt", "w")

#     # get all rules for eventtype, product_id and/or event_type, category_id -> eventtype, product_id and/or event_type, category_id
#     # values of the previous rulelists will possibly also be included in this ruleset, if their support is high enough,
#     # but it is seperated because the previous rulesets are more interesting, even with a smaller support.
#     # and putting them seperately allows those even more interesting rules to be together in the same space
#     # These values will be stored in results2.txt
#     generate_rules(df, 1500, 0.75, f, [("event_type", "category_id"), ("event_type", "product_id")], only_key=False,
#                    clean=(True, [1, 1], True))

#     f.close()
#     f = open("./results3.txt", "w")
#     # get all rules for eventtype, product_id and/or event_type, category_id and/or event_type, brand
#     # -> eventtype, product_id and/or event_type, category_id and/or event_type, brand
#     # values of the previous rulelists will possibly also be included in this ruleset, if their support is high enough,
#     # but it is seperated because the previous rulesets are more interesting, even with a smaller support.
#     # and putting them seperately allows those even more interesting rules to be together in the same space
#     # These values will be stored in results3.txt
#     generate_rules(df, 5000, 0.8, f,
#                    [("event_type", "category_id"), ("event_type", "product_id"), ("event_type", "brand")], only_key=False,
#                    clean=(True, [1, 1], True))


#     f.close()
#     df2 = df.loc[df["event_type"] == "purchase"]
#     f = open("./results_purchased.txt", "w")
#     # get all rules for  product_id -> product_id ONLY for logs that have eventtype purchased
#     # These values will be stored in results_purchased.txt

#     f.write("\n---products_id rules of only purchased logs----")
#     generate_rules(df2, 200, 0.6, f, ["product_id"], clean=(False,[1], False))

#     f.close()
