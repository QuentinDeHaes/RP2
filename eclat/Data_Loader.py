import pandas as pd


class DataLoader:
    def __init__(self, filename: str):
        """
        load in a dataframe from a csv
        :param filename: the location of the csv
        """
        self.df = pd.read_csv(filename,encoding='latin1')

    def append_to_df(self, filename: str) -> None:
        """
        add another df to the dataframe
        :param filename: the location of the csv that needs to be appended to the dataframe
        :return: None
        """
        self.df = self.df.append(pd.read_csv(filename))

    def get_df(self) -> pd.DataFrame:
        """
        get the dataframe
        :return: Dataframe
        """
        return self.df


def load_full_data():
    """
    load the complete data given in our assignment
    :return: the dataframe with all data
    """
    dl = DataLoader("./DATA/2019-Oct.csv")
    dl.append_to_df("./DATA/2019-Nov.csv")
    dl.append_to_df("./DATA/2019-Dec.csv")
    dl.append_to_df("./DATA/2020-Jan.csv")
    dl.append_to_df("./DATA/2020-Feb.csv")
    return dl.get_df()


def load_small_data():
    """
    load only a single csv (october)
    :return: dataframe of october csv
    """
    dl = DataLoader("./DATA/2019-Oct.csv")
    return dl.get_df()
