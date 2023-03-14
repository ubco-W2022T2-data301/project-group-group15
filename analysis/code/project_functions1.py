import pandas as pd
import numpy as np
from sklearn import preprocessing

class EquityData:
    def __init__(self, common_data_path: str="us_equities_tradingview_data_", extension: str=".csv"):
        """Includes a series of data loading and processing functions
        
        :common_data_path: for raw data files that have a common path up to a certain point, specify this to optimize the loading process of multiple files
        """
        self.common_data_path = common_data_path
        self.extension = extension

    def load_and_process(self, file_name: str, directory_path="../data/raw/", number_of_rows: int=500, exclude_columns: list()=[]) -> pd.DataFrame:
        """Uses method chaining to read in the raw data up to a specified number of columns while also dropping any desired columns
        
        :file_name: the name of the file, with the extension included
        :number_of_rows: the total number of rows that the dataframe should have
        :exclude_columns: a list of column names that should be dropped from the data frame
        :returns: a new Pandas DataFrame
        """
        assert type(number_of_rows) == int, "Number of rows must be an integer"
        
        self.df = (
            pd.read_csv(directory_path + self.common_data_path + file_name + self.extension)
            .iloc[:number_of_rows]
            .drop(columns=exclude_columns)
            .dropna()
            )
        
        return self.df
    
    def save_processed_data(self, data: list, file_names: list(), directory_path: str="../data/processed/"):
        for df, file_name in zip(data, file_names):
            df.to_csv(directory_path + "processed_" + self.common_data_path + file_name + self.extension)

def rank(self, df: pd.DataFrame, col: str, normalize_only: bool=False, threshold: float=1.5,
            below_threshold: bool=True, filter_outliers: bool=True, normalize_after: bool=False,
            lower_quantile: float=0.05, upper_quantile: float=0.95):
    """The scoring algorithm for determining the weight of each equity in the construction of the portfolio for this specific column examined.
    Features a custom outlier-filtering algorithm that is robust to outliers in the data set while still returning normalized values.
    
    :df: The original dataframe\n
    :col: The name of the column being extracted from the dataframe provided\n
    :normalize_only: if True, does not apply a threshold to the screening algorithm, and only normalizes values with a minmax scaler\n
    :threshold: the minimum value that equities must have for that column in order to be considered for further analysis\n
    :below_threshold: if True, removes equities that are below the threshold for that column\n
    :filter_outliers: if True, does not consider equities in the data normalization algorithm, but assigns a min or max value to all outliers depending on the below_threshold parameter\n
    :normalize_after: if True, normalizes the data only after the threshold filter has been applied\n
    :lower_quantile: specifies the lower quantile of the distribution when filtering outliers\n
    :upper_quantile: specifies the upper quantile of the distribution when filtering outliers\n
    """
    
    #NOTE: should make an option for no threshold
    self.x = df[col]
    new_col = col + " Score"
    
    # normalization can be done either before or after equities have been filtered by the threshold
    # the difference is that by filtering initially, the min and max values of that smaller set will become 0 and 1 respectively
    df[new_col] = np.NaN # initialize the score column with only NaN values
    
    def outlier_filter(self):
        """
        Nested helper function to filter outliers
        """
        upper_fence = self.x.quantile(upper_quantile)
        lower_fence = self.x.quantile(lower_quantile)
        
        if below_threshold:
            df.loc[self.x > upper_fence, new_col] = 1 # outliers still need to be included in the data (max score assigned)
            df.loc[self.x < lower_fence, new_col] = 0 # lowest score assigned
        else:
            # inverse of the above
            df.loc[self.x > upper_fence, new_col] = 0
            df.loc[self.x < lower_fence, new_col] = 1

        # now only take the rows that are not outliers into the minmax scaler
        self.x = self.x[(self.x <= upper_fence) & (self.x >= lower_fence)]
        
        if normalize_only:
            normalize_after = False
            
        if normalize_after:
            if below_threshold:
                # since we are only taking valid values, we consider the inverse of the values that are below the threshold to be valid values
                self.x = self.x[self.x >= threshold]
            else:
                self.x = self.x[self.x <= threshold]
    
    if filter_outliers:
        outlier_filter(self)
    
    self.y = np.array(self.x).reshape(-1, 1)
    self.y = preprocessing.MinMaxScaler().fit_transform(self.y)

    for col_idx, array_idx in zip(self.x.index, range(len(self.y))):
        df.at[col_idx, new_col] = self.y[array_idx]
    
    # if we are giving the minimum score to values below the threshold, assign 0 to those values
    if not normalize_only:
        if below_threshold:
            df.loc[df[col] <= threshold, new_col] = 0
        else:
            df.loc[df[col] >= threshold, new_col] = 0