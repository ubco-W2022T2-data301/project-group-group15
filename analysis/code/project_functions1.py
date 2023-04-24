import pandas as pd
import plotly as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as mplt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import plotly.express as px
import plotly.figure_factory as ff
from IPython.display import display, Markdown
from tqdm import tqdm
from typing import *
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn import metrics
from plotly.subplots import make_subplots

@dataclass
class ValueRange:
    """A data class that is used to validate value range inputs across class and function definitions."""
    def __init__(self, min: float=0, max: float=1):
        """Constructs the necessary attributes of the ValueRange class.
        
        Args:
            min: the minimum value for the accepted data range.
            max: the maximum value for the accepted data range.
        """
        self.min = min
        self.max = max
    
    def validate(self, x) -> None:
        """Checks if inputs to variables that must lie within a specific range are valid.
        
        Args:
            x: The value that must be checked as satisfying the specified range.
        
        Raises:
            ValueError: If the value does not lie within the specified range.
        
        Returns:
            None.
        """
        if not (self.min <= x <= self.max):
            raise ValueError(f'{x} must be between 0 and 1 (including).')

class EquityData:
    """Includes a series of data loading and processing functions."""
    def __init__(self, common_data_path: str="us_equities_tradingview_data_", extension: str=".csv"):
        """Constructs the necessary attributes of the EquityData class.
        
        Args:
            common_data_path: As each exported processed data set includes an attribution to TradingView, a common name is used for the first part of every file name.
            extension: The file extension attached to each exported file.
        """
        self.common_data_path = common_data_path
        self.extension = extension

    def load_and_process(self, file_name: str, directory_path="../data/raw/", number_of_rows: int=500, exclude_columns: list()=[],
                         additional_data: pd.DataFrame=None, additional_column: str=None, dropna: bool=False) -> pd.DataFrame:
        """Uses method chaining to read in the raw data up to a specified number of columns while also dropping any desired columns.
        
        Args:
            file_name: The name of the file, with the extension included.
            number_of_rows: The total number of rows that the dataframe should have.
            exclude_columns: A list of column names that should be dropped from the data frame.
        
        Returns:
            A new Pandas DataFrame
        """
        assert type(number_of_rows) == int, "Number of rows must be an integer"
        df = pd.DataFrame()
        
        # NOTE: method chains are also used outside of this function--see line 396
        # The nature of our data requires us to do a lot of processing in large functions with specific requirements for different data frames
        df = ( # method chain 1
            pd.read_csv(directory_path + self.common_data_path + file_name + self.extension)
            .iloc[:number_of_rows]
            .drop(columns=exclude_columns)
            .rename_axis('S&P500 Position')
            .sort_index() # making sure the index is always sorted as there are a few edge cases where the index is not sorted
            )
        
        if additional_data is not None and additional_column is not None:
            df = ( # method chain 2
                df
                .assign(new_col=additional_data[additional_column])
                .rename(columns={"new_col": additional_column})
                )
            
        if dropna: # an important conditional since there are some cases where NaN values should be kept
            df = df.dropna() 
        
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        return df
    
    def save_processed_data(self, data: list, file_names: list(), directory_path: str="../data/processed/") -> None:
        """Iterates through the list of Pandas DataFrames provided and creates .csv files for each one in the processed data directory.
        
        Args:
            data: A list of ordered Pandas DataFrames to convert to .csv files.
            file_names: A list of ordered file names for each Pandas DataFrame in the data list, without extensions.
            directory_path: The destination for each .csv file.

        Returns:
            None.
        """
        for df, file_name in zip(data, file_names):
            df.to_csv(directory_path + "processed_" + self.common_data_path + file_name + self.extension)
    
    def combined_data_frame(self, data: list, dropna: bool=True) -> pd.DataFrame:
        """Constructs and returns the central DataFrame used throughout the project; a combination of all data sets.
        
        Args:
            data: a list of Pandas DataFrames to be joined together.
            dropna: whether to drop rows that contain NaN values or not. Applies a threshold.
        
        Returns:
            A Pandas DataFrame.
        """
        df = pd.concat(data, axis=1)
        
        if dropna:
            df = df.dropna(thresh=33)
            
        df = df.loc[:,~df.columns.duplicated()].copy()
        return df

# NOTE: ANALYSIS FUNCTIONS--------------------------------------------------------------------------------------------------------------------
class QuantitativeAnalysis:
    """Includes several analysis functions that process select data across all data sets."""
    def __init__(self, number_of_companies: int=500):
        """Constructs the necessary attributes of the QuantitativeAnalysis class.

        Args:
            number_of_companies: The number of companies included in the sample, with the default being those from the S&P500 Index.
        """
        self.number_of_companies = number_of_companies
        
    def lin_reg_coef_determination(self, df: pd.DataFrame, X: str, y: str='3-Month Performance', filter_outliers: bool=False) -> np.float64:
        """Computes the coefficient of determination for a singular linear regression model.
        
        Args:
            df: A Pandas DataFrame that contains the x and y values to be paired in a linear regression model.
            X: The name of the independent variable to be paired with the dependent variable in the regression model.
            y: The name of the dependent variable to be paired with the independent variable in the regression model.
            filter_outliers: If true, removes outliers from the data set (not recommended).
        
        Returns:
            The coefficient of determination for the linear regression model.
        """
        if filter_outliers:
            df = self.outlier_filtered_df(df, col=y)
        
        X = df[X]
        y = df[y]
        
        y = y.dropna()
        X = X.dropna()
        
        if len(X) > len(y):
            X = X[:len(y)]
        else:
            y = y[:len(X)]
        
        X = np.array(X).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(X, y)
         
        return model.score(X, y)

    def get_lin_reg_coefs(self, df: pd.DataFrame, x_values: list(), y_value: str='3-Month Performance') -> pd.DataFrame:
        """Returns a Pandas DataFrame with the coefficients of determination for each y-on-x regression.
        Example: 3-Month Performance against Price to Earnings Ratio (TTM).
        
        Args:
            df: The DataFrame that contains the columns to process.
            x_values: A list of strings of the names of each column to process.
            y_value: A common y-value to map each x value against in the regression analysis.
            returns: A Pandas DataFrame with the coefficients of determination for each y-on-x regression.
       
        Returns:
            A Panda DataFrame of the coefficients of determination for each predictor against the target dependent variable.
        """
        coef_dict = dict.fromkeys(x_values, 0) # initialize a dict with all the columns assigned to a value of 0
        
        for predictor in tqdm(x_values, desc="Constructing linear regression models", total=len(x_values)):
            coef_dict[predictor] = self.lin_reg_coef_determination(df, X=predictor, y=y_value)
        
        processed_df = pd.DataFrame(list(zip(coef_dict.keys(), coef_dict.values())), columns=[f'Equity Data Against {y_value}', 'Coefficient of Determination'])
        
        return processed_df
        
    def multiple_linear_regression(self, df: pd.DataFrame, predictors: list(), target_y: str='Market Capitalization', model_name: str='Model Results') -> pd.DataFrame:
        """Constructs a multiple linear regression model.
        
        Args:
            df: A Pandas DataFrame containing the data to be processed.
            predictors: The x values that will be used to predict the target y value.
            target_y: The y value to be predicted.
            returns: A Pandas DataFrame containing a statistical summary of the performance of the model.
        
        Returns:
            A Pandas DataFrame containing a statistical summary of the computed model.
        """
        df = df.select_dtypes(exclude='object')
        
        if target_y in predictors:
            predictors.remove(target_y) # so you don't have a perfect correlation for the same variable

        X = df[predictors]
        y = df[target_y]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
        mlr = LinearRegression()
        mlr.fit(X_train, y_train)
        y_pred_mlr = mlr.predict(X_test)

        mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
        mlr_diff.head()

        meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
        meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
        rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
        R2 = mlr.score(X,y)
        n = len(df) # sample size
        p = len(predictors) # number of independent variables
        adjusted_R2 = 1 - (1-R2)*(len(y)-1)/(len(y)-X.shape[1]-1)
                
        results = {'R Squared': R2, 'Adj. R Squared': adjusted_R2,'Mean Absolute Error (MAE)': meanAbErr, 'Mean Square Error (MSE)': meanSqErr, 'Root Mean Square Error (RMSE)': rootMeanSqErr}
        results_df = pd.DataFrame(results, index=[model_name])
    
        return results_df
    
    def rank(self, df: pd.DataFrame, col: str, normalize_only: bool=True, threshold: float=1.5,
             below_threshold: bool=True, filter_outliers: bool=True, normalize_after: bool=False,
             lower_quartile: float=0.05, upper_quartile: float=0.95, inplace: bool=False,
             inverse_normalization_cols: list()=['Price to Revenue Ratio (TTM)', 'Price to Earnings Ratio (TTM)', 'Total Debt (MRQ)', 'Net Debt (MRQ)', 'Debt to Equity Ratio (MRQ)']) -> None:
        
        """The scoring algorithm for determining the weight of each equity in the construction of the portfolio for this specific column examined.
        Features a custom outlier-filtering algorithm that is robust to outliers in the data set while still returning normalized values.
        Normalizes one column at a time.
        
        Args:
            df: The original DataFrame.
            col: The name of the column being extracted from the dataframe provided.
            normalize_only: If true, does not apply a threshold to the screening algorithm, and only normalizes values with a minmax scaler.
            threshold: The minimum value that equities must have for that column in order to be considered for further analysis.
            below_threshold: If true, removes equities that are below the threshold for that column.
            filter_outliers: If true, does not consider equities in the data normalization algorithm, but assigns a min or max value to all outliers depending on the below_threshold parameter.
            normalize_after: If true, normalizes the data only after the threshold filter has been applied.
            lower_quartile: Specifies the lower quantile of the distribution when filtering outliers.
            upper_quartile: Specifies the upper quantile of the distribution when filtering outliers.
            inplace: If true, specifies that the normalization algorithm should directly modify the column being processed, otherwise, a new column is created.
        
        Returns:
            None.
        """
        self.x = df[col]
        new_col = col + " Score"
        
        # normalization can be done either before or after equities have been filtered by the threshold
        # the difference is that by filtering initially, the min and max values of that smaller set will become 0 and 1 respectively
        df[new_col] = np.NaN # initialize the score column with only NaN values
        
        def outlier_filter(self):
            """Nested helper function to filter outliers"""
            upper_fence = self.x.quantile(upper_quartile)
            lower_fence = self.x.quantile(lower_quartile)
            
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
        
        if col in inverse_normalization_cols:
            # NOTE: if it is better if a financial variable has a lower score, then the minimum is assigned a score of 1, and maximum a score of 0
            # therefore, subtracting each element in the array from 1 will return the inverse of the original [0, 1] feature range
            self.y = 1 - self.y
 
        if inplace: # NOTE: this is currently an unstable feature and does not give accurate results
            df.drop(columns=[new_col], inplace=True) # directly modifying the original column, so the new column should be removed
            for col_idx, array_idx in zip(self.x.index, range(len(self.y))):
                df.at[col_idx, col] = self.y[array_idx]
        else:
            for col_idx, array_idx in zip(self.x.index, range(len(self.y))):
                df.at[col_idx, new_col] = self.y[array_idx]
        
        # if we are giving the minimum score to values below the threshold, assign 0 to those values
        if not normalize_only:
            if below_threshold:
                df.loc[df[col] <= threshold, new_col] = 0
            else:
                df.loc[df[col] >= threshold, new_col] = 0
    
    def outlier_filtered_df(self, df: pd.DataFrame, col: str, lower_quartile: float=0.05, upper_quartile: float=0.95) -> pd.DataFrame:
        """Removes outliers from the Pandas DataFrame provided and returns a new DataFrame that excludes outliers according to filtering done by lower and upper quantiles
        
        Args:
            df: A Pandas DataFrame of the data to be filtered.
            col: The name of the column to filter.
            lower_quartile: The quartile used for the lower fence variable.
            upper_quartile: The quartile used for the upper fence variable.
        
        Returns:
            Returns a DataFrame that excludes outliers from the original DataFrame according to the column provided
        """
        upper_fence = df[col].quantile(upper_quartile)
        lower_fence = df[col].quantile(lower_quartile)

        df = df[(df[col] <= upper_fence) & (df[col] >= lower_fence)]
        
        return df
    
    def extract_corr_plot_counts(self, df: pd.DataFrame, correlation_threshold: int=0.7) -> pd.DataFrame:
        """Extracts the number of high correlation counts that each variable has in relation to every other variable in the data set.
        
        Args:
            df: A Pandas DataFrame containing all of the variables to be examined
            correlation_threshold: The minimum Pearson correlation coefficient that each correlation must have in order to be counted as being of a high correlation.
        
        Returns:
            A Pandas DataFrame that contains summary statistics regarding the number of high correlation counts.
        """
        corr = df.corr(numeric_only=True)

        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        corr_df = corr.mask(mask).dropna(how='all').dropna('columns', how='all')

        cols = [col for col in corr_df]
        score_count_df = corr_df
        score_count_df['Count'] = 0
        score_count_df = score_count_df.drop(columns=cols)
        score_count_df = score_count_df.T
        score_count_df['Price'] = 0
        score_count_df.columns = score_count_df.columns

        for col in corr_df.columns:
            for row in corr_df.index:
                current_score = abs(corr_df.loc[row, col])
                if current_score != 1 and current_score >= correlation_threshold:
                    score_count_df[col] += 1
                    score_count_df[row] += 1
        
        return score_count_df

# NOTE: VISUALIZATION FUNCTIONS--------------------------------------------------------------------------------------------------------------------
class DataVisualization(QuantitativeAnalysis):
    """Contains a series of visualization wrapper functions for graphics created with Plotly or Seaborn. Inherits from QuantitativeAnalysis."""
    def __init__(self):
        """Constructs the necessary attributes of the DataVisualization class."""
        QuantitativeAnalysis.__init__(self)
        EquityData.__init__(self)
        
        self.processed_equities = EquityData("processed_us_equities_tradingview_data_")
    
    def cross_regression_model_comparison(self, target_y: str, known_predictors: list(), computed_predictors: list(), control_test_predictors: list(), title: str) -> plt.graph_objs._figure.Figure:
        """A graphical representation of a semi-algorithmic multiple linear regression optimization algorithm that uses the Scikit-Learn linear regression model. Is not used for predictions, but rather, to confirm trends.
        
        Args:
            target_y: The name of the column from the processed data set to be used on the y-axis and paired with the predictors.
            known_predictors: A list of column names from the processed data set to be used as predictors against the target y. These columns should be those which are known by financial specialists to be particularly important. Used as a secondary control test.
            computed_predictors: A list of column names from the processed data set to be used as predictors against the target y. These columns should be those that are intended to be tested against the control test.
            control_test_predictors: A list of column names from the processed data set to be used as predictors against the target y. Used as the primary control test and should feature only low-correlation columns.
            title: The name of the model being developed.
        
        Returns:
            A bar plot.
        """
        mlr_data = ( # method chain 3
                    self.processed_equities.load_and_process('normalized_data_unweighted_aggregated_score', '../data/processed/')
                    .select_dtypes(exclude='object')
                    .dropna()
                    )

        combined_predictors = computed_predictors + known_predictors

        mlr_known_predictors = self.multiple_linear_regression(mlr_data, known_predictors, target_y, 'Known Predictors')
        mlr_computed_predictors = self.multiple_linear_regression(mlr_data, computed_predictors, target_y, 'Computed Predictors')
        mlr_combined_predictors = self.multiple_linear_regression(mlr_data, combined_predictors, target_y, 'Combined Predictors')
        mlr_control_test = self.multiple_linear_regression(mlr_data, control_test_predictors, target_y, 'Low Correlation Columns (Control Test)')

        models = [mlr_known_predictors, mlr_computed_predictors, mlr_combined_predictors, mlr_control_test]
        mlr_complete = pd.concat(models)
        mlr_complete

        barplot = px.bar(
            mlr_complete,
            x=mlr_complete.index,
            y=mlr_complete.columns,
            template='plotly_dark',
            title=title,
            labels={'value':'Statistical Value', 'index':'Regression Model', 'variable':'Metric'})
        
        return barplot

    def score_density_plot(self, df: pd.DataFrame, cols: list(), title: str="Density Plot", normalization: bool=True, search_for_score: bool=True) -> plt.graph_objs._figure.Figure:
        """Constructs an interactive compound density plot based on a histogram of the data provided, plotting a density curve with clusters of data points below.
        
        Args:
            df: A Pandas DataFrame of equity data.
            cols: A list of column names to be plotted.
            title: The title of the density plot.
            normalization: If true, normalizes the data that has been passed.
            search_for_score: If true, assumes that normalized data has already been passed, and searches for columns that end with "Score".
        
        Returns:
            A density plot.
        """
        df = df.select_dtypes(exclude='object')[:self.number_of_companies]
        df = df.dropna() # mandatory for the function to work
        
        if normalization:
            for column in cols:
                self.rank(df, col=column, upper_quartile=0.99, lower_quartile=0.01)

        if search_for_score:
            hist_data = [df[x + " Score"] for x in cols]
            group_labels = [x + " Score" for x in cols]
        else:
            hist_data = [df[x] for x in cols]
            group_labels = [x for x in cols]
        colors = ['#94F3E4', '#333F44', '#37AA9C']

        fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)
        fig.update_layout(title_text=title, template='plotly_dark')

        fig.update_xaxes(title='Score (0 = low, 1 = high)')
        fig.update_yaxes(title='Density')
        
        return fig

    def legacy_score_density_plot(self, df: pd.DataFrame, data_name: str) -> plt.graph_objs._figure.Figure:
        """Constructs an interactive compound density plot based on a histogram of the data provided, plotting a density curve with clusters of data points below.
        Is included as an older version of the newer score_density_plot function for backwards compatibility with the exploratory data analysis (EDA) files in the ungraded section of the project.
        
        Args:
            :df: A Pandas DataFrame of equity data.
            :data_name: The name of the type of data that has been input into the plot.
       
        Returns:
            A density plot.
        """
        df = df.select_dtypes(exclude='object')[:self.number_of_companies]
        df = df.dropna() # mandatory for the function to work
        n = len(df)
        
        for column in df.columns:
            self.rank(df, col=column, upper_quartile=0.99, lower_quartile=0.01)
            
        score_data_length = len(df.axes[1])
        input_df = df.T[int(score_data_length/2 + 1):].T
        hist_data = [input_df[x] for x in input_df.columns]
        
        group_labels = [x for x in input_df.columns]
        colors = ['#333F44', '#37AA9C', '#94F3E4']

        fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)
        fig.update_layout(title_text=f'Distribution for Normalized {data_name} of {n} Companies in the S&P500', template='plotly_dark')
        
        fig.update_xaxes(title='Score (0 = low, 1 = high)')
        fig.update_yaxes(title='Density')
        
        return fig

    def heatmap_plot(self, df: pd.DataFrame, title: str='Heat Map', number_of_companies: int=500, number_of_subset_companies: int=20,
                    plot_last_companies: bool=False, sort_by: str='Market Capitalization', correlation_plot: bool=False,
                    plot_width: int=1000,plot_height: int=1000) -> plt.graph_objs._figure.Figure:
        """A wrapper function for the default heat map plot, constructing an interactive heat map plot of equity data against each company (ticker).
        
        Args:
            df: A Pandas DataFrame of equity data.
            data_name: The name of the type of data that has been input into the plot.
            number_of_companies: The number of companies that the normalization algorithm.
            number_of_subset_companies: The number of companies that will be included in the.
            correlation_plot: If true, creates a correlation plot instead of a heat map plot.

        Returns:
            A heat map plot.
        """
        df = df.dropna() # to prevent gaps in the heat map
        
        def construct_correlation_plot() -> plt.graph_objs._figure.Figure:
            """A helper function to convert the heat map into a correlation plot"""
            # Correlation
            df_corr = df.corr(numeric_only=True).round(1)
            # Convert to a triangular correlation plot
            mask = np.zeros_like(df_corr, dtype=bool)
            mask[np.triu_indices_from(mask)] = True
            # Final visualization
            df_corr_viz = df_corr.mask(mask).dropna(how='all').dropna('columns', how='all')
            
            fig = px.imshow(
                df_corr_viz,
                text_auto=True,
                template='plotly_dark',
                title=title,
                width=plot_width,
                height=plot_height)
            
            return fig
        
        df = df.sort_values(by=sort_by, ascending=False)
    
        if correlation_plot:            
            return construct_correlation_plot()
            
        else:
            df = df[:number_of_companies] # selecting only x number of companies in order
                
            z = []
            tickers = df['Ticker']
            index = df['Ticker']
            df = df.select_dtypes(exclude='object')
            for column in df.columns:
                self.rank(df, col=column) # scoring the data
                        
            if plot_last_companies:
                df = df[-number_of_subset_companies:]
                tickers = tickers[-number_of_subset_companies:]
            else:
                df = df[:number_of_subset_companies] # the normalization algorithm has been applied on number_of_companies but we choose a subset from that
                tickers = tickers[:number_of_subset_companies]
            
            score_data_length = len(df.axes[1])
            input_df = df.T[int(score_data_length/2 + 1):].T
            for column in input_df.columns:
                z.append(input_df[column].round(3))
            
            fig = px.imshow(
                z,
                text_auto=True,
                template='plotly_dark',
                title=title,
                x=[x for x in tickers], 
                y=[x for x in df.columns[int(score_data_length/2 + 1):]],
                width=plot_width,
                height=plot_height)
        
        return fig
    
    def subplot_generator(self, df: pd.DataFrame, predictors: list, title: str, height_reduction_factor = 8,
                          width_multiplier = 1, horizontal_spacing= 0.02, vertical_spacing = 0.005, rows=4, cols=4) -> plt.graph_objs._figure.Figure:
        """Creates a custom faceted heat map plot.
        
        Args:
            df: A Pandas DataFrame containing the normalized data to be used in the construction of heat map subplots.
            predictors: A list of columns to be included in the heat map.
            title: The title of the heat map.
            height_reduction_factor: Depending on the output of the heat map, reduces the spacing based on the number passed.
            width_multiplier: Depending on the output of the heat map, scales the width of the heat map.
            horizontal_spacing: Depending on the output of the heat map, further reduces the spacing based on the number passed.
            vertical_spacing: Depending on the output of the heat map, further reduces the spacing based on the number passed.
            rows: The number of rows for the faceted heat map plot.
            cols: The number of columns for the faceted heat map plot.
        
        Returns:
            A custom faceted heat map plot.
        """
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_yaxes=True,
            subplot_titles = [predictor for set in predictors for predictor in set],
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
            )
        
        def pair_construction(row: int, col: int, predictor) -> None:
            """A helper function to construct the faceted plot."""
            row += 1
            col += 1
            
            fig.add_trace(
            trace=self.heatmap_plot(
            df=df,
            plot_last_companies=False,
            sort_by=predictor).data[0],
            row = row,
            col = col
        )
            fig.add_trace(
            trace=self.heatmap_plot(
            df=df,
            plot_last_companies=True,
            sort_by=predictor).data[0],
            row = row,
            col = col
        )
        
        for row in range(rows):
            for col in range(cols):
                pair_construction(row, col, predictors[row][col])

        height_multiplier = rows*cols - height_reduction_factor
        fig.update_layout(
            title_text=title,
            template='plotly_dark',
            width=1500*width_multiplier,
            height=1500*height_multiplier)
        
        return fig

    def binary_subplot_generator(self, df: pd.DataFrame, predictors: list, title: str, height_reduction_factor = 8,
                                 width_multiplier = 1, horizontal_spacing= 0.02, vertical_spacing = 0.005, rows=4, cols=2) -> plt.graph_objs._figure.Figure:
        """Creates a custom faceted heat map plot that separates the originally merged heat maps seen in the first generator into distinct columns.
        
        Args:
            df: A Pandas DataFrame containing the normalized data to be used in the construction of heat map subplots.
            predictors: A list of columns to be included in the heat map.
            title: The title of the heat map.
            height_reduction_factor: Depending on the output of the heat map, reduces the spacing based on the number passed.
            width_multiplier: Depending on the output of the heat map, scales the width of the heat map.
            horizontal_spacing: Depending on the output of the heat map, further reduces the spacing based on the number passed.
            vertical_spacing: Depending on the output of the heat map, further reduces the spacing based on the number passed.
            rows: The number of rows for the faceted heat map plot.
            cols: The number of columns for the faceted heat map plot.
        
        Returns:
            A custom faceted heat map plot.
        """
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_yaxes=True,
            column_titles=['Top 20 Companies', 'Bottom 20 Companies'],
            row_titles = ['Sorted by ' + predictor for predictor in predictors],
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
            )
        
        for col in range(1, cols+1):
            for row, predictor in zip(range(1, rows+1), predictors):
                if (col % 2 == 0):
                    fig.add_trace(
                    trace=self.heatmap_plot(
                    df=df,
                    plot_last_companies=True,
                    sort_by=predictor).data[0],
                    row = row,
                    col = col
                )
                else:
                    fig.add_trace(
                    trace=self.heatmap_plot(
                    df=df,
                    plot_last_companies=False,
                    sort_by=predictor).data[0],
                    row = row,
                    col = col
                )

        height_multiplier = rows*cols - height_reduction_factor
        fig.update_layout(
            title_text=title,
            template='plotly_dark',
            width=1500*width_multiplier,
            height=1500*height_multiplier)
        
        return fig

    def scatter_3d(self, df: pd.DataFrame, x: str, y: str, z: str) -> plt.graph_objs._figure.Figure:
        """Constructs a 3D interactive plot of equity data on 3 axes. Used in the exploratory data analysis phase.
        
        Args:
            df: A Pandas DataFrame of equity data.
            x: The name of the column data to be plotted on the x-axis, as a string.
            y: The name of the column data to be plotted on the y-axis, as a string.
            z: The name of the column data to be plotted on the z-axis, as a string.
        
        Returns:
            A 3D scatter plot.
        """
        df.index = df['Ticker']
        df = df.select_dtypes(exclude='object')
        
        for column in df.columns:
            self.rank(df, col=column)
            
        fig = px.scatter_3d(df, x=x, y=y, z=z,
                    title='3D Scatter Plot of Normalized Equity Data',
                    template='plotly_dark',
                    size_max=18,
                    color='3-Month Performance Score',
                    opacity=0.7)

        return fig
    
    def correlation_plot(self, df, data_name) -> None: 
        """Produces a correlation plot that maps all of the data points in the Data Frame provided.
        
        Args:
            df: A Pandas DataFrame of the data to be processed.
            data_name: The name of the data being plotted.
        
        Returns:
            None.
        """
        # Compute the correlation matrix
        corr = df.corr(numeric_only=True)

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(1, 10, as_cmap=True)

        #Draw the heat map with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        mplt.title(f"Correlation Plot of {data_name}")

class PortfolioRecommendation(EquityData, QuantitativeAnalysis):
    """A portfolio recommendation class that allocates user funds to a series of assets in conjunction with the results from the analysis algorithms applied. Inherits from EquityData, QuantitativeAnalysis."""
    def __init__(self, portfolio_size: int=50, initial_capital: float=100000.00, capital_per_period: float=100.00, period: int=7, dividends_importance: bool=False, preferred_industries: list=["Technology Services, Electronic Technology"],
                volatility_tolerance: Annotated[float, ValueRange(0.0, 1.0)]=0.7, preferred_companies: list=["Apple, Google, Microsoft, Amazon"], diversification: Annotated[float, ValueRange(0.0, 1.0)]=0.2, investment_strategy: str="Growth"):
        """Constructs the necessary attributes of the PortfolioRecommendation class.
        
        Args:
            portfolio_size: The number of assets included in the final portfolio.
            initial_capital: The initial amount of cash to be invested by the client, in CAD.
            capital_per_period: The amount of cash to be invested by the client at a fixed rate in addition to the initial capital invested, in CAD.
            period: The frequency (in days) at which additional cash is invested, if desired.
            dividends_importance: Specifies whether dividends are important to the client, dictating whether analysis algorithms should place greater importance on dividends.
            preferred_industries: Specifies a list of industries that the analysis algorithms should prioritize when constructing the investment portfolio.
            volatility_tolerance: Accepts a range of values from 0 to 1, with 1 implying maximum volatility tolerance (i.e. the client is willing to lose 100% of their investment to take on more risk).
            preferred_companies: Specifies a list of companies that the analysis algorithms will accommodate in the final portfolio irrespective of their score.
            diversification: Accepts a range of values from 0 to 1, with 1 implying maximum diversification (i.e. funds will be distributed evenly across all industries and equally among all companies).
            investment_strategy: Specifies the investment strategy that will guide the output of the analysis algorithms, in which this analysis notebook strictly focuses on growth investing.
            
        Raises:
            ValueError if an input parameter does not satisfy its accepted range.
        """
        EquityData.__init__(self)
        QuantitativeAnalysis.__init__(self)
        
        self.initial_capital = initial_capital
        self.portfolio_size = portfolio_size
        self.capital_per_period = capital_per_period
        self.period = period
        self.dividends_importance = dividends_importance
        self.preferred_industries = preferred_industries
        self.volatility_tolerance = volatility_tolerance
        self.preferred_companies = preferred_companies
        self.diversification = diversification
        self.preferred_companies = preferred_industries
        self.investment_strategy = investment_strategy
        
        ValueRange().validate(volatility_tolerance) # ensures that the value is within the allowed range
        ValueRange().validate(diversification) # ensures that the value is within the allowed range
    
    def compute_diversification(self) -> tuple:
        """Determines the degree of diversification for the sample portfolio.
        
        Returns:
            A tuple of lower and upper quantiles to be used to classify outliers, with outliers receiving an equal number of cash.
        """
        if 0.9 <= self.diversification <= 1: # highest degree of diversification (evenly split funds)
            return 0.3, 0.7
        elif 0.75 <= self.diversification < 0.9:
            return 0.2, 0.8
        elif 0.5 <= self.diversification < 0.75:
            return 0.1, 0.9
        elif 0.25 <= self.diversification < 0.5:
            return 0.01, 0.99
        else:
            return 0.011, 0.999 # lowest degree of diversification (do not split evenly split funds for many companies)
    
    def asset_allocation(self, save_aggregated_scores: bool=False) -> pd.DataFrame:
        """Divides the initial capital invested across a series of ranked equities, creating a sample portfolio of equities as a Pandas DataFrame.
        
        Args:
            save_aggregated_scores: If true, uploads the normalized data set with the unweighted aggregated scores column added to the DataFrame
        
        Returns:
            A Pandas DataFrame that represents a sample portfolio of equities.
        """
        equities = EquityData("processed_us_equities_tradingview_data_")
        scored_equities = equities.load_and_process("normalized_data", directory_path="../data/processed/")
        complete_df = equities.load_and_process("complete_data", directory_path="../data/processed/")

        score_count_df = self.extract_corr_plot_counts(complete_df).T
        
        def extract_top_predictors(threshold: int=6, exclude_columns: list=['Gross Profit (FY)', 'Enterprise Value (MRQ)']) -> pd.DataFrame:
            """A helper function to extract the top predictors."""
            score_count_df['Assigned Weight'] = score_count_df['Count'] / sum(score_count_df['Count'])
            
            top_predictors_narrowest = score_count_df[score_count_df['Count'] >= threshold].index
            top_predictors_narrowest = top_predictors_narrowest.drop(exclude_columns)
            
            return top_predictors_narrowest
        
        score_count_df['Assigned Weight'] = score_count_df['Count'] / sum(score_count_df['Count'])
        before_weighting = scored_equities
        scored_equities = scored_equities.select_dtypes(exclude='object')

        if not save_aggregated_scores:
            for col in scored_equities.columns:
                standard_col = col[:-6]
                scored_equities[col] = scored_equities[col] * score_count_df.T[standard_col]['Assigned Weight']

        scored_equities['Aggregated'] = scored_equities[scored_equities.columns].sum(axis=1, numeric_only=True)
        
        degree_of_diversification = self.compute_diversification()
        lower_quartile = degree_of_diversification[0]
        upper_quartile = degree_of_diversification[1]

        self.rank(scored_equities, 'Aggregated', lower_quartile=lower_quartile, upper_quartile=upper_quartile) # NOTE: we control the degree of portfolio diversification via the number of outliers
        scored_equities = scored_equities.drop(columns=['Aggregated'])

        scored_equities['Ticker'] = before_weighting['Ticker']
        scored_equities['Sector'] = before_weighting['Sector']
        scored_equities['Description'] = before_weighting['Description']
        
        scored_equities = scored_equities.sort_values(by='Aggregated Score', ascending=False)
        top_predictors_narrowest_adjusted = extract_top_predictors()

        vars = [predictor + ' Score' for predictor in top_predictors_narrowest_adjusted]
        vars.append('Aggregated Score')
        vars.append('Ticker')
        vars.append('Sector')
        vars.append('Description')
        
        if save_aggregated_scores:
            self.save_processed_data([scored_equities], ['normalized_data_unweighted_aggregated_score'])
        
        scored_equities = scored_equities[vars]
        for col in scored_equities.columns:
            if col != 'Aggregated Score' and col != 'Sector':
                scored_equities[col] = before_weighting[col] # restoring the non-weighted normalized values for clarity

        scored_equities = scored_equities[:self.portfolio_size]
        scored_equities['Funds Allocated'] = round((scored_equities['Aggregated Score'] / sum(scored_equities['Aggregated Score'])) * self.initial_capital, 2)
        scored_equities['Percentage Allocated'] = (scored_equities['Funds Allocated'] / self.initial_capital) * 100
        
        scored_equities = scored_equities.rename(columns={"Description": "Name"})
        
        keep_columns = [
            'Ticker',
            'Name',
            'Sector',
            'Funds Allocated',
            'Percentage Allocated',
            'Aggregated Score',
            'Market Capitalization Score',
            'Gross Profit (MRQ) Score',
            'Total Current Assets (MRQ) Score',
            'EBITDA (TTM) Score']
    
        scored_equities = scored_equities.loc[:, keep_columns]
        scored_equities = scored_equities.rename_axis('S&P500 Position')
        
        display(Markdown("# Sample Portfolio: Algorithmic Asset Allocation"))
        return scored_equities

class DataUploadFunctions(EquityData, QuantitativeAnalysis):
    """A series of functions that were used to create the processed data under the processed data folder. Inherits from EquityData, QuantitativeAnalysis."""
    def __init__(self):
        """Constructs the necessary attributes of the DataUploadFunctions class."""
        EquityData.__init__(self)
        QuantitativeAnalysis.__init__(self)
        
        self.processed_data = EquityData('processed_us_equities_tradingview_data_')
        self.complete_df = self.processed_data.load_and_process('complete_data', directory_path='../data/processed/')
    
    def save_normalized_data(self) -> None:
        """Saves the normalized data to the processed data folder.
        
        Returns:
            None.
        """
        str_cols_only = self.complete_df.select_dtypes(include='object')
        complete_df = self.complete_df.select_dtypes(exclude='object')

        previous_cols_no_str = complete_df.columns

        for col in complete_df.columns:
            self.rank(complete_df, col)

        complete_df = complete_df[[col + ' Score' for col in previous_cols_no_str]]
        for col in str_cols_only:
            complete_df[col] = complete_df[col]

        self.save_processed_data([complete_df], ['normalized_data'])
    
    def save_top_predictors(self, return_top_predictors: bool=False):
        """Saves the top predictors identified to the processed data folder.
        
        Args:
            return_top_predictors: If true, returns a tuple containing two Pandas DataFrames of the top identified predictors.
        
        Returns:
            None (if return_top_predictors is False)
            A tuple of Pandas DataFrames (if return_top_predictors is True)
        """
        score_count_df = self.extract_corr_plot_counts(self.complete_df).T
        score_count_df['Assigned Weight'] = score_count_df['Count'] / sum(score_count_df['Count'])

        worst_predictors = score_count_df[score_count_df['Count'] == 0].index
        top_predictors_wide = score_count_df[score_count_df['Count'] >= 4].index # 4 x 4 grid
        top_predictors_narrowest = score_count_df[score_count_df['Count'] >= 6].index
                
        score_count_df = score_count_df.sort_values(by='Assigned Weight', ascending=False)
        self.rank(score_count_df, 'Assigned Weight', filter_outliers=False)
        self.save_processed_data([score_count_df], ['top_predictors'])

        if return_top_predictors:
            return top_predictors_wide, top_predictors_narrowest, worst_predictors # worst_predictors included for control tests
    
    def save_demo_portfolio(self) -> None:
        """Saves the demo portfolio constructed to the processed data folder.
        
        Returns:
            None.
        """
        portfolio = self.PortfolioRecommendation(500, initial_capital=500000)
        demo_portfolio = portfolio.asset_allocation()
        demo_portfolio = demo_portfolio[['Ticker', 'Aggregated Score']]
        self.save_processed_data([demo_portfolio], ['complete_aggregated_scores'])
    
    def save_complete_data(self, dfs: list(), dfs_names: list()) -> None:
        """Saves the entire processed DataFrame (with all the columns merged from every individual DataFrame) to the processed data folder.
        
        Args:
            dfs: a list of all the data sets used in the project, in this exact order: overview_df, income_statement_df, balance_sheet_df, dividends_df, margins_df, performance_df, valuation_df.
            dfs_names: a list of all the names of the data sets used in the project, in this exact order: Overview Data", "Balance Sheet Data", "Dividends Data", "Income Statement Data", "Margins Data", "Performance Data", "Valuation Data".
        
        Returns:
            None.
        """
        index = dfs_names.index("Performance Data")
        performance_df = dfs[index]
        
        complete_df = self.combined_data_frame(dfs)
        complete_df['6-Month Performance'] = performance_df['6-Month Performance']
        complete_df['YTD Performance'] = performance_df['YTD Performance']
        complete_df['Yearly Performance'] = performance_df['Yearly Performance']
        
        self.save_processed_data([complete_df], ['complete_data'])