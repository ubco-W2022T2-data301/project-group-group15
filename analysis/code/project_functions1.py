import pandas as pd
import plotly as plt
import seaborn as sns
import numpy as np
import datetime as dt
import matplotlib.pyplot as mplt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from IPython.display import display, HTML, Markdown, Latex
from tqdm import tqdm, trange
from typing import *
from dataclasses import dataclass
from scipy import stats
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn import metrics
from plotly.subplots import make_subplots

@dataclass
class ValueRange:
    min: float
    max: float
    
    def validate(self, x):
        """Checks if inputs to variables that must lie within a specific range are valid
        
        :x: the value that must be checked as satisfying the specified range
        :raises ValueError: if the value does not lie within the specified range
        """
        if not (self.min <= x <= self.max):
            raise ValueError(f'{x} must be between 0 and 1 (including).')

class EquityData:
    def __init__(self, common_data_path: str="us_equities_tradingview_data_", extension: str=".csv"):
        """Includes a series of data loading and processing functions
        
        :common_data_path: for raw data files that have a common path up to a certain point, specify this to optimize the loading process of multiple files
        """
        self.common_data_path = common_data_path
        self.extension = extension

    def load_and_process(self, file_name: str, directory_path="../data/raw/", number_of_rows: int=500, exclude_columns: list()=[], additional_data: pd.DataFrame=None, additional_column: str=None) -> pd.DataFrame:
        """Uses method chaining to read in the raw data up to a specified number of columns while also dropping any desired columns
        
        :file_name: the name of the file, with the extension included
        :number_of_rows: the total number of rows that the dataframe should have
        :exclude_columns: a list of column names that should be dropped from the data frame
        :returns: a new Pandas DataFrame
        """
        assert type(number_of_rows) == int, "Number of rows must be an integer"
        df = pd.DataFrame()
        
        def method_chain():
            """A helper function to create a central method chain"""
            df = (
                pd.read_csv(directory_path + self.common_data_path + file_name + self.extension)
                .iloc[:number_of_rows]
                .drop(columns=exclude_columns)
                .dropna()
                )
            return df
        
        if additional_data is not None and additional_column is not None:
            df = (
                method_chain()
                .assign(new_col=additional_data[additional_column])
                .rename(columns={"new_col": additional_column})
                )
        else:
            df = method_chain()
        
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        return df
    
    def save_processed_data(self, data: list, file_names: list(), directory_path: str="../data/processed/") -> None:
        for df, file_name in zip(data, file_names):
            df.to_csv(directory_path + "processed_" + self.common_data_path + file_name + self.extension)
    
    def combined_data_frame(self, data: list, drop_strings: bool=False) -> pd.DataFrame:
        df = (pd.concat(data, axis=1)
            .dropna()
            )
        df = df.loc[:,~df.columns.duplicated()].copy()
        return df

# NOTE: ANALYSIS FUNCTIONS--------------------------------------------------------------------------------------------------------------------
class QuantitativeAnalysis:
    def __init__(self, number_of_companies: int=500, initial_capital: float=100000.00, capital_per_period: float=100.00, period: int=7, dividends_importance: bool=False, preferred_industries: list=["Technology Services, Electronic Technology"],
                volatility_tolerance: Annotated[float, ValueRange(0.0, 1.0)]=0.7, preferred_companies: list=["Apple, Google, Microsoft, Amazon"], diversification: Annotated[float, ValueRange(0.0, 1.0)]=0.4, investment_strategy: str="Growth"):
        """Includes several analysis functions that process select data across all data sets

        :number_of_companies: the number of companies included in the sample, with the default being those from the S&P500 Index\n
        :initial_capital: the initial amount of cash to be invested by the client, in USD\n
        :capital_per_period: the amount of cash to be invested by the client at a fixed rate in addition to the initial capital invested, in USD\n
        :period: the frequency (in days) at which additional cash is invested, if desired\n
        :dividends_importance: specifies whether dividends are important to the client, dictating whether analysis algorithms should place greater importance on dividends\n
        :preferred_industries: specifies a list of industries that the analysis algorithms should prioritize when constructing the investment portfolio\n
        :volatility_tolerance: accepts a range of values from 0 to 1, with 1 implying maximum volatility tolerance (i.e. the client is willing to lose 100% of their investment to take on more risk)\n
        :preferred_companies: specifies a list of companies that the analysis algorithms will accomodate in the final portfolio irrespective of their score\n
        :diversification: accepts a range of values from 0 to 1, with 1 implying maximum diversification (i.e. funds will be distributed evenly across all industries and equally among all companies)\n
        :investment_strategy: specifies the investment strategy that will guide the output of the analysis algorithms, in which this analysis notebook strictly focuses on growth investing\n
        :raises: ValueError if an input parameter does not satisfy its accepted range
        """
        
        self.number_of_companies = number_of_companies
        self.initial_capital = initial_capital
        self.capital_per_period = capital_per_period
        self.period = period
        self.dividends_importance = dividends_importance
        self.preferred_industries = preferred_industries
        self.volatility_tolerance = volatility_tolerance
        self.preferred_companies = preferred_companies
        self.diversification = diversification
        self.preferred_companies = preferred_industries
        self.investment_strategy = investment_strategy
        
    def lin_reg_coef_determination(self, df: pd.DataFrame, X: str, y: str='3-Month Performance', filter_outliers: bool=True) -> np.float64:
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
        """Returns a Pandas DataFrame with the coefficients of determination for each y-on-x regression
        Example: 3-Month Performance against Price to Earnings Ratio (TTM)
        
        :df: the data frame that contains the columns to process\n
        :x_values: a list of strings of the names of each column to process\n
        :y_value: a common y-value to map each x value against in the regression analysis\n
        :returns: A Pandas DataFrame with the coefficients of determination for each y-on-x regression\n
        
        """
        coef_dict = dict.fromkeys(x_values, 0) # initialize a dict with all the columns assigned to a value of 0
        
        for predictor in tqdm(x_values, desc="Constructing linear regression models", total=len(x_values)):
            coef_dict[predictor] = self.lin_reg_coef_determination(df, X=predictor, y=y_value)
        
        processed_df = pd.DataFrame(list(zip(coef_dict.keys(), coef_dict.values())), columns=[f'Equity Data Against {y_value}', 'Coefficient of Determination'])
        
        return processed_df
        
    def multiple_linear_regression(self, df: pd.DataFrame, predictors: list(), target_y: str='Market Capitalization') -> pd.DataFrame:
        """Consturcts a multiple linear regression model
        :df: a Pandas DataFrame containing the data to be processed
        :predictors: the x values that will be used to predict the target y value
        :target_y: the y value to be predicted
        :returns: a Pandas DataFrame containing a statistical summary of the performance of the model
        """
        df = df.select_dtypes(exclude='object')
        
        if target_y in predictors:
            predictors.remove(target_y) # so you don't have a perfect correlation for the same variable

        X = df[predictors]
        y = df[target_y]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
        mlr = LinearRegression()
        mlr.fit(x_train, y_train)
        y_pred_mlr = mlr.predict(x_test)

        mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
        mlr_diff.head()

        meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
        meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
        rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
        
        results = {'R squared': mlr.score(X,y) * 100, 'Mean Absolute Error': meanAbErr, 'Mean Square Error': meanSqErr, 'Root Mean Square Error': rootMeanSqErr}
        results_df = pd.DataFrame(results, index=['Model Results'])
        return results_df
    
    def rank(self, df: pd.DataFrame, col: str, normalize_only: bool=True, threshold: float=1.5,
             below_threshold: bool=True, filter_outliers: bool=True, normalize_after: bool=False,
             lower_quantile: float=0.05, upper_quantile: float=0.95, inplace: bool=False) -> None:
        """The scoring algorithm for determining the weight of each equity in the construction of the portfolio for this specific column examined.
        Features a custom outlier-filtering algorithm that is robust to outliers in the data set while still returning normalized values.
        Normalizes one column at a time.
        
        :df: The original dataframe\n
        :col: The name of the column being extracted from the dataframe provided\n
        :normalize_only: if True, does not apply a threshold to the screening algorithm, and only normalizes values with a minmax scaler\n
        :threshold: the minimum value that equities must have for that column in order to be considered for further analysis\n
        :below_threshold: if True, removes equities that are below the threshold for that column\n
        :filter_outliers: if True, does not consider equities in the data normalization algorithm, but assigns a min or max value to all outliers depending on the below_threshold parameter\n
        :normalize_after: if True, normalizes the data only after the threshold filter has been applied\n
        :lower_quantile: specifies the lower quantile of the distribution when filtering outliers\n
        :upper_quantile: specifies the upper quantile of the distribution when filtering outliers\n
        :inplace: if true, specifies that the normalization algorithm should directly modify the column being processed, otherwise, a new column is created
        """
        
        #NOTE: should make an option for no threshold
        self.x = df[col]
        new_col = col + " Score"
        
        # normalization can be done either before or after equities have been filtered by the threshold
        # the difference is that by filtering initially, the min and max values of that smaller set will become 0 and 1 respectively
        df[new_col] = np.NaN # initialize the score column with only NaN values
        
        def outlier_filter(self):
            """Nested helper function to filter outliers"""
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
    
    def outlier_filtered_df(self, df: pd.DataFrame, col: list(), lower_quantile: float=0.05, upper_quantile: float=0.95):
        upper_fence = df[col].quantile(upper_quantile)
        lower_fence = df[col].quantile(lower_quantile)

        df = df[(df[col] <= upper_fence) & (df[col] >= lower_fence)]
        
        return df

# NOTE: VISUALIZATION FUNCTIONS--------------------------------------------------------------------------------------------------------------------
class DataVisualization(QuantitativeAnalysis):
    def __init__(self):
        QuantitativeAnalysis.__init__(self)

    def score_density_plot(self, df: pd.DataFrame, cols: list(), title: str="Density Plot") -> plt.graph_objs._figure.Figure:
        """Constructs an interactive compound density plot based on a histogram of the data provided, plotting a density curve with clusters of data points below
        
        :df: a Pandas DataFrame of equity data
        :cols: a list of column names to be plotted
        :returns: a density plot
        """
        df = df.select_dtypes(exclude='object')[:self.number_of_companies]
        
        for column in cols:
            self.rank(df, col=column, upper_quantile=0.99, lower_quantile=0.01)

        hist_data = [df[x + " Score"] for x in cols]
        group_labels = [x + " Score" for x in cols]
        colors = ['#94F3E4', '#333F44', '#37AA9C']

        fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)
        fig.update_layout(title_text=title, template='plotly_dark')

        fig.update_xaxes(title='Score (0 = low, 1 = high)')
        fig.update_yaxes(title='Density')
        
        return fig

    def legacy_score_density_plot(self, df: pd.DataFrame, data_name: str) -> plt.graph_objs._figure.Figure:
        """Constructs an interactive compound density plot based on a histogram of the data provided, plotting a density curve with clusters of data points below
        
        :df: a Pandas DataFrame of equity data
        :data_name: the name of the type of data that has been input into the plot
        :returns: a density plot
        """
        df = df.select_dtypes(exclude='object')[:self.number_of_companies]
        n = len(df)
        
        for column in df.columns:
            self.rank(df, col=column, upper_quantile=0.99, lower_quantile=0.01)
            
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
        """A wrapper function for the default heatmap plot, constructing an interactive heatmap plot of equity data against each company (ticker)
        
        :df: a Pandas DataFrame of equity data
        :data_name: the name of the type of data that has been input into the plot
        :number_of_companies: the number of companies to include in the heatmap
        :correlation_plot: if true, creates a correlation plot instead of a heatmap plot
        :returns: a heatmap plot
        """
        def construct_correlation_plot() -> plt.graph_objs._figure.Figure:
            """A helper function to convert the heat map into a correlation plot"""
            # Correlation
            df_corr = df.corr(numeric_only=True).round(1)
            # Conver to a triangular correlation plot
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
    
    def subplot_generator(self, df: pd.DataFrame, predictors: list, title: str, height_reduction_factor = 8, width_multiplier = 1, horizontal_spacing= 0.02, vertical_spacing = 0.005, rows=4, cols=4):
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_yaxes=True,
            subplot_titles = ['Sorted by ' + predictor for set in predictors for predictor in set],
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
            )
        
        def pair_construction(row: int, col: int, predictor) -> None:
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
        
        num_cols = 4
        
        for row in range(num_cols):
            for col in range(num_cols):
                pair_construction(row, col, predictors[row][col])

        height_multiplier = rows*cols - height_reduction_factor
        fig.update_layout(
            title_text=title,
            template='plotly_dark',
            width=1500*width_multiplier,
            height=1500*height_multiplier)
        
        return fig

    def binary_subplot_generator(self, df: pd.DataFrame, predictors: list, title: str, height_reduction_factor = 8, width_multiplier = 1, horizontal_spacing= 0.02, vertical_spacing = 0.005, rows=4, cols=2):
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
        """Constructs a 3D interactive plot of equity data on 3 axes
        :df: a Pandas DataFrame of equity data
        :x: the name of the column data to be plotted on the x-axis, as a string
        :y: the name of the column data to be plotted on the y-axis, as a string
        :z: the name of the column data to be plotted on the z-axis, as a string
        :returns: a 3D scatter plot
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
        """Produces a correlation plot that maps all of the data points in the Data Frame provided
        :df: a Pandas DataFrame of the data to be processed
        :data_name: the name of the data being plotted
        """
        # Compute the correlation matrix
        corr = df.corr(numeric_only=True)

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(1, 10, as_cmap=True)

        #Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        mplt.title(f"Correlation Plot of {data_name}")

class PortfolioConstruction(DataVisualization):
    def __init__(self):
        DataVisualization.__init__(self)
    
    def asset_allocation(self):
        pass
    
    def construct_portfolio(self):
        pass