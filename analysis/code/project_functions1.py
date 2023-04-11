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

    def load_and_process(self, file_name: str, directory_path="../data/raw/", number_of_rows: int=500, exclude_columns: list()=[],
                         additional_data: pd.DataFrame=None, additional_column: str=None, dropna: bool=False) -> pd.DataFrame:
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
        
        if dropna:
            df = df.dropna() 
        
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        return df
    
    def save_processed_data(self, data: list, file_names: list(), directory_path: str="../data/processed/") -> None:
        for df, file_name in zip(data, file_names):
            df.to_csv(directory_path + "processed_" + self.common_data_path + file_name + self.extension)
    
    def combined_data_frame(self, data: list, dropna: bool=True) -> pd.DataFrame:
        df = pd.concat(data, axis=1)
        
        if dropna:
            df = df.dropna(thresh=33)
            
        df = df.loc[:,~df.columns.duplicated()].copy()
        return df

# NOTE: ANALYSIS FUNCTIONS--------------------------------------------------------------------------------------------------------------------
class QuantitativeAnalysis:
    def __init__(self, number_of_companies: int=500):
        """Includes several analysis functions that process select data across all data sets

        :number_of_companies: the number of companies included in the sample, with the default being those from the S&P500 Index
        """
        
        self.number_of_companies = number_of_companies
        
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
        mlr = LinearRegression()
        mlr.fit(X_train, y_train)
        y_pred_mlr = mlr.predict(X_test)

        mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
        mlr_diff.head()

        meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
        meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
        rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
        
        results = {'R squared': mlr.score(X,y), 'Mean Absolute Error': meanAbErr, 'Mean Square Error': meanSqErr, 'Root Mean Square Error': rootMeanSqErr}
        results_df = pd.DataFrame(results, index=['Model Results'])
        return results_df
    
    def rank(self, df: pd.DataFrame, col: str, normalize_only: bool=True, threshold: float=1.5,
             below_threshold: bool=True, filter_outliers: bool=True, normalize_after: bool=False,
             lower_quantile: float=0.05, upper_quantile: float=0.95, inplace: bool=False,
             inverse_normalization_cols: list()=['Price to Revenue Ratio (TTM)', 'Price to Earnings Ratio (TTM)', 'Total Debt (MRQ)', 'Net Debt (MRQ)', 'Debt to Equity Ratio (MRQ)']) -> None:
        
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
    
    def outlier_filtered_df(self, df: pd.DataFrame, col: list(), lower_quantile: float=0.05, upper_quantile: float=0.95):
        upper_fence = df[col].quantile(upper_quantile)
        lower_fence = df[col].quantile(lower_quantile)

        df = df[(df[col] <= upper_fence) & (df[col] >= lower_fence)]
        
        return df
    
    def extract_corr_plot_counts(self, df: pd.DataFrame, correlation_threshold: int=0.7) -> pd.DataFrame:
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
    def __init__(self):
        QuantitativeAnalysis.__init__(self)

    def score_density_plot(self, df: pd.DataFrame, cols: list(), title: str="Density Plot", normalization: bool=True, search_for_score: bool=True) -> plt.graph_objs._figure.Figure:
        """Constructs an interactive compound density plot based on a histogram of the data provided, plotting a density curve with clusters of data points below
        
        :df: a Pandas DataFrame of equity data
        :cols: a list of column names to be plotted
        :returns: a density plot
        """
        df = df.select_dtypes(exclude='object')[:self.number_of_companies]
        df = df.dropna() # mandatory for the function to work
        
        if normalization:
            for column in cols:
                self.rank(df, col=column, upper_quantile=0.99, lower_quantile=0.01)

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
        """Constructs an interactive compound density plot based on a histogram of the data provided, plotting a density curve with clusters of data points below
        
        :df: a Pandas DataFrame of equity data
        :data_name: the name of the type of data that has been input into the plot
        :returns: a density plot
        """
        df = df.select_dtypes(exclude='object')[:self.number_of_companies]
        df = df.dropna() # mandatory for the function to work
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
        df = df.dropna() # to prevent gaps in the heatmap
        
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
            subplot_titles = [predictor for set in predictors for predictor in set],
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

class PortfolioRecommendation(EquityData, QuantitativeAnalysis):
    def __init__(self, portfolio_size: int=50, initial_capital: float=100000.00, capital_per_period: float=100.00, period: int=7, dividends_importance: bool=False, preferred_industries: list=["Technology Services, Electronic Technology"],
                volatility_tolerance: Annotated[float, ValueRange(0.0, 1.0)]=0.7, preferred_companies: list=["Apple, Google, Microsoft, Amazon"], diversification: Annotated[float, ValueRange(0.0, 1.0)]=0.2, investment_strategy: str="Growth"):
        EquityData.__init__(self)
        QuantitativeAnalysis.__init__(self)
        
        """A portfolio recommendation class that allocates user funds to a series of assets in conjunction with the results from the analysis algorithms applied
        :portfolio_size: the number of assets included in the final portfolio
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
    
    def compute_diversification(self):
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
    
    def asset_allocation(self) -> pd.DataFrame:
        equities = EquityData("processed_us_equities_tradingview_data_")
        scored_equities = equities.load_and_process("normalized_data", directory_path="../data/processed/") # update this to inverse normalization for ratios where lower is better
        complete_df = equities.load_and_process("complete_data", directory_path="../data/processed/")

        score_count_df = self.extract_corr_plot_counts(complete_df).T
        
        def extract_top_predictors(threshold: int=6, exclude_columns: list=['Gross Profit (FY)', 'Enterprise Value (MRQ)']) -> pd.DataFrame:
            score_count_df['Assigned Weight'] = score_count_df['Count'] / sum(score_count_df['Count'])
            
            top_predictors_narrowest = score_count_df[score_count_df['Count'] >= threshold].index
            top_predictors_narrowest = top_predictors_narrowest.drop(exclude_columns)
            
            return top_predictors_narrowest
        
        score_count_df['Assigned Weight'] = score_count_df['Count'] / sum(score_count_df['Count'])
        before_weighting = scored_equities
        scored_equities = scored_equities.select_dtypes(exclude='object')

        for col in scored_equities.columns:
            standard_col = col[:-6]
            scored_equities[col] = scored_equities[col] * score_count_df.T[standard_col]['Assigned Weight']

        scored_equities['Aggregated'] = scored_equities[scored_equities.columns].sum(axis=1, numeric_only=True)
        
        degree_of_diversification = self.compute_diversification()
        lower_quantile = degree_of_diversification[0]
        upper_quantile = degree_of_diversification[1]

        self.rank(scored_equities, 'Aggregated', lower_quantile=lower_quantile, upper_quantile=upper_quantile) # NOTE: we control the degree of portfolio diversification via the number of outliers
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
        
        display(Markdown("# My Portfolio"))
        return scored_equities

    def get_user_input(self):
        pass

class DataUploadFunctions(EquityData, QuantitativeAnalysis):
    def __init__(self):
        EquityData.__init__(self)
        QuantitativeAnalysis.__init__(self)
        
        self.processed_data = EquityData("processed_us_equities_tradingview_data_")
        self.complete_df = self.processed_data.load_and_process('complete_data', directory_path="../data/processed/")
    
    def save_normalized_data(self) -> None:
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
        score_count_df = self.extract_corr_plot_counts(self.complete_df).T

        score_count_df['Assigned Weight'] = score_count_df['Count'] / sum(score_count_df['Count'])

        top_predictors_wide = score_count_df[score_count_df['Count'] >= 4].index # 4 x 4 grid
        top_predictors_narrowest = score_count_df[score_count_df['Count'] >= 6].index
        top_predictors_narrowest_adjusted = top_predictors_narrowest.drop(['Gross Profit (FY)', 'Enterprise Value (MRQ)'])
        # Gross Profit (FY) and MRQ are the same, so FY is removed.
        # Enterprise Value (MRQ) subtracts total debt from market capitalization, so it tracks the score of Market Capitalization nearly identically, so it is removed.
        score_count_df = score_count_df.sort_values(by='Assigned Weight', ascending=False)
        self.rank(score_count_df, 'Assigned Weight', filter_outliers=False)
        self.save_processed_data([score_count_df], ['top_predictors'])

        if return_top_predictors:
            return top_predictors_wide, top_predictors_narrowest_adjusted
    
    def save_demo_portfolio(self) -> None:
        portfolio = self.PortfolioRecommendation(500, initial_capital=500000)
        demo_portfolio = portfolio.asset_allocation()
        demo_portfolio = demo_portfolio[['Ticker', 'Aggregated Score']]
        self.save_processed_data([demo_portfolio], ['complete_aggregated_scores'])
    
    def save_complete_data(self, dfs: list(), dfs_names: list()) -> None:
        """the function used to save the "complete_data.csv" file
        :dfs: a list of all the data sets used in the project, in this exact order:
        overview_df, income_statement_df, balance_sheet_df, dividends_df, margins_df, performance_df, valuation_df
        :dfs_names: a list of all the names of the data sets used in the project, in this exact order:
        "Overview Data", "Balance Sheet Data", "Dividends Data", "Income Statement Data", "Margins Data", "Performance Data", "Valuation Data"
        """
        index = dfs_names.index("Performance Data")
        performance_df = dfs[index]
        
        complete_df = self.combined_data_frame(dfs)
        complete_df['6-Month Performance'] = performance_df['6-Month Performance']
        complete_df['YTD Performance'] = performance_df['YTD Performance']
        complete_df['Yearly Performance'] = performance_df['Yearly Performance']
        
        self.save_processed_data([complete_df], ['complete_data'])