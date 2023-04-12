# Project Functions 2

def payout_risk(value):
    
    #if in range, automatically low risk:
    if value >= 0.3 and value <= 0.5:
        return 0.0  
    
    #if not in range, goes through normalization then 
    
    elif value < 0.3:
        return 1.0 - (0.3 - value)  # higher score for values below the range
    else:
        return 1.0 - (value - 0.5) 
    

def remove_outliers(df, column, sdv=3):
    """
    Takes out infinite or NA values and removes outliers (based on z-score values) of a specific column in a DataFrame.
    :param df: DataFrame 
    :param col_name: Column with outliers
    :return: DataFrame without outliers
    """
    # Remove non-integers (e.g. inf or NA)
    filtered_col = pd.to_numeric(df[column], errors='coerce')
    non_integers = ~np.isfinite(filtered_col)
    filtered_col[non_integers] = np.nan
    filtered_df = df.loc[~non_integers]
    
    # Calculate z-score
    z_scores = np.abs((filtered_df['Worthiness'] - filtered_df['Worthiness'].mean()) / filtered_df['Worthiness'].std())

    # Remove z-score outliers
    outliers = (-3 > z_scores) | (z_scores > 3)
    df_no_outliers = filtered_df.loc[~outliers]

    return df_no_outliers


def normalize(df, column):
    max_val = df[column].max()
    min_val = df[column].min()
    return 100 * ( df[column] - min_val) / (max_val - min_val)


def heatmap_sector(df,value,sector,sector_col='Sector',label='Ticker'):
    '''
    This function returns a heatmap of the specified sector of a DataFrame. 
    
    :df: DataFrame to be processed.
    :value: Column used as variables in the heatmap creation process.
    :sector_col: Name of the column to be analyzed.
    :sector: Sector of variables to be analyzed.
    '''
    sector = df[df[sector_col] == sector]
    pivot = pd.pivot_table(sector, values = value, columns = label)
    heatmap = sns.heatmap(pivot)
    
    return heatmap