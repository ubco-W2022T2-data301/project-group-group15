# Remove outliers function

def remove_outliers(df, column):
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
    filtered_df = filtered_df.dropna(axis=0)

    return filtered_df


# Remove outliers function

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


def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)