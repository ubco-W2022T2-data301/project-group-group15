# Dividend Stock Portfolio
## What makes a dividend stock of value companies worthy to invest in?

## **Introduction**

Stocks have a reputation of being inherently unpredictable as it involves multiple levels of prediction of a stock's future price to yield significant profit by only using available information.

Amongst the available information, most investors look at the Earnings per Share (EPS, Basic and Diluted) to gauge the future prices of a particular stock.

However, the calculation of a **value** **stock's** future price (or commonly referred to as worthiness of investing in this analysis) involves more variables that account for the long-term growth and health of a company and industry. This shows that merely calculating through EPS values would not account for a company's performance in industry's fluctuations, especially differing economic situations.

Hence, this analysis hopes to break down the variables that represents a company's strength in varying economic conditions based on our current data, alongisde the strength of the industry itself.

## **Research Question**
What are the best metrics to use to rank the value of a value stock for a dividend stock portfolio calculation and how can it be best enumerated to yield the most accurate results?

## **Our Data**
Combining 

## **Analysis Plan**


## **Analysis & Discussion**

## I. Determining a good dividend stock

Before determining the key variables to quantify dividend stocks, we must first try to understand what dividend stocks professions are recommending. Upon research, it is seen that U.S. News & World Report, Investopedia, and the Motley Fool recommended the following 10-15 dividends to invest in:
1. Microsoft Corporation
1. Chevron Corporation
1. Verizon Communications Inc.
1. Amgen Inc.
1. American Express Company
1. International Business Machines Corporation
1. Lowe's Companies, Inc.
1. Target Corporation
1. Dominion Energy, Inc.
1. Fidelity National Information Services, Inc.
1. Walgreens Boots Alliance, Inc.
1. Tyson Foods, Inc.
1. Brookfield Infrastructure Partners LP Limited Partnership
1. Home Depot, Inc. (The)
1. Texas Instruments Incorporated
1. Automatic Data Processing, Inc.
1. Air Products and Chemicals, Inc.


Amongst these stocks, a heatmap is created to identify the characteristics that differentiate them from the other stocks. 

<br />

![First HeatMap](../images/final_df.png)

<div align="right"> Picture 1.1: Heatmap of all stocks and its stock attributes in Plotly

<div align="left">

<br />

Here, the left heatmap is the plotting between the top 17 companies and the right heatmap is the plotting between the rest of the companies to be in comparison.

Another version of this heatmap was also made in Microsoft Excel to specifically highlight the top recommended companies. 

![Second HeatMap](../images/ExcelHeatmap.png)

<div align="right"> Picture 1.2: Heatmap of all stocks and its stock attributes in Microsoft Excel

<br />

<div align="left">

In this heatmap, the table is sorted based on the recommended dividend stocks, which is highlighted in yellow. It can be seen that while most variables remained "normal" the Current Ratio, Quick Ratio, and Dividends Paid of the recommended stocks is seen to be the most nontypical in comparison to its stock counterparts. 

However, it is not enough to only take these three variables and use them as metrics to calculate the best dividend stocks. This is because there are a large amount of stocks which have similar values as they do, but the difference is the other stocks may have a good current ratio but a bad quick ratio, or a good quick ratio but an insufficient amount of dividends paid. For this reason, the values behind current ratio, quick ratio, and dividends paid will be broken down into separate variables that can be precisely measured and sorted for further compatisons. 

 **Variables that portrude differing weights in comparison to the other stocks and their meaning:**

1. **Current ratio**: a company's ability to pay short-term liabilities (debt and obligations) with its current assets (cash, inventory, and receivables).
1. **Quick ratio**: often referred to the acid-test ratio, it measures the liquidity of a company by gauging its current assets' abilities to cover its current liabilities.
1. **Dividends paid**: this is the amount of dividends the company pays out to the shareholders of the dividend stock.

These variables show that these variables are highly correlated with a company's debts, assets, cash flow, liquidity, and dividend payments. Through these, we can move forward in selecting the essential the  variables from the equities trading data in the following stages: 

### Stage 1.1: Defining a good dividend stock

After research of reliable trading websites such as U.S. News & World Report, Investopedia, and the Motley Fool, around 10-15 dividend stocks were selected as "the top dividend stocks to consider buying". By laying the basics of what best describes a good dividend stock presented by their variables in the trading data, the analysis and aggregation will be usable to different and updated stock lists. Upon further research and analysis, the following variables present to be the variables commonly seen in these strongly endorsed dividend stocks, as correlated with the three variables previously mentioned:

1. Consistent Dividend Payments: a good track record of dividend payments over time indicates that the company has a stable financial position shown from generating sufficient cash flow which is an important variable to support dividend payments.
2. Strong Financials: A strong balance sheet, healthy cash flow, and sustainable earnings growth ensures that the company can continue to pay dividends through tough economic times.
3. Low payout Ratio (percentage of earnings paid out as dividends): a high payout ratio means the company is not reinvesting enough capital to support future growth, which may be a red flag for long-term investment.
4. Competitive Dividend Yield: This is compared to other companies in the same industry of the broader market. It is important to note that although a high yield is attractive, it may not be sustainable for the company itself.

### Stage 1.2: Calculating the risk score (based on the company’s industry position) 

As dividend stocks are typically run long-term, the risk factors would involve the history of a company's performance. Hence the following key variables to determine risk factors are measured:

#### **Key risk factors**:
##### Quantifying the variables into a score of 0-100% using a weighted average:

1. **Payout Ratio (25%)**: when a company has a higher percentage of its earnings as dividends this may present as a higher risk as it means they are not investing enough in the company. This variable takes into account the dividend consistency of a company in its calculation, as it is *dividends per share divided by basic EPS*. As this is risky for the long-term and critical for dividend sustainability, it receives a score of 10%.
1. **Debt levels (25%)**: companies with higher debt may struggle to continue paying dividends and are in more of a risk to declare bankruptcy. As this is an element that significantly impacts a company's health and company health is a major aspect seen from the derivations above, it will receive a score of 25%.
1. **Value stock volatility (50%)**: This is the number . 


### Stage 2.2: Calculating the Industry Safety Score

The Industry Safety Score is a metric made to account for the safety of an industry by viewing its industry stability and financial health of companies in a particular industry.

1. **Industry stability (50%)**: Calculated using the reciprocal of the volatile variable in the balance sheet, stable industries have steady demand for their products or services, hence they are less susceptible to economic downturns or competition from new entrants. Long history of stable growth and profitability shows a stable industry, shows better likelihood that the company will not fall behind in dividend payments.
1. **Financial health of companies within an industry (50%)**: As mentioned above, the financial health of a company is a large variable to account in a company's ability to fulfil their dividend payments in full. These financial health variables rely on cash flows, debt-to-equity ratio, earnings growth, and history of dividend payments. It also accounts for the possibility of the company having the foundation for future dividend increases.

### Stage 3: Calculating the Worthiness of a Value Stock

The worthiness of investing in a value stock will take account the previously calculated variables alongside other variables which effect a company's current ratio, quick ratio, and dividends paid - variables which were found to be most commonly different of endorsed value stocks from the rest of the value stocks.

### Fixed Variables (60%)

1. **Financial Health (20%)**: As financial health impacts a company's sustainability to maintain and grow its dividend payments over time, it is a crucial aspect to include when calculating a value stock's worthiness in investing. 
2. **Price to Earnings Ratio (20%)**: 
3. **Industry Safety Score (20%)**: Cyclical industries are at risk of volatile earnings and cash flows, hence not generating enough funds to paying dividends. Especially in times of economic uncertainty, some companies will perserve cash over paying dividends - resulting in a cut of dividends. This is why this variable is 30% as it is a variable strongly related to dividend payments.

### Client-driven Variables (40%)

These are the variables which will vary depending on the client input. In this research, it will be represented in only three levels:

1. Moderate risk, moderate yield
1. High risk, high outcome yield
1. Low risk, low outcome yield

Where the following variables will vary in client input to effect worthiness of investing in a value stock from 1-20% but both should add up to 40% of the worthiness calculation. 

1. **Dividend yield (20%)**: Dividend yield is the annual dividend payment as a percentage of stock price. A higher dividend yield will provide a higher stream of income for the investor. If the client prefers a higher yield, they will increase the percentage of dividend yield accordingly to calcualte the worthiness of a value stock.
1. **Risk(20%)**: Risk will account for the client's [bearability] to bear for risks. The higher the risk, the more the worthiness score will account for high-risk value stocks.  

## II. Analysis of Variables
After defining the given variables, data cleaning, and data aggregation, the following final DataFrame is outputted:

![Final DataFrame](../images/final_df.png)

<div align="right">*Picture 2.1: DataFrame of all calculated variables*

<div align="left">

<br />

From this DataFrame, we created a RidgeLine plot to visualize the average worthiness of each industry. 

![RidgeLine plot](../images/Ridgeline_Worthiness.png)

From this plot, we can see that the industry with the highest value stock worthiness is Health Technology and Technology Services, alongside that, there is a lack of variability and data points in the Finance and Health Services.

To better visualize this distribution, the following plot shows the worthiness of value stocks per industry using a boxplot, showing the individual plots to better understand how each distribution came to be. 

![BoxPlot](../images/BoxPlot_Worthiness.png)

From this distribution, we can see that there are outliers most prominently seen in the following industries:

1. Technology Services
1. Consumer Non-Durables
1. Process Industries
1. Producer Manufacturing
1. Distribution Services

This may be because the nature of these industries are monopolies of oligopolies, meaning that only a few large firms make most of the sales in a particular industry. Through this plot, we are able to see which industries are of this type, and hence increase the chances of bigger returns - as investing in the big players will have less risk as they bring a more stable cash flow.

> This raises the question: How accurate are these calculations?

This pattern best seen in the Hexplot, where we are able to compare both the Worthiness of the Stocks in each sector plotted against the companies' Earnings per Share (EPS). As EPS is a common metric investors use to calculate the return of a stock where a higher EPS indicates greater value.

![HexPlot](../images/HexPlot__EPS_Worthiness.png)

From this plot, we can see how the variability of stock amount in each industry may impact the average score, and hence the data shown in the previous plots. Nonetheless, it can be seen how some companies have a large variation in the intensity between its Value Stock Worthiness as compared to its companies' EPS value. This may show how some industries' Value Stock Worthiness may be in correspondence with its companies' EPS value, meaning that some industries are influenced by volatility and economic fluctuations more than others.

As the EPS values do not incorporate industry volatility in its calculations, it shows that the more the Stock Worthiness diverges from the EPS value, the more it is influenced by economic of industrial fluctuations. In this case, the following industries are more volatile to economic change than other industries:

1. Non-Energy Minerals
2. Distribution Services
3. Retail Trade
4. Industrial Services

An evidence to this volatility is the fact that the pandemic is a recent economic change that influenced the stock market. It was seen that the pandemic caused a major halt in industries of Industrial Services and Retail Trade due to pandemic regulations. In contrast, we also know that due to current political shifts towards cheaper energy is adopted, and hence why there is a large degree of impact in Producer Manufacturing as compared to, say, Industrial Services.

> How does the data present when compared to the three initial prominent variables?

The three initial prominent variables were seen to be Current Ratio, Quick Ratio, and Dividend Yield. In creating the worthiness score, we made sure to break down each of these variables into the components that it was created by so that we can truly analyze and classify the true raw variables that indicate the worthiness of a value stock. The relationship and correlation between the variables can be seen in the following plots.

![Correlation of Div. Yield, Ratios, and Worthiness](../images/Correlation_QuickRatio_Worthiness.png)

It is seen here that there is a positive correlation, albeit weak, between the worthiness of the value stocks and the companies' quick ratio.

![Correlation of Div. Yield, Ratios, and Worthiness](../images/Correlation_CurrentRatio_Worthiness.png)

Here, we can see that there is stronger positive correlation between the worthiness of the value stocks and the companies' quick ratio than the previous plot, however still being a weak correlation.

![Correlation of Div. Yield, Ratios, and Worthiness](../images/Correlation_DividendYield_Worthiness.png)

In this plot, it shows a very small correlation between the two variables, however there are separated patterns of strong correlation. This may be because the data used for these plots are a combination of all value stock worthiness.

Nonetheless, the following plot shows a better presentation of these values can be seen when each plot is labelled by the industries they are in.

![Relationship of Div. Yield, Ratios, and Worthiness](../images/Relationship_CurrentRatio_Worthiness.png)

From this plot, we can see how the stocks of each sector are confined in their own specified area. For instance, the Worthiness column shows how the variation is seen between industries. When we take a closer look, we can find that each companies' value stock worthiness in a particular indudstry has a positive correlation with Dividend Yield. This pattern is also seen in the current and quick ratios, meaning that we have compartmentalized and broken down the components which make the three initial variables and applied them to our calculation of value stock worthiness accurately.

## **Conclusion & Remarks**

In conclusion, we were able to create a quite precise model in which accomodates the variables that create a worthy value stock to invest in. This process was authenticated by plotting the initial variables that started our variable aggregation analysis, alongside seeing its relationship in each industry. Even so, there are a few variables that we were not able to quantify and include in our analysis. For example, the regulatory risks of an industry (e.g. tax laws, tariffs, or environmental regulations) which can influence the value of stocks to a large extent was not calculated as there were no values that we could enumerate to represent them. That, on top of the fact that the nature of stocks are in general very unpredictable shows that there is also inaccuracy to this analysis to some extent. 

Overall, this analysis has shown that even the most unpredictable datasets that show large deviations can be analyzed when categorized and distinguished appropriately. 