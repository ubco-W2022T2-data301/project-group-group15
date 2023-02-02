# Group 15 - Algorithmic Asset Allocation and Portfolio Construction | Client-Driven Optimization

## Milestones

Details for Milestone are available on Canvas (left sidebar, Course Project).

## Describe your topic/interest in about 150-200 words

Our goal is to develop software for constructing an optimized portfolio of equities according to client requirements by leveraging quantitative analysis, financial modelling and data visualization techniques. The software is adaptable to client preferences by accepting a series of predetermined parameters which the client will input, and the software will use across a chain of analyses to generate a final investment portfolio. A significant component of the final recommendation is the distribution of client capital across the optimal equities selected.

From an industry perspective, we intend to provide a solution to the problem of maximizing client returns by optimizing risk and return in the case of optimal asset allocation. Clients have diverse investment goals, which is why client-driven algorithmic portfolio allocation is the basis of our project. We achieve this by having the client define their preferences according to the risk-management parameters we have defined, and further process this data to select equities from the top 200 publicly-traded companies by market capitalization in the United States. We are passionate about revealing underlying trends across financial markets, being composed of a diverse group of Computer Science, Data Science and Management students.

## Describe your dataset in about 150-200 words

The raw data is composed of 7 data sets of equity data that includes data on the financial accounts, valuation, performance, dividends and margins of 8115 publicly-traded companies in the United States. The data is sourced from [TradingView](https://www.tradingview.com/), which grants the permission to use and distribute the data per their [attribution policy](https://www.tradingview.com/policies/) cited in their Terms of Service[^1]. This data has been extracted from TradingView's publicly available [Stock Screener](https://www.tradingview.com/screener/).

There are 76 data columns shared across all 7 data sets, of which 70 of them are unique. However, only 200 rows of data will be used, in which the number of columns will be significantly reduced according to their relevance in our analysis. Consequently, the total number of raw data points is 616740, of which our analysis will focus on 15200 raw data points. The data sets capture static financial market data for the 30th of January, 2023, which can be updated by uploading new data, but for our analysis, will remain constant.

## Team Members

- Colin Lefter: I am a 1st year Computer Science student who is passionate about algorithmic finanical modelling, having experience with Python, Java and R.
- Person 2: one sentence about you!
- Person 3: one sentence about you!

## Images

{You should use this area to add a screenshot of an interesting plot, or of your dashboard}

<img src ="images/test.png" width="100px">

## References

[^1]: From TradingView's [Terms of Service](https://www.tradingview.com/policies/) page:

> TradingView grants all users of tradingview.com, and all other available versions of the site, to use snapshots of TradingView charts in analysis, press releases, books, articles, blog posts and other publications. In addition, TradingView grants the use of all previously mentioned materials in education sessions, the display of TradingView charts during video broadcasts, which includes overviews, news, analytics and otherwise use or promote TradingView charts or any products from the TradingView website on the condition that TradingView attribution is clearly visible at all times when such charts and products are used.

> **Attribution must include a reference to TradingView, including, but not limited to, those described herein.**