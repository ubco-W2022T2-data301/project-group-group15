# Final Report

## Introduction

Our combined goal is to investigate what equity data is the most important to consider when developing a portfolio of equities that preferences either a growth portfolio or value investment strategy. From an industry perspective, we intend to provide a solution to the problem of maximizing client returns by optimizing risk and return in the case of optimal asset allocation. Clients have diverse investment goals, which is why we have divided our group project into the explortation of both growth and value investment strategies. Our data is composed of 7 data sets of equity data that includes data on the financial accounts, valuation, performance, dividends and margins of companies part of the S&P500 Index. We are passionate about revealing underlying trends across financial markets, being composed of a group of Computer Science and Data Science students.

## Exploratory Data Analysis

Due to the lack of time series data, our initial definition of equity performance related the values of each quarterly (MRQ) metric to the 3-month change in the price of the equity. With the 3-month change metric referring to the change in the price of each equity since the release of quarterly data, it made sense to use only quarterly data for our analysis. However, constructing several linear regression models yielded no correlation for any metric with the 3-month change in price.

![test](../images/EDA1_plt1.png)
This signaled that the definition of equity performance had to be adjusted to accurately account for the static-data limitation of our data set.

![test](../images/EDA1_plt2.png)

After constructing a density plot for every financial metric across all 7 data sets, we noticed that many distributions were significantly skewed, hinting that the value in our analysis may in fact be by defining equity performance as the degree at which a company scores highly across all financial metrics. This proposition lead to the hypothesis that perhaps the top performing companies are those which exist as outliers in distributions that are skewed towards the low score range[^1]. This same hypothesis lead us to use heat maps in our analysis to map the performance of companies across every metric.

[^1]: As the nature of our analysis requires us to compare metrics that have different value ranges, we devised a custom scoring algorithm that utilises a modified normalization algorithm that classifies outliers as especially important. See the `analysis1.ipynb` notebook for more details.

## Part 1: Algorithmic Asset Allocation | Growth Portfolio Investment Strategy

#### Colin Lefter

> What equity data is the most indicative of the performance of an equity, and of this data, which is the most relevant for a growth portfolio investment strategy such that we can compute an optimized portfolio of equities?

![test](../images/FinalDashboard1.png)
![test](../images/analysis1_plt1.png)
![test](../images/analysis1_plt5.png)
![test](../images/analysis1_plt6.png)

---
Content.

---
## Part 2: Title

#### Keisha Kwek

---
Content.