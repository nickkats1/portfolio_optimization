import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yfinance as yf
from pypfopt import risk_models,EfficientFrontier,expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation,get_latest_prices
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

start_date = dt.datetime(2020,10,10)
end_date = dt.datetime(2024,1,8)

def get_tickers(tickers,start,end):
    
    
    return yf.download(tickers,start=start_date,end=end_date)['Close']




stock_tickers = ['AAPL', 'TGT', 'MCD', 'IBM', 'TSLA']
ETF_tickers = ['BND', 'HYG', 'TIP', 'IEF', 'LQD']
FX_tickers = ['USDJPY=X', 'USDGBP=X', 'USDEUR=X', 'USDAUD=X', 'USDRUB=X']
CRYPTO_tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD']

stocks_df = get_tickers(stock_tickers, start=start_date, end=end_date)
FX_df = get_tickers(FX_tickers, start=start_date, end=end_date)
ETF_df = get_tickers(ETF_tickers, start=start_date, end=end_date)
CRYPTO_df = get_tickers(CRYPTO_tickers, start=start_date, end=end_date)

risk_free_rate = 0.05

assets = (stocks_df,ETF_df,FX_df,CRYPTO_df)


stocks_df.dropna()
FX_df.dropna()
ETF_df.dropna()
CRYPTO_df.dropna()








def get_returns(assets):
    returns = {}
    for model_name, model in zip(["stocks","ETF's","FX","Crypto"],assets):
        model_returns = model.pct_change().dropna()
        returns[model_name] = model_returns
        print(f'Returns for {model_name}:')
        print(model_returns.head())
    return returns


get_returns(assets)



def correlation_heatmap(assets):
    for model_name, model in zip(["stocks","ETF's","FX","Crypto"],assets):
        plt.figure(figsize=(10,6))
        sns.heatmap(model.corr(), fmt="f", annot=True, cmap="coolwarm")
        plt.title(f'Correlation Heatmap for {model_name}')
        plt.show()
        

correlation_heatmap(assets)


def plot_assets(assets):
    for model_name, model in zip(["stocks","ETF's","FX","Crypto"], assets):
        plt.figure(figsize=(10,6))
        plt.plot(model)
        plt.title(f'{model_name} Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(model.columns)
        plt.show()


plot_assets(assets)



def portfolio_optimization(assets):
    for model_name, model in zip(["stocks","ETF's","FX","Crypto"], assets):
        mu = expected_returns.mean_historical_return(model)
        risk = risk_models.sample_cov(model)
        ef = EfficientFrontier(mu, risk)
        weights = ef.max_sharpe()
        sharpe_ratio = ef.portfolio_performance()[2]
        ### optimal portfolio performance
        
        optimal_expected_returns = ef.portfolio_performance()[0]
        optimal_portfolio_risk = ef.portfolio_performance()[1]
        optimal_sharpe_ratio = (optimal_expected_returns - risk_free_rate) / optimal_portfolio_risk
        print(f'Expected returns of {model_name}: {mu}')
        print(f'Risk (Covariance Matrix) of {model_name}: {risk}')
        print(f'Optimized weights for {model_name}: {weights}')
        print(f'{model_name}; --Sharpe Ratio-- {sharpe_ratio}')
        print(f'Optimal Expected Returns on {model_name}: {optimal_expected_returns}')
        print(f'Risk of Optimal Portfolio {model_name}: {optimal_portfolio_risk}')
        print(f'Sharpe Ratio of optimal Portfolio {model_name}: {optimal_sharpe_ratio}')

        
    return model

portfolio_optimization(assets)






### all portfolio's combined

df = pd.concat(assets,axis=1,keys=["stocks","ETFs","FX","Crypto"])
df.dropna(inplace=True)



def full_portfolio_heatmap(df):
    plt.figure(figsize=(15,6))
    sns.heatmap(df.corr(),annot=True,fmt='.2f',cmap="coolwarm")
    plt.title('Correlation Matrix for Full Portfolio')
    plt.show()
    

full_portfolio_heatmap(df)



def full_portfolio_performance(df, risk_free_rate=0.05):
    mu = expected_returns.mean_historical_return(df)
    risk = risk_models.sample_cov(df)
    ef = EfficientFrontier(mu, risk)
    weights = ef.max_sharpe()
    portfolio_return, portfolio_volatility, sharpe_ratio = ef.portfolio_performance()
    optimal_expected_returns = portfolio_return
    optimal_portfolio_risk = portfolio_volatility
    optimal_sharpe_ratio = (optimal_expected_returns - risk_free_rate) / optimal_portfolio_risk

    """ Using Greedy Investor for how much money you receive back"""
    
    latest_prices = get_latest_prices(df)
    print(latest_prices)
    da = DiscreteAllocation(weights, latest_prices,total_portfolio_value=1000000)
    print(f'Optimized weights for the full portfolio: {weights}')
    print(f'Expected return of the full portfolio: {optimal_expected_returns}')
    print(f'Risk of the full portfolio: {optimal_portfolio_risk}')
    print(f'Sharpe ratio of the full portfolio: {optimal_sharpe_ratio}')
    allocation, leftover = da.greedy_portfolio()
    print('Discrete Allocation: ',allocation)
    print('Left over cash',leftover)



    return weights, optimal_expected_returns, optimal_portfolio_risk, optimal_sharpe_ratio,da,allocation,leftover



full_portfolio_performance(df,risk_free_rate)



### ok, time for CAPM

## stock tickers and S&P 500 ticker
stock_ticker = ['AAPL','IBM','^GSPC','TSLA','DIS','MCD','GOOGL']


df = yf.download(tickers=stock_ticker,start=start_date,end=end_date)['Close']
df.head(10)

print(df.pct_change().dropna())



def single_index_model(stock_ticker, sp500, risk_free_rate):
    stock_data = df[stock_ticker].to_frame()
    sp500_data = df[sp500].to_frame()
    stock_data['Excess_Return'] = stock_data[stock_ticker] - risk_free_rate
    sp500_data['Excess_Return'] = sp500_data['^GSPC'] - risk_free_rate
    model = sm.OLS(endog=stock_data['Excess_Return'], exog=sm.add_constant(sp500_data['Excess_Return'])).fit()
    print(f'single index for each stock ticker{stock_data}')
    print(model.summary())
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=sp500_data['Excess_Return'], y=stock_data['Excess_Return'], label=stock_ticker)
    sns.lineplot(x=sp500_data['Excess_Return'], y=model.fittedvalues, color='red', label='Security Market Line')
    plt.title(f'Single Index Model for {stock_ticker}')
    plt.xlabel('Market Excess Return')
    plt.ylabel(f'{stock_ticker} Excess Return')
    plt.legend()
    plt.show()



single_index_model("AAPL","^GSPC",risk_free_rate)
single_index_model("IBM","^GSPC",risk_free_rate)
single_index_model("MCD","^GSPC",risk_free_rate)
single_index_model("TSLA",'^GSPC',risk_free_rate)
single_index_model("DIS","^GSPC",risk_free_rate)
single_index_model("AAPL","^GSPC", risk_free_rate)




#time for clustering

df = yf.download(stock_tickers,start=start_date,end=end_date)
df.head(10)



from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

X = df[['Open','Close','High','Volume']]
X_scaled = MinMaxScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

cc = []
for i in range(2,21):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=20,random_state=42).fit(X_pca)
    cc.append(kmeans.inertia_)



### The Elbow Method
plt.plot(range(2,21),cc,marker='*')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.title('The Elbow Method')
plt.show()


kmeans = KMeans(n_clusters=16,init='k-means++',n_init=20,random_state=42).fit(X_pca)
labels = kmeans.fit_predict(X_pca)
X['Cluster 1'] = labels
X['Cluster 2'] = labels


plt.scatter(X_pca[:,0],X_pca[:,1],c=X['Cluster 1'],s=300,marker='*',edgecolors='r')
plt.scatter(X_pca[:,0],X_pca[:,1],c=X['Cluster 2'],s=300,marker='X',edgecolors='m')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='o',color='m')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-Means Clustering')
plt.show()



from sklearn.metrics import silhouette_score
lables = kmeans.fit_predict(X_pca)
X['Cluster'] = labels
sh = silhouette_score(X_pca, labels)
print(f'The silhoutte score is {sh*100:.2f}%')
print(kmeans.inertia_)

