import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from Assignment2.SNP500 import SNP_data


def daily_stock_AMZN_GOOGL():
    stocks = SNP_data.parse()
    AMZN_stocks = stocks[stocks['symbol'] == 'AMZN'][['date', 'high']]
    GOOGL_stocks = stocks[stocks['symbol'] == 'GOOGL'][['date', 'high']]

    fig, axes = plt.subplots()
    axes.xaxis.set_major_locator(MaxNLocator(5))
    plt.plot(AMZN_stocks['date'], AMZN_stocks['high'], label="AMZN stock")
    plt.plot(GOOGL_stocks['date'], GOOGL_stocks['high'], label="GOOGL stock")
    plt.legend()
    plt.show()


daily_stock_AMZN_GOOGL()
