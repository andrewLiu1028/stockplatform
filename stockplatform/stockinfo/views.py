import yfinance as yf
import talib
import pandas as pd
from django.shortcuts import render
from .forms import StockForm
import json
import numpy as np
from django.http import JsonResponse

#### 交易訊號判斷 #####
# 短期均線和中期均線交叉判斷
def ma_cross_strategy(short_ma, medium_ma, time):
    signals = [0] * len(time)  # 初始化信號列表匹配時間序列長度
    for i in range(1, len(time)):
        if short_ma[i] > medium_ma[i] and short_ma[i - 1] <= medium_ma[i - 1]:
            signals[i] = 1  # 買入訊號
        elif short_ma[i] < medium_ma[i] and short_ma[i - 1] >= medium_ma[i - 1]:
            signals[i] = -1  # 賣出訊號
    return signals

# K 值和 D 值交叉判斷
def kd_cross_strategy(k, d, time):
    signals = [0] * len(time)  # 初始化信號列表匹配時間序列長度
    for i in range(1, len(time)):
        if k[i] > d[i] and k[i - 1] <= d[i - 1]:
            signals[i] = 1  # 買入訊號
        elif k[i] < d[i] and k[i - 1] >= d[i - 1]:
            signals[i] = -1  # 賣出訊號
    return signals

# RSI 判斷
def rsi_strategy(rsi_values, time, buy_threshold=30, sell_threshold=70):
    signals = [0] * len(time)  # 初始化信號列表匹配時間序列長度
    for i in range(len(time)):
        if rsi_values[i] < buy_threshold:
            signals[i] = 1  # 買入訊號
        elif rsi_values[i] > sell_threshold:
            signals[i] = -1  # 賣出訊號
    return signals

# MACD 判斷
def macd_strategy(macd, signal, time):
    signals = [0] * len(time)  # 初始化信號列表匹配時間序列長度
    for i in range(1, len(time)):
        if macd[i] > signal[i] and macd[i - 1] <= signal[i - 1]:
            signals[i] = 1  # 買入訊號
        elif macd[i] < signal[i] and macd[i - 1] >= signal[i - 1]:
            signals[i] = -1  # 賣出訊號
    return signals

# 布林帶判斷
def bollinger_bands_strategy(close_prices, upper_band, lower_band, time):
    signals = [0] * len(time)  # 初始化信號列表匹配時間序列長度
    for i in range(len(time)):
        if close_prices[i] > upper_band[i]:
            signals[i] = -1  # 賣出訊號
        elif close_prices[i] < lower_band[i]:
            signals[i] = 1  # 買入訊號
    return signals

# OBV 判斷
def obv_strategy(obv_values, time):
    signals = [0] * len(time)  # 初始化信號列表匹配時間序列長度
    for i in range(1, len(time)):
        if obv_values[i] > obv_values[i - 1]:
            signals[i] = 1  # 買入訊號
        elif obv_values[i] < obv_values[i - 1]:
            signals[i] = -1  # 賣出訊號
    return signals

def query_stock_data(symbol, start_date, end_date):
    # 將日期格式轉換為 "YYYY-MM-DD"
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    
    stock = yf.Ticker(symbol)
    stock_data = stock.history(start=start_date, end=end_date)
    return stock_data


#### 計算技術指標 ####
def calculate_technical_indicators(stock_data):
    # 計算MA指標
    short_ma = talib.MA(stock_data['Close'], timeperiod=5)
    medium_ma = talib.MA(stock_data['Close'], timeperiod=20)
    long_ma = talib.MA(stock_data['Close'], timeperiod=50)

    # 計算KD指標
    high = stock_data['High'].values
    low = stock_data['Low'].values
    close = stock_data['Close'].values
    k, d = talib.STOCH(high, low, close)

    # 計算RSI指標
    rsi = talib.RSI(stock_data['Close'])

    # 計算MACD指標
    macd, signal, _ = talib.MACD(stock_data['Close'])

    # 計算布林通道指標
    upper_band, middle_band, lower_band = talib.BBANDS(stock_data['Close'])

    # 計算OBV指標
    obv = talib.OBV(stock_data['Close'], stock_data['Volume'])

    # 使用技術指標判斷策略生成交易信號
    ma_cross_signals = ma_cross_strategy(short_ma, medium_ma, stock_data.index)
    kd_cross_signals = kd_cross_strategy(k, d, stock_data.index)
    rsi_signals = rsi_strategy(rsi, stock_data.index)
    macd_signals = macd_strategy(macd, signal, stock_data.index)
    bollinger_bands_signals = bollinger_bands_strategy(stock_data['Close'], upper_band, lower_band, stock_data.index)
    obv_signals = obv_strategy(obv, stock_data.index)

    return {
        'short_ma': short_ma.tolist(),
        'medium_ma': medium_ma.tolist(),
        'long_ma': long_ma.tolist(),
        'k': k.tolist(),
        'd': d.tolist(),
        'rsi': rsi.tolist(),
        'macd': macd.tolist(),
        'signal': signal.tolist(),
        'upper_band': upper_band.tolist(),
        'middle_band': middle_band.tolist(),
        'lower_band': lower_band.tolist(),
        'obv': obv.tolist(),
        'ma_cross_signals': ma_cross_signals,
        'kd_cross_signals': kd_cross_signals,
        'rsi_signals': rsi_signals,
        'macd_signals': macd_signals,
        'bollinger_bands_signals': bollinger_bands_signals,
        'obv_signals': obv_signals,
    }

# 準備股票數據JSON
def prepare_stock_data_json(stock_data):
    stock_data = stock_data[stock_data['Volume'] > 0]
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = stock_data['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')  # 將日期格式轉換為 "YYYY-MM-DD"
    return stock_data.to_dict(orient='records')

# 計算交易成本
def calculate_transaction_cost(price, volume, is_sell=False):
    """
    計算交易成本。
    :param price: 成交價格
    :param volume: 成交量
    :param is_sell: 是否為賣出交易
    :return: 手續費和證交稅的總和
    """
    fee_rate = 0.001425  # 手續費率
    tax_rate = 0.003 if is_sell else 0  # 賣出時的證交稅率

    fee = price * volume * fee_rate
    tax = price * volume * tax_rate

    return fee + tax

# 計算累積回報
def calculate_cumulative_returns(signals, stock_data):
    cumulative_returns = [1.0]  # 初始值設為1.0
    current_balance = 1.0  # 初始投資資本設為1.0
    for i in range(1, len(signals)):
        if signals[i] == 1:  # 買入訊號
            cost = stock_data['Close'][i] * 0.001425
            current_balance *= (stock_data['Close'][i] - cost) / stock_data['Close'][i]
        elif signals[i] == -1:  # 賣出訊號
            cost = stock_data['Close'][i] * (0.001425 + 0.003)
            current_balance *= (stock_data['Close'][i] - cost) / stock_data['Close'][i]
        cumulative_returns.append(current_balance)
    return cumulative_returns

# 計算年化回報
def calculate_annualized_returns(cumulative_returns, stock_data):
    years = (stock_data.index[-1] - stock_data.index[0]).days / 365.0
    final_balance = cumulative_returns[-1]
    annualized_returns = (final_balance ** (1 / years)) - 1
    return annualized_returns

# 計算夏普比率
def calculate_sharpe_ratio(annualized_returns, annualized_std_dev):
    # 通常，無風險利率是根據市場標準或特定國家的短期國庫券利率來設定的。
    # 這裡，我們假設無風險利率為0，但這個值應根據當前經濟環境來調整。
    risk_free_rate = 0.0
    # 計算夏普比率
    sharpe_ratio = (annualized_returns - risk_free_rate) / annualized_std_dev
    return sharpe_ratio

# 計算最大回撤
def calculate_maximum_drawdown(cumulative_returns):
    # 初始化最大回撤和峰值
    max_drawdown = 0
    peak = cumulative_returns[0]

    for value in cumulative_returns:
        # 如果當前價值高於目前峰值，更新峰值
        if value > peak:
            peak = value

        # 計算從峰值到當前值的回撤
        drawdown = (peak - value) / peak

        # 如果這個回撤大於目前的最大回撤，更新最大回撤
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown

# 計算勝率
def calculate_win_rate(signals):
    # 計算成功交易的數量
    successful_trades = sum(1 for signal in signals if signal == 1)
    # 計算總交易數
    total_trades = len(signals)
    # 計算勝率
    if total_trades == 0:
        return 0  # 避免除以零的情況
    win_rate = successful_trades / total_trades
    return win_rate

# 計算年化標準差
def calculate_annualized_std_dev(signals, stock_data):
    # 計算日回報率
    daily_returns = []
    for i in range(1, len(stock_data)):
        if signals[i] == 1 or signals[i] == -1:  # 只考慮有交易信號的日子
            daily_return = (stock_data['Close'][i] - stock_data['Close'][i - 1]) / stock_data['Close'][i - 1]
            daily_returns.append(daily_return)

    # 計算日回報率的標準差
    std_dev = np.std(daily_returns)

    # 年化標準差
    annualized_std_dev = std_dev * (252 ** 0.5)

    return annualized_std_dev

# 計算盈利因子
def calculate_profit_factor(profitable_trades, losing_trades):
    # 計算所有盈利交易的總和
    total_profit = sum(profitable_trades)

    # 計算所有虧損交易的絕對總和
    total_loss = abs(sum(losing_trades))

    # 避免除以零的情況
    if total_loss == 0:
        return float('inf')  # 如果沒有虧損交易，返回無限大

    # 計算盈利因子
    profit_factor = total_profit / total_loss
    return profit_factor


#### 視圖 ####
def stock_list(request):
    form = StockForm(request.POST or None)
    stock_data_json = None
    indicators_data = None
    data_empty = True

    if request.method == 'POST' and form.is_valid():
        symbol = form.cleaned_data['symbol']
        start_date = form.cleaned_data['start_date']
        end_date = form.cleaned_data['end_date']

        stock_data = query_stock_data(symbol, start_date, end_date)

        # 在視圖中計算報酬指標
        if not stock_data.empty:
            indicators_data = calculate_technical_indicators(stock_data)
            stock_data_json = prepare_stock_data_json(stock_data)
            data_empty = False
            
            # 將日期部分保留為年月日（yyyymmdd）
            for i in range(len(stock_data_json)):
                stock_data_json[i]['Date'] = stock_data_json[i]['Date'][:10]

            # 計算報酬指標
            cumulative_returns = calculate_cumulative_returns(indicators_data['ma_cross_signals'], stock_data)
            annualized_returns = calculate_annualized_returns(cumulative_returns, stock_data)
            sharpe_ratio = calculate_sharpe_ratio(annualized_returns, calculate_annualized_std_dev(indicators_data['ma_cross_signals'], stock_data))
            max_drawdown = calculate_maximum_drawdown(cumulative_returns)
            win_rate = calculate_win_rate(indicators_data['ma_cross_signals'])
            profit_factor = calculate_profit_factor([0 if signal != 1 else 1 for signal in indicators_data['ma_cross_signals']], [0 if signal != -1 else 1 for signal in indicators_data['ma_cross_signals']])

            # 表現json
            for i in range(len(stock_data_json)):
                stock_data_json[i]['cumulative_returns'] = cumulative_returns[i]

            performance_metrics = {
                'annualized_returns': annualized_returns,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor
            }


        request.session['form_data'] = request.POST.dict()
        request.session['stock_data_json'] = stock_data_json
        request.session['indicators_data'] = indicators_data
        request.session['performance_metrics'] = performance_metrics  # 交報酬表現指標放入performance_metrics

    elif 'form_data' in request.session:
        form = StockForm(request.session['form_data'])
        stock_data_json = request.session.get('stock_data_json')
        indicators_data = request.session.get('indicators_data')
        data_empty = not stock_data_json

    context = {
        'form': form,
        'stock_data_json': json.dumps(stock_data_json) if stock_data_json else None,
        'indicators_data': json.dumps(indicators_data) if indicators_data else None,
        'data_empty': data_empty,
    }

    return render(request, 'stockinfo/stock_list.html', context)

# stock_table
def stock_table(request):
    # Get the stock data and indicators data from the session.
    stock_data_json = request.session.get('stock_data_json', [])
    indicators_data = request.session.get('indicators_data', {})

    # Check if there is any stock data.
    if stock_data_json:
        data_empty = False

        # Process each stock data entry to include the signals.
        for i, stock_data in enumerate(stock_data_json):
            # Update each entry with the corresponding signals.
            stock_data['ma_cross_signal'] = indicators_data.get('ma_cross_signals', ["N/A"])[i]
            stock_data['kd_cross_signal'] = indicators_data.get('kd_cross_signals', ["N/A"])[i]
            stock_data['rsi_signal'] = indicators_data.get('rsi_signals', ["N/A"])[i]
            stock_data['macd_signal'] = indicators_data.get('macd_signals', ["N/A"])[i]
            stock_data['bollinger_bands_signal'] = indicators_data.get('bollinger_bands_signals', ["N/A"])[i]
            stock_data['obv_signal'] = indicators_data.get('obv_signals', ["N/A"])[i]
    else:
        data_empty = True

    # Prepare the context for the template.
    context = {
        'stock_data_json': stock_data_json,  # This now includes the signals.
        'data_empty': data_empty,
    }

    return render(request, 'stockinfo/stock_table.html', context)

