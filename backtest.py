import numpy as np
import pandas as pd

def calculate_five_minute_vwap(mmep_data):
    """
    Calculate the 5-minute VWAP (Volume Weighted Average Price) from minute-level data.

    VWAP is calculated by weighting the price by the volume traded during each period.

    :param mmep_data: A pandas DataFrame in MMEP format, containing 'vwap' and 'volume' columns.
                      'vwap' refers to the minute-level VWAP, and 'volume' represents the traded volume for that minute.
    :return: A pandas Series containing the 5-minute VWAP values.
    """
    # Group data by date ('didx') and 5-minute intervals (grouping 'tidx' into chunks of 5 minutes)
    grouped = mmep_data.groupby(['didx', mmep_data.index.get_level_values('tidx') // 5])
    
    # Calculate the 5-minute VWAP by applying a weighted sum of vwap * volume, divided by the total volume
    vwap_5min = grouped.apply(lambda x: (x['vwap'] * x['volume']).sum() / x['volume'].sum())
    
    # Return the result as a pandas Series
    return vwap_5min

def calculate_transaction_costs(trade_value, position_type='buy'):
    """
    Calculate transaction costs in the Hong Kong market, including stamp duty, trading fees, SFC levy, and slippage.

    :param trade_value: The total trade value (notional value of the transaction).
    :param position_type: Either 'buy' or 'sell', to differentiate between different costs for buying or selling.
    :return: The total transaction cost.
    """
    # Stamp duty is 0.13%, applied to every transaction
    stamp_duty = 0.001 if position_type == 'buy' else 0.001
    
    # Other transaction fees
    trading_fee = 0.00005  # HKEX trading fee
    sfc_levy = 0.00002  # SFC levy
    afrc_levy = 0.000001  # AFRC levy
    
    # Slippage assumption of 0.05%
    slippage = 0.0002  # Assuming slippage is 0.02%
    
    # Total transaction cost
    total_cost = trade_value * (stamp_duty + trading_fee + sfc_levy + afrc_levy + slippage)
    
    return total_cost

def backtest_vwap_strategy(mmep_data, position, initial_capital=1e7):
    
    capital = initial_capital
    pnl_data = []  # Store pnl data (time, capital, return, turnover, long/short averages)
    
    # Calculate VWAP for every five minutes, handle NaN beforehand
    vwap_5min = calculate_five_minute_vwap(mmep_data).fillna(method='ffill')
    stocks = position.columns
    position = position.shift(1)
    previous_position = position.iloc[0]  # Track the previous position for PnL and costs
    previous_capital = capital

    for idx in range(1, len(position)):  # Iterate over every five-minute interval (start from the second interval)
        didx = position.index.get_level_values('didx')[idx]
        tidx = position.index.get_level_values('tidx')[idx]

        new_position = position.iloc[idx]
        current_vwap = vwap_5min.iloc[idx - 1]  # VWAP at the previous interval
        next_vwap = vwap_5min.iloc[idx]  # VWAP at the current interval

        if pd.isna(current_vwap).any() or pd.isna(next_vwap).any():
            continue  # Skip if there are NaN values in VWAP

        # Calculate return
        ret = next_vwap / current_vwap - 1
        ret = np.sum(previous_position * ret)  # Calculate returns using previous position and previous VWAP
        
        # Calculate new capital
        new_capital = previous_capital * (1 + ret)

        # Calculate transaction costs: the change in position from the previous one
        position_change = abs(new_position - previous_position)  # Change in position (absolute value)
       
        # Total transaction costs
        total_cost = np.sum(calculate_transaction_costs(position_change * previous_capital))  # Total transaction cost
        new_capital -= total_cost  # Subtract transaction cost from new capital

        # Store turnover: position change * previous capital
        turnover = np.sum(position_change)
        
        # Store the average of long and short positions for this interval
        long_position = round(new_position[new_position > 0].mean() if len(new_position[new_position > 0]) > 0 else 0, 2)
        short_position = round(new_position[new_position < 0].mean() if len(new_position[new_position < 0]) > 0 else 0, 2)

        # Store time, capital, return, turnover, long position, and short position in pnl_data
        pnl_data.append({
            'Date': f"{didx}-{tidx:02d}",
            'Capital': new_capital,
            'Return': ret,
            'Turnover': turnover,
            'Long': long_position,
            'Short': short_position
        })

        # Update the previous position and capital for the next loop
        previous_position = new_position

    # Create a DataFrame for pnl data
    pnl_df = pd.DataFrame(pnl_data)

    # Save to file
    pnl_df.to_csv('pnl_file.csv', index=False)

    return pnl_df


