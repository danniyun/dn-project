import numpy as np
import pandas as pd

def calculate_daily_pnl_metrics(pnl_file):
    pnl_df = pnl_file

    # Extract the 'MMDD' portion of the date for daily grouping
    pnl_df['Day'] = pnl_df['Date'].str[:4]

    # Initialize lists for results
    results = []

    # Group data by day (assuming MMDD format)
    daily_groups = pnl_df.groupby('Day')

    for day, group in daily_groups:
        # Total PnL (sum of returns)
        tpnl = np.sum(group['Return'])

        # Long and short exposure as sums for the day
        long = np.sum(group['Long'])
        short = np.sum(group['Short'])

        # Average return for the day (percentage)
        ret = np.mean(group['Return']) * 100

        # Total turnover for the day
        turnover = np.sum(group['Turnover'])

        # Maximum Drawdown calculation
        capital = group['Capital'].values
        high_water_mark = capital[0]  # Start with the first capital as the peak
        max_drawdown = 0  # Initialize max drawdown

        for current_cap in capital:
            if current_cap > high_water_mark:
                high_water_mark = current_cap  # Update peak capital
            current_drawdown = (high_water_mark - current_cap) / high_water_mark * 100  # Peak-to-trough drop in percentage
            max_drawdown = max(max_drawdown, current_drawdown)  # Maximum drawdown observed

        # Add result for this day
        results.append({
            'from': group['Date'].iloc[0][:4],
            'to': group['Date'].iloc[-1][:4],
            'long': 10,
            'short': -10,
            'return': ret,
            'turnover': turnover,
            'max_drawdown': max_drawdown  # Maximum drawdown in percentage
        })

    # Convert results to DataFrame
    pnl_summary_df = pd.DataFrame(results)

    return pnl_summary_df


def calculate_monthly_pnl_metrics(daily_pnl_summary):
    # Extract the month part from the 'from' or 'to' column (assuming format MMDD)
    daily_pnl_summary['Month'] = daily_pnl_summary['from'].str[:2]  # Take only the first two characters for month

    # Group by Month and calculate the maximum drawdown per month
    monthly_drawdown = daily_pnl_summary.groupby('Month')['max_drawdown'].max()

    # Initialize the results list to store the summary for each month
    results = []

    # Loop over each month group
    for month, group in daily_pnl_summary.groupby('Month'):
        # Calculate other metrics (sum or average) for the month as needed
        long = group['long'].sum()
        short = group['short'].sum()
        ret = group['return'].sum()
        turnover = group['turnover'].mean()
        
        # Calculate the Sharpe ratio: (mean return) / (std return) * sqrt(20) assuming 20 days/month
        sharpe_ratio = (group['return'].mean() / group['return'].std()) * np.sqrt(20) if group['return'].std() != 0 else 0
        
        # Get the maximum drawdown for the month (calculated earlier)
        max_drawdown = monthly_drawdown[month]
        
        # Append the monthly result to the list
        results.append({
            'from': group['from'].iloc[0][:4],
            'to': group['to'].iloc[-1][:4],
            'long': 10,
            'short': -10,
            'return': ret,
            'sharpe': sharpe_ratio,
            'turnover': turnover,
            'max_drawdown': max_drawdown
        })

    # Convert the results list into a DataFrame
    monthly_summary_df = pd.DataFrame(results)

    return monthly_summary_df



