import numpy as np
import pandas as pd


def calculate_five_minute_alpha_factors(mmep_data):
    """
    Calculate Alpha factors at 5-minute intervals for all stocks in the data.
    
    :param mmep_data: MMEP-format DataFrame with multi-index (didx, tidx)
    :return: A dictionary of DataFrames containing different Alpha factors, indexed by (didx, tidx // 5).
    """
    alpha_results = {}

    # Extract relevant data for all stocks (assuming multilevel columns in the DataFrame)
    lift_volume = mmep_data.xs('lift_volume', level=0, axis=1)
    hit_volume = mmep_data.xs('hit_volume', level=0, axis=1)
    lift_vwap = mmep_data.xs('lift_vwap', level=0, axis=1)
    hit_vwap = mmep_data.xs('hit_vwap', level=0, axis=1)
    num_trade = mmep_data.xs('num_trade', level=0, axis=1)

    # Group by 5-minute intervals (assuming tidx is the minute index)
    grouped = mmep_data.groupby([mmep_data.index.get_level_values('didx'),
                                 mmep_data.index.get_level_values('tidx') // 5])

    # Alpha1: Number of times active buy volume > active sell volume within 5 minutes
    alpha1 = (lift_volume - hit_volume).groupby(
        [mmep_data.index.get_level_values('didx'), mmep_data.index.get_level_values('tidx') // 5]
    ).apply(lambda x: (x > 0).sum())
    alpha1 = alpha1.reindex(grouped.size().index, fill_value=0)

    # Alpha2: Bid-ask VWAP imbalance in 5-minute intervals
    alpha2 = -((mmep_data.xs('ask_twap', level=0, axis=1) - mmep_data.xs('bid_twap', level=0, axis=1)) / mmep_data.xs('vwap', level=0, axis=1)).groupby(
        [mmep_data.index.get_level_values('didx'), mmep_data.index.get_level_values('tidx') // 5]
    ).apply(lambda x: (x > 0).sum())
    alpha2 = alpha2.reindex(grouped.size().index, fill_value=0)

    # Alpha3: Number of times capital inflow > capital outflow within 5 minutes 
    capital_inflow = (lift_volume * lift_vwap).groupby(
        [mmep_data.index.get_level_values('didx'), mmep_data.index.get_level_values('tidx') // 5]
    ).sum()
    capital_outflow = (hit_volume * hit_vwap).groupby(
        [mmep_data.index.get_level_values('didx'), mmep_data.index.get_level_values('tidx') // 5]
    ).sum()
    alpha3 = (capital_inflow > capital_outflow).astype(int)
    alpha3 = alpha3.reindex(grouped.size().index, fill_value=0)

    # Alpha4: Number of times active buy volume > 5-minute rolling average within 5 minutes  
    avg_buy_volume_5min = lift_volume.rolling(window=5).mean()
    alpha4 = (lift_volume > avg_buy_volume_5min).groupby(
        [mmep_data.index.get_level_values('didx'), mmep_data.index.get_level_values('tidx') // 5]
    ).sum()
    alpha4 = alpha4.reindex(grouped.size().index, fill_value=0)

    # Alpha5: Number of times active sell volume <  5-minute rolling average within 5 minutes
    avg_sell_volume_5min = hit_volume.rolling(window=5).mean()
    alpha5 = (hit_volume < avg_sell_volume_5min).groupby(
        [mmep_data.index.get_level_values('didx'), mmep_data.index.get_level_values('tidx') // 5]
    ).sum()
    alpha5 = alpha5.reindex(grouped.size().index, fill_value=0)

    #   
    avg_num_trade_5min = num_trade.rolling(window=5).mean()
    alpha6 = -(num_trade > avg_num_trade_5min).groupby(
        [mmep_data.index.get_level_values('didx'), mmep_data.index.get_level_values('tidx') // 5]
    ).sum()
    alpha6 = alpha6.reindex(grouped.size().index, fill_value=0)

    # Alpha7: Price momentum within 5-minute intervals
    alpha7 = -mmep_data.xs('close', level=0, axis=1).pct_change().groupby(
        [mmep_data.index.get_level_values('didx'), mmep_data.index.get_level_values('tidx') // 5]
    ).apply(lambda x: (x > 0).sum())
    alpha7 = alpha7.reindex(grouped.size().index, fill_value=0)

    # Alpha8: Volume imbalance in 5-minute intervals
    alpha8 = ((lift_volume - hit_volume) / (lift_volume + hit_volume)).groupby(
        [mmep_data.index.get_level_values('didx'), mmep_data.index.get_level_values('tidx') // 5]
    ).apply(lambda x: (x > 0).sum())
    alpha8 = alpha8.reindex(grouped.size().index, fill_value=0)

    # Alpha9: Price deviation from VWAP in 5-minute intervals
    alpha9 = -((mmep_data.xs('close', level=0, axis=1) - mmep_data.xs('vwap', level=0, axis=1)) / mmep_data.xs('vwap', level=0, axis=1)).groupby(
        [mmep_data.index.get_level_values('didx'), mmep_data.index.get_level_values('tidx') // 5]
    ).apply(lambda x: (x > 0).sum())
    alpha9 = alpha9.reindex(grouped.size().index, fill_value=0)

    # Alpha10: Bid-ask spread relative to VWAP
    alpha10 = -((mmep_data.xs('last_ask', level=0, axis=1) - mmep_data.xs('last_bid', level=0, axis=1)) / mmep_data.xs('vwap', level=0, axis=1)).groupby(
        [mmep_data.index.get_level_values('didx'), mmep_data.index.get_level_values('tidx') // 5]
    ).apply(lambda x: (x > 0).sum())
    alpha10 = alpha10.reindex(grouped.size().index, fill_value=0)

    # Alpha11: Spike in trade count
    alpha11 = -(num_trade - num_trade.rolling(window=5).mean()).groupby(
        [mmep_data.index.get_level_values('didx'), mmep_data.index.get_level_values('tidx') // 5]
    ).apply(lambda x: (x > 0).sum())
    alpha11 = alpha11.reindex(grouped.size().index, fill_value=0)

    # Store all alpha results in a dictionary
    alpha_results['alpha1'] = alpha1
    alpha_results['alpha2'] = alpha2
    alpha_results['alpha3'] = alpha3
    alpha_results['alpha4'] = alpha4
    alpha_results['alpha5'] = alpha5
    alpha_results['alpha6'] = alpha6
    alpha_results['alpha7'] = alpha7
    alpha_results['alpha8'] = alpha8
    alpha_results['alpha9'] = alpha9
    alpha_results['alpha10'] = alpha10
    alpha_results['alpha11'] = alpha11

    return alpha_results


def opPower(alpha_series):
    """
    Apply the opPower transformation on the alpha factor series:
    
    1. Rank the alpha values, transforming them into a range of [-0.5, 0.5].
    2. Apply an exponential transformation to the absolute values while keeping the original signs.
    3. Scale the positive and negative portions separately such that:
       - The positive part sums to +0.5.
       - The negative part sums to -0.5.
    
    :param alpha_series: A pandas Series representing the alpha factor for a cross-section (e.g., across multiple stocks).
    :return: A transformed pandas Series where the positive and negative parts sum to +0.5 and -0.5, respectively.
    """
    
    # Step 1: Rank alpha values and transform to the range [-0.5, 0.5]
    ranked_alpha = alpha_series.rank(pct=True) - 0.5
    
    # Step 2: Apply the exponential transformation to the absolute values while preserving the signs
    transformed_alpha = ranked_alpha.apply(lambda x: np.sign(x) * np.exp(np.abs(x)))
    
    # Step 3: Scale the positive and negative parts separately
    positive_sum = transformed_alpha[transformed_alpha > 0].sum()
    negative_sum = transformed_alpha[transformed_alpha < 0].sum()
    
    if positive_sum != 0:
        # Scale positive values to ensure their sum is 0.5
        transformed_alpha[transformed_alpha > 0] *= 0.5 / positive_sum
    if negative_sum != 0:
        # Scale negative values to ensure their sum is -0.5
        transformed_alpha[transformed_alpha < 0] *= 0.5 / abs(negative_sum)
    
    return transformed_alpha

def calculate_and_transform_position(mmep_data):
    """
    Calculate and transform the target positions using Alpha factors.
    
    This function first calculates several Alpha factors at 5-minute intervals from the input MMEP data.
    Each Alpha factor is then transformed using the opPower function, which balances the positive and
    negative positions. Finally, these transformed Alpha factors are combined to generate the target 
    positions, which are also passed through the opPower function to ensure they are balanced.
    
    :param mmep_data: A pandas DataFrame in MMEP format, containing the necessary input data for all stocks.
    :return: A pandas DataFrame representing the final transformed target positions for each stock at each time step.
    """
    # Step 1: Calculate Alpha factors based on the MMEP data
    alphas_dict = calculate_five_minute_alpha_factors(mmep_data)
    
    # Dictionary to hold the transformed Alpha factors
    alphas_transformed = {}

    # Step 2: Process each Alpha factor DataFrame
    for name, alpha_df in alphas_dict.items():
        # Create a DataFrame to hold the transformed alphas (same structure as original alpha_df)
        transformed_alpha = pd.DataFrame(index=alpha_df.index, columns=alpha_df.columns)

        # Apply opPower function to each cross-section
        for time_idx in alpha_df.index:
            cross_section = alpha_df.loc[time_idx]  # Extract cross-section for this time
            transformed_alpha.loc[time_idx] = opPower(cross_section)  # Apply opPower transformation

        # Store the transformed alpha in the dictionary
        alphas_transformed[name] = transformed_alpha

    # Step 3: Combine all the processed Alpha factors into a single signal DataFrame
    # Here, summing the Alpha factors creates the combined signal
    combined_signal = sum(alphas_transformed.values())

    # Step 4: Apply the opPower transformation to the combined signal for the final position assignment
    position = combined_signal.apply(opPower, axis=1)  # Transform across each time point

    # Return the final positions
    return position
