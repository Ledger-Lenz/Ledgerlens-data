import numpy as np
import pandas as pd

def drop_trades(df_trades, drop_prob=0.2):
    """Randomly drop a percentage of trades."""
    if len(df_trades) == 0:
        return df_trades
    mask = np.random.rand(len(df_trades)) > drop_prob
    # Ensure at least one trade remains if original had trades
    if not mask.any():
        mask[np.random.randint(0, len(df_trades))] = True
    return df_trades[mask]

def jitter_timestamps(df_trades, jitter_pct=0.05):
    """Jitter trade timestamps by +/- jitter_pct."""
    if len(df_trades) == 0 or 'timestamp' not in df_trades.columns:
        return df_trades
    df_aug = df_trades.copy()
    time_range = df_aug['timestamp'].max() - df_aug['timestamp'].min()
    jitter_amount = time_range * jitter_pct
    if jitter_amount > 0:
        noise = np.random.uniform(-jitter_amount, jitter_amount, size=len(df_aug))
        df_aug['timestamp'] = df_aug['timestamp'] + noise
        df_aug['timestamp'] = df_aug['timestamp'].sort_values().values
    return df_aug

def mask_amounts(df_trades, mask_prob=0.15):
    """Randomly mask trade amounts (set to 0 or mean)."""
    if len(df_trades) == 0 or 'amount' not in df_trades.columns:
        return df_trades
    df_aug = df_trades.copy()
    mask = np.random.rand(len(df_aug)) < mask_prob
    if mask.any():
        mean_amount = df_aug['amount'].mean()
        df_aug.loc[mask, 'amount'] = mean_amount
    return df_aug

def augment_trade_sequence(df_trades):
    """Apply a series of augmentations to generate a view for contrastive learning."""
    df_aug = df_trades.copy()
    
    # Positive Pair Construction:
    # View 1 and View 2 will use this.
    # The prompt specified: View 1: subsample 80% of trades, View 2: different 80% subsample + jitter timestamps +/- 5%
    # This function is a helper that can be parameterized.
    
    # 1. Drop trades (subsample 80% -> drop_prob=0.2)
    df_aug = drop_trades(df_aug, drop_prob=0.2)
    
    # 2. Jitter timestamps (+/- 5%)
    df_aug = jitter_timestamps(df_aug, jitter_pct=0.05)
    
    # 3. Mask amounts
    df_aug = mask_amounts(df_aug, mask_prob=0.15)
    
    return df_aug
