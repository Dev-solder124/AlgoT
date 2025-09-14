import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def calc_longest_drawdown(pnl_series):
    """Calculate longest consecutive drawdown period in points"""
    if not isinstance(pnl_series, pd.Series):
        pnl_series = pd.Series(pnl_series)
        
    underwater = (pnl_series - pnl_series.cummax()) < 0
    runs = (~underwater).cumsum()[underwater]
    
    if runs.empty:
        return 0, None, None
    
    counts = runs.value_counts(sort=True)
    max_dur = counts.iloc[0]
    inds = runs == counts.index[0]
    inds = inds.where(inds)
    return max_dur, inds.first_valid_index(), inds.last_valid_index()

def calculate_drawdown_metrics(pnl_data, timestamps=None):
    """Comprehensive drawdown analysis in absolute points"""
    # Convert to pandas Series (handles both array and Series input)
    if not isinstance(pnl_data, pd.Series):
        if timestamps is not None:
            pnl_series = pd.Series(pnl_data, index=pd.to_datetime(timestamps))
        else:
            pnl_series = pd.Series(pnl_data)
    else:
        pnl_series = pnl_data.copy()
    
    # Ensure chronological order
    pnl_series = pnl_series.sort_index()
    
    cummax = pnl_series.cummax()
    drawdown_points = pnl_series - cummax  # Points lost from peak
    
    # Maximum drawdown
    max_dd_points = drawdown_points.min()
    max_dd_end = drawdown_points.idxmin()
    max_dd_start = cummax.loc[:max_dd_end].idxmax()
    max_dd_duration = (max_dd_end - max_dd_start).days
    
    # Longest drawdown
    longest_dur, longest_start, longest_end = calc_longest_drawdown(pnl_series)
    if longest_start and longest_end:
        longest_duration = (longest_end - longest_start).days
    else:
        longest_duration = 0
    
    return {
        'max_drawdown_points': max_dd_points,
        'max_drawdown_start': max_dd_start,
        'max_drawdown_end': max_dd_end,
        'max_drawdown_duration': max_dd_duration,
        'longest_drawdown_duration': longest_duration,
        'longest_drawdown_start': longest_start,
        'longest_drawdown_end': longest_end,
        'drawdown_series': drawdown_points,
        'pnl_series': pnl_series  # Return the validated series
    }

def plot_pnl_with_drawdowns(pnl_series, results):
    """Professional two-panel visualization with absolute points"""
    plt.figure(figsize=(10, 6))
    
    # Create gridspec for better layout control
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.4)
    
    # --- Top Plot: P&L Curve ---
    ax1 = plt.subplot(gs[0])
    ax1.plot(pnl_series.index, pnl_series, 
             label='P&L', color='#1f77b4', linewidth=1.5)
    
    # Highlight max drawdown period
    ax1.axvspan(results['max_drawdown_start'], results['max_drawdown_end'],
               color='red', alpha=0.15,
               label=f"Max Drawdown: {results['max_drawdown_points']:.1f} points")
    
    # Formatting
    ax1.set_title('P&L Curve with Maximum Drawdown', pad=20, fontsize=14)
    ax1.set_ylabel('Portfolio Value (Points)', fontsize=12)
    ax1.legend(loc='upper left', framealpha=1)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # --- Bottom Plot: Drawdown in Points ---
    ax2 = plt.subplot(gs[1])
    
    # Area plot for drawdowns
    ax2.fill_between(results['drawdown_series'].index,
                    results['drawdown_series'],
                    color='red', alpha=0.25, label='Drawdown')
    
    # Line plot for better visibility
    ax2.plot(results['drawdown_series'].index,
            results['drawdown_series'],
            color='red', linewidth=1, alpha=0.8)
    
    # Highlight longest drawdown
    if results['longest_drawdown_start']:
        ax2.axvspan(results['longest_drawdown_start'],
                   results['longest_drawdown_end'],
                   color='blue', alpha=0.15,
                   label=f"Longest Drawdown: {results['longest_drawdown_duration']} days")
    
    # Formatting
    ax2.set_title('Drawdown in Absolute Points', pad=20, fontsize=14)
    ax2.set_ylabel('Points Below Peak', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.yaxis.set_major_locator(MaxNLocator(6))  # Clean y-axis ticks
    ax2.legend(loc='lower left', framealpha=1)
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    # Shared date formatting
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(plt.MaxNLocator(8))  # Reasonable number of ticks
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# Example usage with your data:
if __name__ == "__main__":
    # Load your data (replace with actual paths)
    timestamps = np.load('date.npy', allow_pickle=True)
    rev3 = np.load('net.npy')
    rev2 = np.load('rev2.npy')
    inter = np.load('interr.npy')
    pnl_values=(rev3)
    # Calculate metrics
    results = calculate_drawdown_metrics(pnl_values, timestamps)
    
    # Print results
    print(f"Maximum Drawdown: {results['max_drawdown_points']:.2f} points")
    print(f"Period: {results['max_drawdown_start']} to {results['max_drawdown_end']}")
    print(f"Duration: {results['max_drawdown_duration']} days\n")
    
    print(f"Longest Drawdown Period:")
    print(f"Duration: {results['longest_drawdown_duration']} days")
    print(f"Period: {results['longest_drawdown_start']} to {results['longest_drawdown_end']}")
    
    # Generate the professional plot
    plot_pnl_with_drawdowns(results['pnl_series'], results)
