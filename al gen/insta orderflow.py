import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def download_tlt_data():
    """Download TLT historical data"""
    print("📥 Downloading TLT data from Yahoo Finance...")
    ticker = yf.Ticker("^NSEI")
    data = ticker.history(start="2020-01-01", end="2024-12-01")
    print(f"✅ Downloaded {len(data)} trading days of data")
    return data['Close']

def generate_flow_signals(prices):
    """Generate monthly flow effect signals"""
    print("🎯 Generating flow effect signals...")
    
    # Initialize signal arrays
    dates = prices.index
    short_entries = pd.Series(False, index=dates)
    short_exits = pd.Series(False, index=dates) 
    long_entries = pd.Series(False, index=dates)
    long_exits = pd.Series(False, index=dates)
    
    # Group by month and generate signals
    monthly_groups = prices.groupby(prices.index.to_period('M'))
    
    for period, month_data in monthly_groups:
        month_dates = month_data.index.tolist()
        
        if len(month_dates) < 8:  # Skip short months
            continue
            
        # Short signals: Enter on 1st and 5th day of month
        short_entries.loc[month_dates[0]] = True  # 1st day
        if len(month_dates) >= 5:
            short_entries.loc[month_dates[4]] = True  # 5th day
            
        # Short exits: 5 days after entry (around middle of month)
        if len(month_dates) >= 10:
            short_exits.loc[month_dates[9]] = True
        
        # Long entries: 7 days before month end
        if len(month_dates) >= 8:
            long_entries.loc[month_dates[-8]] = True
            
        # Long exits: 1 day before month end
        if len(month_dates) >= 2:
            long_exits.loc[month_dates[-2]] = True
    
    print(f"📊 Generated signals:")
    print(f"   Short entries: {short_entries.sum()}")
    print(f"   Short exits: {short_exits.sum()}")
    print(f"   Long entries: {long_entries.sum()}")
    print(f"   Long exits: {long_exits.sum()}")
    
    return short_entries, short_exits, long_entries, long_exits

def backtest_strategy(prices, short_entries, short_exits, long_entries, long_exits):
    """Backtest the flow effects strategy"""
    print("🔄 Running backtest simulation...")
    
    # Initialize portfolio tracking
    portfolio_value = pd.Series(100.0, index=prices.index)  # Start with $100
    position = 0  # 0 = no position, 1 = long, -1 = short
    entry_price = 0
    trades = []
    
    for i, date in enumerate(prices.index):
        current_price = prices.iloc[i]
        
        # Handle position entries
        if long_entries.iloc[i] and position == 0:
            position = 1
            entry_price = current_price
            
        elif short_entries.iloc[i] and position == 0:
            position = -1  
            entry_price = current_price
            
        # Handle position exits
        elif long_exits.iloc[i] and position == 1:
            pnl_pct = (current_price - entry_price) / entry_price
            portfolio_value.iloc[i:] = portfolio_value.iloc[i-1] * (1 + pnl_pct)
            trades.append({'date': date, 'type': 'long', 'pnl_pct': pnl_pct})
            position = 0
            
        elif short_exits.iloc[i] and position == -1:
            pnl_pct = (entry_price - current_price) / entry_price  # Reverse for short
            portfolio_value.iloc[i:] = portfolio_value.iloc[i-1] * (1 + pnl_pct)
            trades.append({'date': date, 'type': 'short', 'pnl_pct': pnl_pct})
            position = 0
            
        # Carry forward portfolio value if no trade
        if i > 0 and portfolio_value.iloc[i] == 100.0:
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1]
    
    # Calculate benchmark (buy and hold)
    benchmark = (prices / prices.iloc[0]) * 100
    
    print(f"✅ Backtest complete: {len(trades)} trades executed")
    return portfolio_value, benchmark, trades

def calculate_metrics(portfolio_value, benchmark, trades):
    """Calculate performance metrics"""
    print("📈 Calculating performance metrics...")
    
    # Returns
    strategy_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
    benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
    
    # Win rate
    winning_trades = [t for t in trades if t['pnl_pct'] > 0]
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    
    # Drawdown
    running_max = portfolio_value.expanding().max()
    drawdown = (portfolio_value - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # Monthly returns
    daily_returns = portfolio_value.pct_change().dropna()
    monthly_returns = daily_returns.groupby(daily_returns.index.to_period('M')).apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Sharpe ratio (annualized)
    annual_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (252/len(portfolio_value)) - 1
    annual_vol = daily_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    metrics = {
        'strategy_return': strategy_return,
        'benchmark_return': benchmark_return,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'monthly_returns': monthly_returns
    }
    
    return metrics, drawdown

def create_plots(prices, portfolio_value, benchmark, metrics, drawdown):
    """Create comprehensive performance plots"""
    print("📊 Creating performance visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create main dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TLT Flow Effects Strategy - Performance Analysis', fontsize=18, fontweight='bold')
    
    # Plot 1: Portfolio Value vs Benchmark
    axes[0,0].plot(portfolio_value.index, portfolio_value, label='Flow Strategy', 
                   color='#1f77b4', linewidth=2.5)
    axes[0,0].plot(benchmark.index, benchmark, label='Buy & Hold TLT', 
                   color='#ff7f0e', linewidth=2)
    axes[0,0].set_title('Portfolio Value Comparison', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Portfolio Value ($)', fontsize=12)
    axes[0,0].legend(fontsize=11)
    axes[0,0].grid(True, alpha=0.3)
    
    # Add performance text
    perf_text = f"Strategy: {metrics['strategy_return']:.1f}%\nBuy & Hold: {metrics['benchmark_return']:.1f}%"
    axes[0,0].text(0.05, 0.95, perf_text, transform=axes[0,0].transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 2: Monthly Returns Distribution
    axes[0,1].hist(metrics['monthly_returns'] * 100, bins=25, alpha=0.7, 
                   color='skyblue', edgecolor='black')
    axes[0,1].axvline(metrics['monthly_returns'].mean() * 100, color='red', 
                      linestyle='--', linewidth=2)
    axes[0,1].set_title('Monthly Returns Distribution', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Monthly Return (%)', fontsize=12)
    axes[0,1].set_ylabel('Frequency', fontsize=12)
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Drawdown Analysis
    axes[1,0].fill_between(drawdown.index, drawdown, 0, alpha=0.6, color='red')
    axes[1,0].plot(drawdown.index, drawdown, color='darkred', linewidth=1)
    axes[1,0].set_title('Strategy Drawdown', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Drawdown (%)', fontsize=12)
    axes[1,0].grid(True, alpha=0.3)
    
    # Add max drawdown text
    dd_text = f"Max Drawdown: {metrics['max_drawdown']:.1f}%"
    axes[1,0].text(0.05, 0.05, dd_text, transform=axes[1,0].transAxes,
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8))
    
    # Plot 4: Rolling Sharpe Ratio
    rolling_returns = portfolio_value.pct_change().rolling(window=252)
    rolling_sharpe = rolling_returns.mean() / rolling_returns.std() * np.sqrt(252)
    rolling_sharpe.plot(ax=axes[1,1], color='green', linewidth=2)
    axes[1,1].set_title('Rolling 1-Year Sharpe Ratio', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Sharpe Ratio', fontsize=12)
    axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tlt_flow_strategy_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Dashboard saved as 'tlt_flow_strategy_dashboard.png'")
    plt.show()
    
    # Create simple performance chart
    plt.figure(figsize=(12, 7))
    plt.plot(portfolio_value.index, portfolio_value, label='Flow Effects Strategy', 
             color='#1f77b4', linewidth=3)
    plt.plot(benchmark.index, benchmark, label='Buy & Hold TLT', 
             color='#ff7f0e', linewidth=2)
    
    plt.title('TLT Flow Effects Strategy Performance', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add final values as text
    final_text = f'Final Values:\nStrategy: ${portfolio_value.iloc[-1]:.0f} ({metrics["strategy_return"]:.1f}%)\nBuy & Hold: ${benchmark.iloc[-1]:.0f} ({metrics["benchmark_return"]:.1f}%)'
    plt.text(0.02, 0.98, final_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('tlt_flow_performance.png', dpi=300, bbox_inches='tight')
    print("✅ Performance chart saved as 'tlt_flow_performance.png'")
    plt.show()

def print_strategy_stats(metrics):
    """Print detailed strategy statistics"""
    print("\n" + "="*60)
    print("🎯 TLT FLOW EFFECTS STRATEGY RESULTS")
    print("="*60)
    print(f"💰 Total Return (Strategy): {metrics['strategy_return']:.2f}%")
    print(f"📊 Total Return (Buy & Hold): {metrics['benchmark_return']:.2f}%")
    print(f"🎯 Outperformance: {metrics['strategy_return'] - metrics['benchmark_return']:.2f}%")
    print(f"📈 Total Trades: {metrics['total_trades']}")
    print(f"🎪 Win Rate: {metrics['win_rate']:.1f}%")
    print(f"📉 Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"⚡ Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"📅 Average Monthly Return: {metrics['monthly_returns'].mean()*100:.2f}%")
    print(f"📊 Monthly Return Volatility: {metrics['monthly_returns'].std()*100:.2f}%")
    print("="*60)

def main():
    """Main execution function"""
    print("🚀 Starting TLT Flow Effects Strategy Analysis")
    print("="*50)
    
    # Download data
    prices = download_tlt_data()
    
    # Generate signals
    short_entries, short_exits, long_entries, long_exits = generate_flow_signals(prices)
    
    # Run backtest
    portfolio_value, benchmark, trades = backtest_strategy(
        prices, short_entries, short_exits, long_entries, long_exits
    )
    
    # Calculate metrics
    metrics, drawdown = calculate_metrics(portfolio_value, benchmark, trades)
    
    # Print results
    print_strategy_stats(metrics)
    
    # Create plots
    create_plots(prices, portfolio_value, benchmark, metrics, drawdown)
    
    print("\n🎉 Analysis complete! Check the generated PNG files for visualizations.")
    
    return {
        'prices': prices,
        'portfolio': portfolio_value,
        'benchmark': benchmark,
        'metrics': metrics,
        'trades': trades
    }

if __name__ == "__main__":
    results = main()