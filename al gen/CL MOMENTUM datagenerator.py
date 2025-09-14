import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import parse
from sklearn.preprocessing import StandardScaler, LabelEncoder
import ta
import warnings
warnings.filterwarnings('ignore')

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators for ML features"""
    
    # Price-based features
    df['price_change'] = df['close'].pct_change()
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    df['price_change_20'] = df['close'].pct_change(20)
    
    # Volatility features
    df['volatility_5'] = df['close'].rolling(5).std()
    df['volatility_10'] = df['close'].rolling(10).std()
    df['volatility_20'] = df['close'].rolling(20).std()
    df['high_low_ratio'] = df['high'] / df['low']
    df['price_range'] = df['high'] - df['low']
    df['price_range_norm'] = df['price_range'] / df['close']
    
    # Moving averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    df['ema_5'] = df['close'].ewm(span=5).mean()
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    
    # MA ratios and positions
    df['price_vs_ema20'] = df['close'] / df['ema_20']
    df['price_vs_ema50'] = df['close'] / df['ema_50']
    df['ema20_vs_ema50'] = df['ema_20'] / df['ema_50']
    df['price_vs_sma20'] = df['close'] / df['sma_20']
    
    # Momentum indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['rsi_5'] = ta.momentum.RSIIndicator(df['close'], window=5).rsi()
    df['rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume indicators (using price range as proxy since volume=0)
    df['volume_proxy'] = df['price_range']
    df['volume_sma'] = df['volume_proxy'].rolling(20).mean()
    df['volume_ratio'] = df['volume_proxy'] / df['volume_sma']
    
    # Support/Resistance levels
    df['resistance_20'] = df['high'].rolling(20).max()
    df['support_20'] = df['low'].rolling(20).min()
    df['price_vs_resistance'] = df['close'] / df['resistance_20']
    df['price_vs_support'] = df['close'] / df['support_20']
    
    # Trend indicators
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    
    # Williams %R
    df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
    
    return df

def calculate_time_features(df, datetime_col):
    """Calculate time-based features"""
    
    df['hour'] = datetime_col.dt.hour
    df['minute'] = datetime_col.dt.minute
    df['day_of_week'] = datetime_col.dt.dayofweek
    df['month'] = datetime_col.dt.month
    
    # Market session features
    df['market_open_minutes'] = (df['hour'] - 9) * 60 + df['minute'] - 15
    df['market_open_minutes'] = np.where(df['market_open_minutes'] < 0, 0, df['market_open_minutes'])
    
    # Time-based cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    
    # Market session indicators
    df['is_opening'] = ((df['hour'] == 9) & (df['minute'] <= 30)).astype(int)
    df['is_closing'] = ((df['hour'] == 15) & (df['minute'] >= 0)).astype(int)
    df['is_lunch_time'] = ((df['hour'] == 12) | ((df['hour'] == 13) & (df['minute'] <= 30))).astype(int)
    
    return df

def calculate_pattern_features(df):
    """Calculate candlestick and pattern features"""
    
    # Basic candle features
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
    df['total_range'] = df['high'] - df['low']
    
    # Normalized features
    df['body_ratio'] = df['body_size'] / df['total_range']
    df['upper_shadow_ratio'] = df['upper_shadow'] / df['total_range']
    df['lower_shadow_ratio'] = df['lower_shadow'] / df['total_range']
    
    # Candle type
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    df['is_doji'] = (df['body_ratio'] < 0.1).astype(int)
    df['is_hammer'] = ((df['lower_shadow_ratio'] > 0.6) & (df['upper_shadow_ratio'] < 0.1)).astype(int)
    df['is_shooting_star'] = ((df['upper_shadow_ratio'] > 0.6) & (df['lower_shadow_ratio'] < 0.1)).astype(int)
    
    # Multi-candle patterns
    df['consecutive_bullish'] = df['is_bullish'].rolling(3).sum()
    df['consecutive_bearish'] = (1 - df['is_bullish']).rolling(3).sum()
    
    return df

def calculate_breakout_features(df, window=30):
    """Calculate breakout and momentum features"""
    
    # Rolling highs and lows
    df[f'high_{window}'] = df['high'].rolling(window).max()
    df[f'low_{window}'] = df['low'].rolling(window).min()
    df[f'high_{window//2}'] = df['high'].rolling(window//2).max()
    df[f'low_{window//2}'] = df['low'].rolling(window//2).min()
    
    # Breakout indicators
    df[f'breakout_high_{window}'] = (df['close'] > df[f'high_{window}'].shift(1)).astype(int)
    df[f'breakdown_low_{window}'] = (df['close'] < df[f'low_{window}'].shift(1)).astype(int)
    
    # Breakout strength
    df[f'breakout_strength_{window}'] = np.maximum(
        df['close'] - df[f'high_{window}'].shift(1),
        df[f'low_{window}'].shift(1) - df['close']
    )
    df[f'breakout_strength_{window}'] = np.where(df[f'breakout_strength_{window}'] < 0, 0, df[f'breakout_strength_{window}'])
    
    # Price position in range
    df[f'price_position_{window}'] = (df['close'] - df[f'low_{window}']) / (df[f'high_{window}'] - df[f'low_{window}'])
    
    return df

def generate_labels(df, look_ahead=5, threshold=0.5):
    """Generate labels for ML model based on future price movement"""
    
    # Forward returns
    df['future_return_1'] = df['close'].shift(-1) / df['close'] - 1
    df['future_return_3'] = df['close'].shift(-3) / df['close'] - 1
    df['future_return_5'] = df['close'].shift(-5) / df['close'] - 1
    df['future_return_10'] = df['close'].shift(-10) / df['close'] - 1
    
    # Future high/low within next N periods
    df['future_high_5'] = df['high'].shift(-5).rolling(5, min_periods=1).max()
    df['future_low_5'] = df['low'].shift(-5).rolling(5, min_periods=1).min()
    
    # Maximum favorable/adverse excursion
    df['max_favorable_5'] = (df['future_high_5'] / df['close'] - 1)
    df['max_adverse_5'] = (df['close'] / df['future_low_5'] - 1)
    
    # Classification labels
    df['direction_1'] = np.where(df['future_return_1'] > threshold/100, 1, 
                                np.where(df['future_return_1'] < -threshold/100, -1, 0))
    df['direction_5'] = np.where(df['future_return_5'] > threshold/100, 1, 
                                np.where(df['future_return_5'] < -threshold/100, -1, 0))
    
    # Trade outcome labels (simulate strategy results)
    df['profitable_long'] = ((df['max_favorable_5'] > threshold*2/100) & 
                            (df['max_adverse_5'] < threshold/100)).astype(int)
    df['profitable_short'] = ((df['max_adverse_5'] > threshold*2/100) & 
                             (df['max_favorable_5'] < threshold/100)).astype(int)
    
    return df

def create_ml_dataset(file_path, output_path='ml_dataset.csv', threshold=0.5):
    """
    Main function to create ML dataset from OHLC data
    
    Parameters:
    file_path: Path to CSV file with OHLC data
    output_path: Path to save the ML dataset
    threshold: Threshold for classification (in percentage)
    """
    
    print("Loading data...")
    df = pd.read_csv(file_path, usecols=range(6))
    
    print("Parsing datetime...")
    datetime_objects = []
    for dt in df['date']:
        try:
            obj = datetime.strptime(dt, '%d-%m-%Y %H:%M')
        except ValueError:
            obj = parse(dt)
        datetime_objects.append(obj)
    
    df['datetime'] = datetime_objects
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print("Calculating technical indicators...")
    df = calculate_technical_indicators(df)
    
    print("Calculating time features...")
    df = calculate_time_features(df, df['datetime'])
    
    print("Calculating pattern features...")
    df = calculate_pattern_features(df)
    
    print("Calculating breakout features...")
    df = calculate_breakout_features(df, 30)
    df = calculate_breakout_features(df, 15)
    
    print("Generating labels...")
    df = generate_labels(df, look_ahead=5, threshold=threshold)
    
    # Market regime features
    print("Adding market regime features...")
    df['trend_strength'] = abs(df['ema_20'] - df['ema_50']) / df['close']
    df['volatility_regime'] = pd.qcut(df['volatility_20'].rank(), q=3, labels=[0, 1, 2])
    df['volume_regime'] = pd.qcut(df['volume_ratio'].rank(), q=3, labels=[0, 1, 2])
    
    # Interaction features
    df['rsi_ema_interaction'] = df['rsi'] * df['price_vs_ema20']
    df['volatility_breakout'] = df['volatility_20'] * df['breakout_strength_30']
    df['time_volatility'] = df['market_open_minutes'] * df['volatility_20']
    
    # Remove rows with NaN values (from rolling calculations)
    print("Cleaning data...")
    df = df.dropna()
    
    # Feature selection for ML model
    feature_columns = [
        # Price features
        'price_change', 'price_change_5', 'price_change_10', 'price_change_20',
        'volatility_5', 'volatility_10', 'volatility_20', 'price_range_norm',
        
        # Technical indicators
        'rsi', 'rsi_5', 'rsi_21', 'macd', 'macd_signal', 'macd_histogram',
        'stoch_k', 'stoch_d', 'williams_r', 'adx',
        
        # Moving averages
        'price_vs_ema20', 'price_vs_ema50', 'ema20_vs_ema50',
        
        # Bollinger Bands
        'bb_width', 'bb_position',
        
        # Support/Resistance
        'price_vs_resistance', 'price_vs_support',
        
        # Time features
        'hour', 'market_open_minutes', 'is_opening', 'is_closing', 'is_lunch_time',
        'hour_sin', 'hour_cos', 'day_of_week',
        
        # Pattern features
        'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
        'is_bullish', 'is_doji', 'is_hammer', 'is_shooting_star',
        'consecutive_bullish', 'consecutive_bearish',
        
        # Breakout features
        'breakout_high_30', 'breakdown_low_30', 'breakout_strength_30', 'price_position_30',
        'breakout_high_15', 'breakdown_low_15', 'breakout_strength_15', 'price_position_15',
        
        # Market regime
        'trend_strength', 'volatility_regime', 'volume_regime',
        
        # Interaction features
        'rsi_ema_interaction', 'volatility_breakout', 'time_volatility'
    ]
    
    target_columns = [
        'direction_1', 'direction_5', 'profitable_long', 'profitable_short',
        'future_return_1', 'future_return_3', 'future_return_5', 'future_return_10'
    ]
    
    # Create final dataset
    ml_data = df[['datetime'] + feature_columns + target_columns].copy()
    
    # Add strategy-specific features
    print("Adding strategy-specific features...")
    
    # EMA trend alignment
    ml_data['ema_bullish_alignment'] = (ml_data['ema20_vs_ema50'] > 1.001).astype(int)
    ml_data['ema_bearish_alignment'] = (ml_data['ema20_vs_ema50'] < 0.999).astype(int)
    
    # RSI conditions
    ml_data['rsi_oversold'] = (ml_data['rsi'] < 35).astype(int)
    ml_data['rsi_overbought'] = (ml_data['rsi'] > 65).astype(int)
    ml_data['rsi_neutral'] = ((ml_data['rsi'] >= 35) & (ml_data['rsi'] <= 65)).astype(int)
    
    # Combined signals
    ml_data['long_setup'] = (
        ml_data['ema_bullish_alignment'] & 
        (ml_data['rsi_neutral'] | (ml_data['rsi_oversold'] & (ml_data['rsi'] > 30))) &
        (ml_data['price_vs_ema20'] > 1) &
        (ml_data['breakout_high_30'] == 1)
    ).astype(int)
    
    ml_data['short_setup'] = (
        ml_data['ema_bearish_alignment'] & 
        (ml_data['rsi_neutral'] | (ml_data['rsi_overbought'] & (ml_data['rsi'] < 70))) &
        (ml_data['price_vs_ema20'] < 1) &
        (ml_data['breakdown_low_30'] == 1)
    ).astype(int)
    
    # Save the dataset
    ml_data.to_csv(output_path, index=False)
    
    # Print dataset info
    print(f"\n{'='*60}")
    print("ML DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {len(ml_data)}")
    print(f"Features: {len(feature_columns) + 6}")  # +6 for strategy features
    print(f"Dataset saved to: {output_path}")
    
    # Print feature importance insights
    print(f"\nFEATURE CATEGORIES:")
    print(f"• Technical Indicators: 13")
    print(f"• Time Features: 9") 
    print(f"• Pattern Features: 9")
    print(f"• Breakout Features: 8")
    print(f"• Price Features: 8")
    print(f"• Market Regime: 6")
    print(f"• Strategy Specific: 6")
    
    # Label distribution
    print(f"\nLABEL DISTRIBUTION:")
    print(f"Long Setups: {ml_data['long_setup'].sum()}")
    print(f"Short Setups: {ml_data['short_setup'].sum()}")
    print(f"Profitable Long Signals: {ml_data['profitable_long'].sum()}")
    print(f"Profitable Short Signals: {ml_data['profitable_short'].sum()}")
    
    # Sample data preview
    print(f"\nSAMPLE FEATURES (First 5 rows):")
    print(ml_data[['datetime'] + feature_columns[:5]].head())
    
    return ml_data

# Usage example
if __name__ == "__main__":
    
    # Generate the ML dataset
    file_path = r"D:\AlgoT\NIFTY 50 COPY.csv"
    
    try:
        ml_dataset = create_ml_dataset(
            file_path=file_path,
            output_path='trading_ml_dataset.csv',
            threshold=0.5  # 0.5% threshold for classification
        )
        
        print(f"\n✅ Dataset successfully created!")
        print(f"📊 Shape: {ml_dataset.shape}")
        print(f"💾 File: trading_ml_dataset.csv")
        
        # Quick model training example (optional)
        print(f"\n🤖 QUICK ML MODEL EXAMPLE:")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        
        # Prepare data for long setup prediction
        feature_cols = [col for col in ml_dataset.columns if col not in 
                       ['datetime', 'direction_1', 'direction_5', 'profitable_long', 
                        'profitable_short', 'future_return_1', 'future_return_3', 
                        'future_return_5', 'future_return_10']]
        
        X = ml_dataset[feature_cols].fillna(0)
        y = ml_dataset['profitable_long']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf.predict(X_test)
        
        print("Random Forest Performance (Profitable Long Prediction):")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please check the file path and ensure the CSV file exists.")