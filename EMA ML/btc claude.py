import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
    from tensorflow.keras.optimizers import Adam
    KERAS_AVAILABLE = True
    print("TensorFlow version:", tf.__version__)
except ImportError:
    print("TensorFlow not available. Deep learning models will be skipped.")
    KERAS_AVAILABLE = False

class BTCMLTrainer:
    def __init__(self, data_file=None, df=None):
        if data_file:
            self.df = pd.read_csv(r"D:\AlgoT\btcusd_1-min_data.csv")
        elif df is not None:
            self.df = df.copy()
        else:
            raise ValueError("Either data_file or df must be provided")
        
        self.prepare_data()
        self.models = {}
        self.results = {}
        self.scalers = {}
        
    def prepare_data(self):
        # Convert timestamp
        if 'Timestamp' in self.df.columns:
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], unit='s')
            self.df.set_index('Timestamp', inplace=True)
        
        # Select OHLC columns
        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        if 'Volume' in self.df.columns:
            ohlc_cols.append('Volume')
        
        self.df = self.df[ohlc_cols]
        
        # Remove static price periods
        price_cols = ['Open', 'High', 'Low', 'Close']
        mask = ~(self.df[price_cols].nunique(axis=1) == 1)
        self.df = self.df[mask]
        
        print(f"Data shape after cleaning: {self.df.shape}")
        print(f"Data range: {self.df.index.min()} to {self.df.index.max()}")
    
    def create_features(self, lookback_window=60, forecast_horizon=5):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        
        print("Adding technical indicators...")
        self.add_technical_indicators()
        
        print("Creating sequences...")
        X, y = self.create_sequences()
        
        if len(X) == 0:
            raise ValueError("Not enough data for sequences. Try smaller lookback_window.")
        
        # Train/test split (chronological)
        split_idx = int(len(X) * 0.8)
        
        self.X_train = X[:split_idx]
        self.X_test = X[split_idx:]
        self.y_train = y[:split_idx]
        self.y_test = y[split_idx:]
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        
    def add_technical_indicators(self):
        df = self.df.copy()
        
        # Basic features
        df['price_change'] = df['Close'].pct_change()
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
        df['open_close_pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # Moving averages
        for window in [5, 10, 20]:
            df[f'sma_{window}'] = df['Close'].rolling(window).mean()
            df[f'ema_{window}'] = df['Close'].ewm(span=window).mean()
        
        # Volatility
        df['volatility_10'] = df['Close'].rolling(10).std()
        df['volatility_20'] = df['Close'].rolling(20).std()
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['Close'])
        
        # Bollinger Bands (FIXED)
        bb_period = 20
        bb_mean = df['Close'].rolling(bb_period).mean()
        bb_std = df['Close'].rolling(bb_period).std()
        df['bb_upper'] = bb_mean + (bb_std * 2)
        df['bb_lower'] = bb_mean - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        if 'Volume' in df.columns:
            df['volume_sma'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / (df['volume_sma'] + 1e-8)
        
        self.df_features = df.dropna()
        print(f"Features created. Shape: {self.df_features.shape}")
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_sequences(self):
        data = self.df_features.values
        X, y = [], []
        
        for i in range(self.lookback_window, len(data) - self.forecast_horizon + 1):
            # Features: past lookback_window periods
            X.append(data[i-self.lookback_window:i])
            
            # Target: next forecast_horizon periods of OHLC
            target = data[i:i+self.forecast_horizon, :4]  # Only OHLC
            y.append(target.flatten())
        
        return np.array(X), np.array(y)
    
    def scale_data(self):
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        # Reshape for scaling
        X_train_2d = self.X_train.reshape(-1, self.X_train.shape[-1])
        X_test_2d = self.X_test.reshape(-1, self.X_test.shape[-1])
        
        # Fit and transform
        scaler_X.fit(X_train_2d)
        scaler_y.fit(self.y_train)
        
        X_train_scaled_2d = scaler_X.transform(X_train_2d)
        X_test_scaled_2d = scaler_X.transform(X_test_2d)
        
        self.X_train_scaled = X_train_scaled_2d.reshape(self.X_train.shape)
        self.X_test_scaled = X_test_scaled_2d.reshape(self.X_test.shape)
        self.y_train_scaled = scaler_y.transform(self.y_train)
        self.y_test_scaled = scaler_y.transform(self.y_test)
        
        self.scalers = {'X': scaler_X, 'y': scaler_y}
    
    def train_models(self):
        # Traditional ML models
        X_train_flat = self.X_train_scaled.reshape(len(self.X_train_scaled), -1)
        X_test_flat = self.X_test_scaled.reshape(len(self.X_test_scaled), -1)
        
        from sklearn.multioutput import MultiOutputRegressor
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=50, random_state=42), n_jobs=-1),
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0)
        }
        
        print("Training traditional ML models...")
        for name, model in models.items():
            print(f"  {name}...")
            model.fit(X_train_flat, self.y_train_scaled)
            y_pred_scaled = model.predict(X_test_flat)
            y_pred = self.scalers['y'].inverse_transform(y_pred_scaled)
            
            self.models[name] = model
            self.results[name] = {'predictions': y_pred}
        
        # Deep learning models
        if KERAS_AVAILABLE:
            print("Training deep learning models...")
            
            # LSTM
            print("  LSTM...")
            lstm = Sequential([
                LSTM(32, return_sequences=True, input_shape=(self.lookback_window, self.X_train.shape[2])),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(16),
                Dense(self.y_train.shape[1])
            ])
            lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            lstm.fit(self.X_train_scaled, self.y_train_scaled, epochs=20, batch_size=32, verbose=0)
            
            y_pred_scaled = lstm.predict(self.X_test_scaled, verbose=0)
            y_pred = self.scalers['y'].inverse_transform(y_pred_scaled)
            
            self.models['LSTM'] = lstm
            self.results['LSTM'] = {'predictions': y_pred}
    
    def evaluate_models(self):
        evaluation_results = {}
        
        for name, result in self.results.items():
            y_pred = result['predictions']
            
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # R² for each output
            r2_scores = []
            for i in range(self.y_test.shape[1]):
                r2_scores.append(r2_score(self.y_test[:, i], y_pred[:, i]))
            r2_avg = np.mean(r2_scores)
            
            evaluation_results[name] = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2_avg
            }
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def plot_results(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('BTC Price Prediction Results', fontsize=16)
        
        # 1. Model comparison
        ax = axes[0, 0]
        results_df = pd.DataFrame(self.evaluation_results).T
        results_df[['RMSE', 'MAE']].plot(kind='bar', ax=ax)
        ax.set_title('Model Comparison')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. R² scores
        ax = axes[0, 1]
        r2_scores = [self.evaluation_results[name]['R²'] for name in self.evaluation_results.keys()]
        ax.bar(self.evaluation_results.keys(), r2_scores)
        ax.set_title('R² Scores')
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Best model predictions
        best_model = min(self.evaluation_results.keys(), 
                        key=lambda x: self.evaluation_results[x]['RMSE'])
        
        ax = axes[0, 2]
        # Extract close prices (every 4th element starting from index 3)
        actual_close = self.y_test[:, 3::4][:, 0]  # First forecast step
        pred_close = self.results[best_model]['predictions'][:, 3::4][:, 0]
        
        n_plot = min(100, len(actual_close))
        ax.plot(actual_close[:n_plot], label='Actual', alpha=0.7)
        ax.plot(pred_close[:n_plot], label='Predicted', alpha=0.7)
        ax.set_title(f'Close Price - {best_model}')
        ax.legend()
        
        # 4. Residuals
        ax = axes[1, 0]
        residuals = actual_close - pred_close
        ax.scatter(range(len(residuals[:200])), residuals[:200], alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_title('Residuals')
        
        # 5. Price distribution
        ax = axes[1, 1]
        ax.hist(actual_close, bins=30, alpha=0.7, label='Actual')
        ax.hist(pred_close, bins=30, alpha=0.7, label='Predicted')
        ax.set_title('Price Distribution')
        ax.legend()
        
        # 6. Model predictions correlation
        ax = axes[1, 2]
        pred_df = pd.DataFrame({name: result['predictions'][:, 3] for name, result in self.results.items()})
        corr = pred_df.corr()
        im = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45)
        ax.set_yticklabels(corr.columns)
        ax.set_title('Model Correlation')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()
    
    def run_pipeline(self, lookback_window=60, forecast_horizon=5):
        print("=== BTC ML Training Pipeline ===")
        
        try:
            print("\n1. Creating features...")
            self.create_features(lookback_window, forecast_horizon)
            
            print("2. Scaling data...")
            self.scale_data()
            
            print("3. Training models...")
            self.train_models()
            
            print("4. Evaluating models...")
            results = self.evaluate_models()
            
            print("\n=== RESULTS ===")
            results_df = pd.DataFrame(results).T
            print(results_df.round(4))
            
            best_model = results_df['RMSE'].idxmin()
            print(f"\nBest model: {best_model}")
            
            print("\n5. Creating plots...")
            self.plot_results()
            
            return results
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None

# Generate sample data and run
if __name__ == "__main__":
    print("Creating sample BTC data...")
    
    np.random.seed(42)
    n_samples = 5000
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='1min')
    
    prices = []
    current_price = 45000
    
    for i in range(n_samples):
        change = np.random.normal(0.0001, 0.02)
        current_price *= (1 + change)
        
        volatility = abs(np.random.normal(0, 0.005))
        high = current_price * (1 + volatility)
        low = current_price * (1 - volatility)
        open_price = current_price * (1 + np.random.normal(0, 0.002))
        
        prices.append({
            'Timestamp': timestamps[i].timestamp(),
            'Open': max(open_price, 0),
            'High': max(high, open_price, current_price),
            'Low': min(low, open_price, current_price),
            'Close': max(current_price, 0),
            'Volume': np.random.randint(10, 1000)
        })
    
    df = pd.DataFrame(prices)
    print(f"Sample data created: {df.shape}")
    
    # Run pipeline
    trainer = BTCMLTrainer(df=df)
    results = trainer.run_pipeline(lookback_window=30, forecast_horizon=3)
