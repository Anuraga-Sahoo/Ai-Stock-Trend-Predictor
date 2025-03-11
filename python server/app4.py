from flask import Flask, request, jsonify, render_template
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import yfinance as yf
import json
import logging
import time
from datetime import datetime, timedelta
from niftystocks import ns
from flask_cors import CORS

###############################################################################
# Configuration
###############################################################################

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Node.js server configuration (FYERS API gateway)
NODEJS_SERVER = "http://127.0.0.1:3000"  # Update with your Node.js server address

INDICES = {
    'NIFTY50': {
        'yf': '^NSEI',
        'ns': 'NIFTY 50',
        'nse': 'NIFTY 50',
        'fyers': 'NSE:NIFTY50-INDEX'
    },
    'SENSEX': {
        'yf': '^BSESN',
        'ns': 'SENSEX',
        'nse': 'SENSEX',
        'fyers': 'BSE:SENSEX-INDEX'
    },
    'BANKNIFTY': {
        'yf': '^NSEBANK',
        'ns': 'NIFTY BANK',
        'nse': 'NIFTY BANK',
        'fyers': 'NSE:NIFTYBANK-INDEX'
    },
}

###############################################################################
# Neural Network Components
###############################################################################

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.sum(attention_weights * hidden_states, dim=1)
        return attended

class EnhancedStockBiLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=3, output_size=30):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_size * 2)
        self.dropout = nn.Dropout(0.3)
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.norm2 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm1(x, (h0, c0))
        out = self.norm1(out)
        attended = self.attention(out)
        attended = self.dropout(attended)
        
        out = self.fc1(attended)
        out = self.relu(out)
        out = self.norm2(out)
        out = self.dropout(out)
        return self.fc2(out)

###############################################################################
# Data Processing
###############################################################################

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataFetchError(Exception):
    pass

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + (gain / loss)))

def calculate_macd(prices, fast=12, slow=26):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

def prepare_data(data, sequence_length=30, prediction_days=7):
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        sequences = []
        targets = []
        
        for i in range(len(scaled_data) - sequence_length - prediction_days + 1):
            seq = scaled_data[i:i + sequence_length]
            target = scaled_data[i + sequence_length:i + sequence_length + prediction_days, 0]
            sequences.append(seq)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        split = int(len(sequences) * 0.8)
        return (sequences[:split], sequences[split:], 
                targets[:split], targets[split:], scaler)
        
    except Exception as e:
        logging.error(f"Data preparation error: {str(e)}")
        raise

###############################################################################
# Fyers Data Integration
###############################################################################

def fetch_fyers_historical(symbol, days, data_type="indices"):
    try:
        endpoint = f"{NODEJS_SERVER}/historical/{data_type}"
        params = {"symbol": symbol, "days": days}
        response = requests.get(endpoint, params=params)
        
        if response.status_code == 200:
            df = pd.DataFrame(response.json()["candles"])
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('Date')
            return df
        raise DataFetchError(f"API error: {response.status_code}")
    except Exception as e:
        raise DataFetchError(str(e))

def prepare_features(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    df = df.dropna()
    return df[['Close', 'Volume', 'MA20', 'RSI', 'MACD']].values

###############################################################################
# Data Fetching with Fallbacks
###############################################################################

def get_data_with_fallbacks(symbol, period='1y', max_retries=3):
    errors = []
    days = 365 if period == '1y' else 730

    # Try Fyers API first
    try:
        fyers_symbol = INDICES[symbol]['fyers']
        df = fetch_fyers_historical(fyers_symbol, days, "indices")
        features = prepare_features(df)
        return features, df['Close'].values
    except Exception as e:
        errors.append(f"Fyers error: {str(e)}")
        logging.warning(f"Fyers failed: {str(e)}")

    # Fallback to niftystocks
    try:
        index_name = INDICES[symbol]['ns']
        data = ns.get_historical_index(index_name, days=days)
        df = pd.DataFrame(data).rename(columns={
            'Open Price': 'Open', 'High Price': 'High',
            'Low Price': 'Low', 'Close Price': 'Close'
        })
        features = prepare_features(df)
        return features, df['Close'].values
    except Exception as e:
        errors.append(f"NSE error: {str(e)}")
        logging.warning(f"NSE failed: {str(e)}")

    # Fallback to yfinance
    try:
        yf_symbol = INDICES[symbol]['yf']
        df = yf.Ticker(yf_symbol).history(period=period)
        features = prepare_features(df)
        return features, df['Close'].values
    except Exception as e:
        errors.append(f"YFinance error: {str(e)}")
        logging.error(f"All sources failed: {'; '.join(errors)}")
        raise DataFetchError(f"Data unavailable: {'; '.join(errors)}")

###############################################################################
# Flask Routes
###############################################################################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/indices')
def get_indices():
    return jsonify(list(INDICES.keys()))

@app.route('/api/predict/<symbol>')
def predict(symbol):
    try:
        if symbol not in INDICES:
            return jsonify({'error': 'Invalid index'}), 400
            
        prediction_days = int(request.args.get('days', 7))
        if prediction_days not in [7, 15, 30]:
            prediction_days = 7

        features, close_prices = get_data_with_fallbacks(symbol)
        X_train, X_val, y_train, y_val, scaler = prepare_data(features, 
                                    prediction_days=prediction_days)

        model = EnhancedStockBiLSTM(input_size=features.shape[1], 
                                  output_size=prediction_days)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_loader = DataLoader(StockDataset(X_train, y_train), 
                                batch_size=32, shuffle=True)
        
        model.train()
        for epoch in range(50):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()

        last_sequence = features[-30:]
        X = torch.FloatTensor(scaler.transform(last_sequence)).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            scaled_preds = model(X)[0].numpy()
        
        pred_reshaped = np.zeros((len(scaled_preds), scaler.n_features_in_))
        pred_reshaped[:, 0] = scaled_preds
        predictions = scaler.inverse_transform(pred_reshaped)[:, 0]

        return jsonify({
            'dates': [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                     for i in range(-29, prediction_days + 1)],
            'values': close_prices[-30:].tolist() + predictions.tolist(),
            'predictions': predictions.tolist(),
            'metrics': calculate_metrics(model, X_val, y_val, scaler)
        })

    except DataFetchError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

def calculate_metrics(model, X_val, y_val, scaler):
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_val))
        pred_reshaped = np.zeros((len(preds), scaler.n_features_in_))
        pred_reshaped[:, 0] = preds.numpy()[:, 0]
        y_reshaped = np.zeros((len(y_val), scaler.n_features_in_))
        y_reshaped[:, 0] = y_val[:, 0]
        
        pred_actual = scaler.inverse_transform(pred_reshaped)[:, 0]
        y_actual = scaler.inverse_transform(y_reshaped)[:, 0]
        
        return {
            'mape': float(mean_absolute_percentage_error(y_actual, pred_actual)),
            'r2': float(r2_score(y_actual, pred_actual)),
            'accuracy': float(100 - mean_absolute_percentage_error(y_actual, pred_actual))
        }

###############################################################################
# Fyers API Endpoints
###############################################################################

@app.route("/fetch-historical", methods=["GET"])
def get_historical():
    symbol = request.args.get("symbol")
    days = int(request.args.get("days", 30))
    data_type = request.args.get("data_type", "quotes")
    df = fetch_fyers_historical(symbol, days, data_type)
    return jsonify({
        "symbol": symbol,
        "data": df.to_dict(orient='records')
    })

@app.route('/get-index-history', methods=['GET'])
def get_index_history():
    symbol = request.args.get('symbol', 'NSE:NIFTY50-INDEX')
    days = int(request.args.get('days', 365))
    df = fetch_fyers_historical(symbol, days, "indices")
    return jsonify({
        "symbol": symbol,
        "data": df.to_dict(orient='records')
    })

@app.route("/request-data", methods=["POST"])
def request_data():
    payload = request.json
    symbols = payload.get("symbols")
    data_type = payload.get("data_type", "quotes")
    df = request_specific_data(symbols, data_type)
    print(f"Requested Data for {symbols}:")
    print(df)
    return jsonify(df.to_dict(orient="records"))

@app.route('/request-index-data', methods=['POST'])
def request_index_data():
    """POST endpoint for custom index data requests"""
    payload = request.json
    symbols = payload.get('symbols', ['NSE:NIFTY50-INDEX'])
    days = payload.get('days', 365)
    
    results = {}
    for symbol in symbols:
        df = fetch_historical_data(symbol, days, data_type="indices")
        results[symbol] = df.to_dict(orient='records') if not df.empty else []
    
    print("\nCustom Index Data Request Results:")
    for symbol, data in results.items():
        print(f"{symbol}: {len(data)} records")
    
    return jsonify(results)


#######################################################################################################################
#Function: health_check
#Input: None
#Output: jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat(), 'version': '1.0.0'})
#Description: Function to check the health of the application
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
@app.route('/api/health')
def health_check():
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    except Exception as e:
        logging.error(f"Health check error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

#######################################################################################################################
#Function: not_found_error
#Input: error
#Output: jsonify({'error': 'Not found'}), 404
#Description: Function to handle 404 errors
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

#######################################################################################################################
#Function: internal_error
#Input: error
#Output: jsonify({'error': 'Internal server error'}), 500
#Description: Function to handle 500
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

#######################################################################################################################
#Function: service_unavailable_error
#Input: error
#Output: jsonify({'error': 'Service temporarily unavailable'}), 503
#Description: Function to handle 503 errors
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################
@app.errorhandler(503)
def service_unavailable_error(error):
    return jsonify({'error': 'Service temporarily unavailable'}), 503

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    

#######################################################################################################################
#Function: initialize_app
#Input: app
#Output: None
#Description: This function initializes the application state by checking the data sources
#Author: Ojas Ulhas Dighe
#Date: 24-Feb-2025
#######################################################################################################################    
    initialize_app(app)
    
    



###############################################################################
# Execution
###############################################################################

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)