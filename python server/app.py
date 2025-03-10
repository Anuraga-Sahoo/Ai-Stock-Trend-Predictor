from flask import Flask, request, jsonify
import requests
import pandas as pd
import json

app = Flask(__name__)

# Node.js server URLs
NODEJS_SERVER = "http://127.0.0.1:5000"

# Fetch live index data from Node.js server
def fetch_live_index_data():
    url = f"{NODEJS_SERVER}/index-stream"
    response = requests.get(url, stream=True)
    data = []
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("data:"):
                json_data = decoded_line.replace("data: ", "")
                data.append(json.loads(json_data))
    return pd.DataFrame(data)

# Fetch historical data from Node.js server
def fetch_historical_data(symbol, days, data_type="quotes"):
    endpoint = f"{NODEJS_SERVER}/historical/{data_type}"
    params = {"symbol": symbol, "days": days}
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        return pd.DataFrame(response.json()["candles"])
    else:
        print(f"Error fetching historical data: {response.status_code}")
        return pd.DataFrame()

# Send POST request to Node.js server to get specific data
def request_specific_data(symbols, data_type="quotes"):
    url = f"{NODEJS_SERVER}/request-data"
    payload = {"symbols": symbols, "data_type": data_type}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        print(f"Error requesting specific data: {response.status_code}")
        return pd.DataFrame()

# Routes
@app.route("/fetch-live-index", methods=["GET"])
def get_live_index_data():
    df = fetch_live_index_data()
    print("Live Index Data:")
    print(df)
    return jsonify(df.to_dict(orient="records"))

@app.route("/fetch-historical", methods=["GET"])
def get_historical_data():
    symbol = request.args.get("symbol")
    days = int(request.args.get("days", 30))
    data_type = request.args.get("data_type", "quotes")
    df = fetch_historical_data(symbol, days, data_type)
    print(f"Historical Data for {symbol}:")
    print(df)
    # return jsonify(df.to_dict(orient="records"))
    return jsonify({
            "symbol": symbol,
            "data": df.to_dict(orient='records')
        })

@app.route('/get-index-history', methods=['GET'])
def get_index_history():
    """Endpoint specifically for index historical data"""
    symbol = request.args.get('symbol', 'NSE:NIFTY50-INDEX')
    days = int(request.args.get('days', 365))
    
    df = fetch_historical_data(symbol, days, data_type="indices")
    
    if not df.empty:
        print(f"\nHistorical Index Data for {symbol} ({days} days):")
        print(df.tail())  # Print last 5 records
        return jsonify({
            "symbol": symbol,
            "data": df.to_dict(orient='records')
        })
    return jsonify({"error": "Failed to fetch index data"}), 500

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)