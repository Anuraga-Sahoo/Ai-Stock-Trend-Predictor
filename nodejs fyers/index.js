require("dotenv").config();
const express = require("express");
const FyersApi = require("fyers-api-v3").fyersModel;
const FyersSocket = require("fyers-api-v3").fyersDataSocket;

const app = express();
const PORT = 5000;

// app.use(express.static("public"))

// Store connected clients and market data
let quoteClients = [];
let indexClients = [];
let marketData = {};

// Initialize Fyers API
const fyers = new FyersApi();
fyers.setAppId(process.env.CLIENT_ID);
fyers.setRedirectUrl(`http://127.0.0.1:${PORT}/callback`);

let generateAuthcodeURL = fyers.generateAuthCode();
let fyersdata = null;

// WebSocket configuration
const subscribedSymbols = {
    quotes: ['NSE:SBIN-EQ', 'NSE:RELIANCE-EQ'],
    indices: ['NSE:NIFTY50-INDEX', 'NSE:BANKNIFTY-INDEX']
};

// Historical data configuration
const defaultResolution = 'D';
const defaultDaysBack = 30;

// WebSocket handlers
function onmsg(message) {
    console.log("Real-time Update:", message);
    marketData[message.symbol] = message;

    if (message.type === 'if') {
        indexClients.forEach(client => {
            client.res.write(`data: ${JSON.stringify(message)}\n\n`);
        });
    } else {
        quoteClients.forEach(client => {
            client.res.write(`data: ${JSON.stringify(message)}\n\n`);
        });
    }
}

function onconnect() {
    console.log("WebSocket connected!");
    fyersdata.subscribe([...subscribedSymbols.quotes, ...subscribedSymbols.indices]);
    fyersdata.autoreconnect();
}

function onerror(err) {
    console.log("WebSocket error:", err);
}

function onclose() {
    console.log("WebSocket connection closed");
}

// Historical data endpoints
app.get('/historical/quotes', async (req, res) => {
    try {
        const { symbol, resolution = defaultResolution, days = defaultDaysBack } = req.query;
        
        if (!symbol) return res.status(400).json({ error: "Symbol parameter required" });
        if (!symbol.endsWith('-EQ')) return res.status(400).json({ error: "Invalid quote symbol format" });
        if (!fyers.AccessToken) return res.status(401).json({ error: "Unauthorized - Authenticate first" });

        const dateRange = getDateRange(days);
        const history = await fyers.getHistory({
            symbol,
            resolution,
            date_format: 1,
            range_from: dateRange.from,
            range_to: dateRange.to,
            cont_flag: "1"
        });

        res.json(formatHistoricalData(history, symbol));

    } catch (error) {
        console.error("Quote history error:", error);
        res.status(500).json({ error: "Failed to fetch historical data" });
    }
});

app.get('/historical/indices', async (req, res) => {
    try {
        const { symbol, resolution = defaultResolution, days = defaultDaysBack } = req.query;
        
        if (!symbol) return res.status(400).json({ error: "Symbol parameter required" });
        if (!symbol.endsWith('-INDEX')) return res.status(400).json({ error: "Invalid index symbol format" });
        if (!fyers.AccessToken) return res.status(401).json({ error: "Unauthorized - Authenticate first" });

        const dateRange = getDateRange(days);
        const history = await fyers.getHistory({
            symbol,
            resolution,
            date_format: 1,
            range_from: dateRange.from,
            range_to: dateRange.to,
            cont_flag: "1"
        });

        res.json(formatHistoricalData(history, symbol));

    } catch (error) {
        console.error("Index history error:", error);
        res.status(500).json({ error: "Failed to fetch historical data" });
    }
});

// SSE endpoints
app.get('/quote-stream', (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    Object.values(marketData).forEach(data => {
        if (data.type !== 'if') {
            res.write(`data: ${JSON.stringify(data)}\n\n`);
        }
    });

    const clientId = Date.now();
    quoteClients.push({ id: clientId, res });

    req.on('close', () => {
        quoteClients = quoteClients.filter(client => client.id !== clientId);
    });
});

app.get('/index-stream', (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    Object.values(marketData).forEach(data => {
        if (data.type === 'if') {
            res.write(`data: ${JSON.stringify(data)}\n\n`);
        }
    });

    const clientId = Date.now();
    indexClients.push({ id: clientId, res });

    req.on('close', () => {
        indexClients = indexClients.filter(client => client.id !== clientId);
    });
});

// Helper functions
function getDateRange(days) {
    const toDate = new Date();
    const fromDate = new Date();
    fromDate.setDate(toDate.getDate() - days);
    return {
        from: fromDate.toISOString().split('T')[0],
        to: toDate.toISOString().split('T')[0]
    };
}

function formatHistoricalData(history, symbol) {
    return {
        symbol,
        candles: history.candles?.map(c => ({
            timestamp: new Date(c[0] * 1000),
            open: c[1],
            high: c[2],
            low: c[3],
            close: c[4],
            volume: c[5]
        })) || []
    };
}

app.get("/", (req, res) => {
    res.send(`
        <h3>Click <a href=${generateAuthcodeURL}>here</a> for Access Token</h3>
        <script>
            const quoteSource = new EventSource('/quote-stream');
            quoteSource.onmessage = e => {
                const data = JSON.parse(e.data);
                console.log('Quote Update:', data);
            };

            const indexSource = new EventSource('/index-stream');
            indexSource.onmessage = e => {
                const data = JSON.parse(e.data);
                console.log('Index Update:', data);
            };
        </script>
    `);
});

app.get("/callback", async (req, res) => {
  const { auth_code } = req.query;
  console.log("auth code : ================= ", auth_code)
  if (!auth_code) return res.send("Authorization failed");
  try {
    console.log("try block working")
        const tokenResponse = await fyers.generate_access_token({
            secret_key: process.env.SECRET_KEY,
            auth_code: auth_code,
        });

        fyers.setAccessToken(tokenResponse.access_token);

        // fyersdata = new FyersSocket(
        //     tokenResponse.access_token,
        //     "./log",
        //     true
        // );
        fyersdata = new FyersSocket(`${process.env.CLIENT_ID}:${tokenResponse.access_token}`, "./log", true);
        fyersdata.on("message", onmsg);
        fyersdata.on("connect", onconnect);
        fyersdata.on("error", onerror);
        fyersdata.on("close", onclose);
        fyersdata.connect();

        // Initialize market data
        const initialResponse = await fyers.getQuotes([
            ...subscribedSymbols.quotes,
            ...subscribedSymbols.indices
        ]);
        
        if (initialResponse && initialResponse.d) {
            initialResponse.d.forEach(item => {
                marketData[item.symbol] = item;
            });
            res.json({
              message: "Authentication successful! WebSocket connected.",
              // initial_quote: quotes,
              liveIndex_endpoint: "http://127.0.0.1:5000/index-stream",
      liveQuote_endpoint: "http://127.0.0.1:5000/quote-stream",
      quotes_historicalData_endpoint: "http://127.0.0.1:5000/historical/quotes?symbol=NSE:SBIN-EQ&days=306",
      index_historicalData_endpoint: "http://127.0.0.1:5000/historical/indices?symbol=NSE:NIFTYBANK-INDEX&days=365"

            });
        }

        // res.redirect('/');
    } catch (error) {
        console.error("Authentication error:", error);
        res.status(500).json({error: "Authentication failed"});
    }
});

app.post('/request-data', async (req, res) => {
    const { symbols, data_type } = req.body;

    try {
        let data = [];
        for (const symbol of symbols) {
            const history = await fyers.getHistory({
                symbol,
                resolution: 'D',
                date_format: 1,
                range_from: new Date(new Date().setDate(new Date().getDate() - 30)).toISOString().split('T')[0],
                range_to: new Date().toISOString().split('T')[0],
                cont_flag: "1"
            });
            data.push(...history.candles.map(c => ({
                symbol,
                timestamp: new Date(c[0] * 1000),
                open: c[1],
                high: c[2],
                low: c[3],
                close: c[4],
                volume: c[5]
            })));
        }
        res.json(data);
    } catch (error) {
        console.error("Error fetching requested data:", error);
        res.status(500).json({ error: "Failed to fetch requested data" });
    }
});


// app.get("/callback", async (req, res) => {
//   const { auth_code } = req.query;
//   if (!auth_code) return res.send("Authorization failed - Missing auth code");

//   try {
//       // Add validation for secret key
//       if (!process.env.SECRET_KEY) {
//           throw new Error("Secret key not configured in environment variables");
//       }

//       const tokenResponse = await fyers.generate_access_token({
//           secret_key: process.env.SECRET_KEY,
//           auth_code: auth_code,
//       });

//       // Verify access token was received
//       if (!tokenResponse?.access_token) {
//           throw new Error("Failed to get access token from Fyers");
//       }

//       fyers.setAccessToken(tokenResponse.access_token);

//       // Initialize WebSocket with better error handling
//       fyersdata = new FyersSocket(
//           tokenResponse.access_token,
//           "./log",
//           true
//       );

//       // WebSocket event listeners
//       fyersdata.on("message", onmsg);
//       fyersdata.on("connect", onconnect);
//       fyersdata.on("error", onerror);
//       fyersdata.on("close", onclose);
      
//       // Connect WebSocket
//       fyersdata.connect((err) => {
//           if (err) {
//               console.error("WebSocket connection error:", err);
//               return res.status(500).send("WebSocket connection failed");
//           }
//       });

//       // Initialize market data with error handling
//       try {
//           const initialResponse = await fyers.getQuotes([
//               ...subscribedSymbols.quotes,
//               ...subscribedSymbols.indices
//           ]);
          
//           if (initialResponse?.d) {
//               initialResponse.d.forEach(item => {
//                   marketData[item.symbol] = item;
//               });
//           }
//       } catch (dataError) {
//           console.error("Initial data fetch error:", dataError);
//       }

//       res.redirect('/');

//   } catch (error) {
//       console.error("Authentication error details:", {
//           message: error.message,
//           stack: error.stack,
//           authCode: auth_code,
//           envVars: {
//               clientId: !!process.env.CLIENT_ID,
//               secretKey: !!process.env.SECRET_KEY
//           }
//       });
//       res.status(500).send(`Authentication failed: ${error.message}`);
//   }
// });



app.listen(PORT, () => {
    console.log(`Server running at http://127.0.0.1:${PORT}`);
});