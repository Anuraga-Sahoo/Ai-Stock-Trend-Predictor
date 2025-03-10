let chartData = [];
      let layout = {
        title: 'Candlestick Chart',
        xaxis: { type: 'date' },
        yaxis: { title: 'Price' },
        showlegend: false
      };

      function subscribe() {
        const symbol = document.getElementById('symbol').value;
        fetch('/subscribe', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol })
        })
        .then(r => r.json())
        .then(data => {
          chartData = [{
            x: data.historicalData.map(d => new Date(d[0] * 1000)),
            open: data.historicalData.map(d => d[1]),
            high: data.historicalData.map(d => d[2]),
            low: data.historicalData.map(d => d[3]),
            close: data.historicalData.map(d => d[4]),
            type: 'candlestick'
          }];
          Plotly.newPlot('chart', chartData, layout);
        });
      }

      const eventSource = new EventSource('/stream');
      eventSource.onmessage = e => {
        const data = JSON.parse(e.data);
        if (!data || !data.ltp) return;

        const newDate = new Date(data.timestamp * 1000);
        const lastDate = chartData[0].x[chartData[0].x.length-1];
        
        if (newDate.getDate() === lastDate.getDate()) {
          // Update current candle
          chartData[0].close[chartData[0].close.length-1] = data.ltp;
          chartData[0].high[chartData[0].high.length-1] = 
            Math.max(chartData[0].high[chartData[0].high.length-1], data.ltp);
          chartData[0].low[chartData[0].low.length-1] = 
            Math.min(chartData[0].low[chartData[0].low.length-1], data.ltp);
        } else {
          // Add new candle
          chartData[0].x.push(newDate);
          chartData[0].open.push(data.ltp);
          chartData[0].high.push(data.ltp);
          chartData[0].low.push(data.ltp);
          chartData[0].close.push(data.ltp);
        }
        
        Plotly.update('chart', chartData, layout);
      };