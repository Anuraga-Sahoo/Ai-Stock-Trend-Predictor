// DOM Elements
const tickerInput = document.getElementById('tickerInput');
const predictionDays = document.getElementById('predictionDays');
const loadingOverlay = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const chart = document.getElementById('chart');
let chartInstance = null;

async function diagnoseAPIConnection() {
    try {
        const healthCheck = await fetch('/api/health');
        if (!healthCheck.ok) {
            throw new Error(`Health check failed: ${healthCheck.status}`);
        }
        const healthData = await healthCheck.json();

        const dataStatus = await fetch('/api/data/status');
        if (!dataStatus.ok) {
            throw new Error(`Data status check failed: ${dataStatus.status}`);
        }
        const statusData = await dataStatus.json();

        return {
            serverStatus: 'ok',
            healthCheck: healthData,
            dataSourceStatus: statusData
        };
    } catch (error) {
        return {
            serverStatus: 'error',
            error: error.message
        };
    }
}

async function getPrediction() {
    const ticker = tickerInput.value.trim();
    const days = predictionDays.value;
    
    if (!ticker) {
        showError('Please enter a valid ticker symbol');
        return;
    }
    
    try {
        showLoading(true);
        hideError();
        
        const diagnostics = await diagnoseAPIConnection();
        if (diagnostics.serverStatus === 'error') {
            throw new Error(`Server diagnostic failed: ${diagnostics.error}`);
        }
        
        const response = await fetch(`/api/predict/${ticker}?days=${days}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
            }
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
        }
        
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        
        updateChart(data);
        updateMetrics(data.metrics);
        updatePredictionsTable(data);
        
    } catch (error) {
        console.error('Prediction Error:', error);
        showError(`Prediction failed: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// function updateChart(data) {
//     if (chartInstance) {
//         chartInstance.destroy();
//     }
    
//     const ctx = chart.getContext('2d');
//     chartInstance = new Chart(ctx, {
//         type: 'line',
//         data: {
//             labels: data.dates,
//             datasets: [
//                 {
//                     label: 'Historical',
//                     data: data.values.slice(0, -data.predictions.length),
//                     borderColor: '#226fd4',
//                     backgroundColor: 'rgba(37, 99, 235, 0.1)',
//                     fill: true
//                 },
//                 {
//                     label: 'Predictions',
//                     data: [...Array(data.values.length - data.predictions.length).fill(null), ...data.predictions],
//                     borderColor: '#dc2626',
//                     borderDash: [5, 5]
//                 }
//             ]
//         },
//         options: {
//             responsive: true,
//             maintainAspectRatio: false,
//             interaction: {
//                 intersect: false,
//                 mode: 'index'
//             },
//             plugins: {
//                 legend: {
//                     position: 'top'
//                 },
//                 tooltip: {
//                     enabled: true
//                 }
//             },
//             scales: {
//               x:{
//                   ticks: {
//                       color: '#226fd4', // Hex color
//                       font: {
//                           size: 12,    // Font size
//                           family: 'Arial', // Font family
//                           weight: 'bold' // Font weight
//                       }
//                   },
//               },
//               y: {
//                   ticks: {
//                       gridcolor: '#f5f7fa',
//                       color: '#226fd4', // Hex color
//                       font: {
//                           size: 12,    // Font size
//                           family: 'Arial', // Font family
//                           weight: 'bold' // Font weight
//                       }
//                   },
//                   beginAtZero: false
//               }
//           }
//         }
//     });
// }

function updateChart(data) {
  if (chartInstance) {
      chartInstance.destroy();
  }
  
  const ctx = chart.getContext('2d');
  
  chartInstance = new Chart(ctx, {
      type: 'line',
      data: {
          labels: data.dates,
          datasets: [
              {
                  label: 'Historical',
                  data: data.values.slice(0, -data.predictions.length),
                  borderColor: '#226fd4',
                  backgroundColor: 'rgba(47, 107, 236, 0.1)',
                  fill: true
              },
              {
                  label: 'Predictions',
                  data: [...Array(data.values.length - data.predictions.length).fill(null), ...data.predictions],
                  borderColor: '#dc2626',
                  borderDash: [5, 5],
              }
          ]
      },
      options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: {
              intersect: false,
              mode: 'index'
          },
          plugins: {
              legend: {
                  position: 'top'
              },
              tooltip: {
                  enabled: true
              }
          },
          scales: {
              x: {
                  grid: {
                      color: 'rgba(255, 255, 255, 0.1)', // White grid lines for X-axis
                      lineWidth: 1
                  },
                  ticks: {
                      color: '#226fd4',
                      font: {
                          size: 12,
                          family: 'Arial',
                          weight: 'bold'
                      }
                  },
              },
              y: {
                  grid: {
                      color: 'rgba(255, 255, 255, 0.1)', // White grid lines for Y-axis
                      lineWidth: 1
                  },
                  ticks: {
                      color: '#226fd4',
                      font: {
                          size: 12,
                          family: 'Arial',
                          weight: 'bold'
                      }
                  },
                  beginAtZero: false
              }
          }
      }
  });
}

function updateMetrics(metrics) {
    if (metrics) {
        document.getElementById('accuracyMetric').textContent = `${(metrics.accuracy).toFixed(2)}%`;
        document.getElementById('mapeMetric').textContent = `${(metrics.mape).toFixed(2)}%`;
        document.getElementById('r2Metric').textContent = metrics.r2.toFixed(3);
    }
}

function updatePredictionsTable(data) {
    const tbody = document.getElementById('predictionsBody');
    tbody.innerHTML = '';
    
    data.predictions.forEach((prediction, index) => {
        const row = document.createElement('tr');
        const date = new Date(data.dates[data.dates.length - data.predictions.length + index]);
        const prevValue = index === 0 ? data.lastClose : data.predictions[index - 1];
        const change = ((prediction - prevValue) / prevValue * 100).toFixed(2);
        
        row.innerHTML = `
            <td>Day ${index + 1}</td>
            <td>${date.toLocaleDateString()}</td>
            <td>${prediction.toFixed(2)}</td>
            <td class="${change >= 0 ? 'gain' : 'loss'}">${change}%</td>
            <td>
                <div class="confidence-level">
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${Math.max(60, 90 - index * 5)}%"></div>
                    </div>
                </div>
            </td>
            <td>
                <i class="fas fa-${change >= 0 ? 'arrow-up trend-up' : 'arrow-down trend-down'}"></i>
            </td>
        `;
        tbody.appendChild(row);
    });
}

function toggleFullscreen() {
    const chartCard = document.querySelector('.chart-card');
    chartCard.classList.toggle('fullscreen');
    if (chartInstance) {
        chartInstance.resize();
    }
}

function exportToCSV() {
    if (!chartInstance) return;
    
    const data = chartInstance.data;
    let csv = 'Date,Value\n';
    
    data.labels.forEach((date, i) => {
        const value = data.datasets[1].data[i] || data.datasets[0].data[i];
        if (value !== null) {
            csv += `${date},${value}\n`;
        }
    });
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'predictions.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

function showLoading(show) {
    loadingOverlay.style.display = show ? 'block' : 'none';
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

function hideError() {
    errorDiv.style.display = 'none';
}

// Initialize any required event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Add input validation if needed
    tickerInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') getPrediction();
    });
});