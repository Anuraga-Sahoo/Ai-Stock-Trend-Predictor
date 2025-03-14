// DOM Elements
const indexSelect = document.getElementById('indexSelect');
const predictionDays = document.getElementById('predictionDays');
const loadingOverlay = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const chart = document.getElementById('chart');
let chartInstance = null;


async function diagnoseAPIConnection() {
  try {
      // First check if server is reachable
      const healthCheck = await fetch('/api/health');
      if (!healthCheck.ok) {
          throw new Error(`Health check failed: ${healthCheck.status}`);
      }
      const healthData = await healthCheck.json();
      console.log('Health check:', healthData);

      // Check data sources
      const dataStatus = await fetch('/api/data/status');
      if (!dataStatus.ok) {
          throw new Error(`Data status check failed: ${dataStatus.status}`);
      }
      const statusData = await dataStatus.json();
      console.log('Data sources status:', statusData);

      // Test prediction endpoint with minimal data
      const testPrediction = await fetch('/api/predict/ITC?days=7', {
          method: 'GET',
          headers: {
              'Accept': 'application/json',
          }
      });
      
      const predictionResponse = await testPrediction.text();
      console.log('Prediction endpoint response:', predictionResponse);

      if (!testPrediction.ok) {
          throw new Error(`Prediction test failed: ${testPrediction.status} - ${predictionResponse}`);
      }

      return {
          serverStatus: 'ok',
          healthCheck: healthData,
          dataSourceStatus: statusData,
          predictionStatus: 'ok'
      };
  } catch (error) {
      console.error('API Diagnosis Error:', error);
      return {
          serverStatus: 'error',
          error: error.message
      };
  }
}

// Fetch available indices when page loads
async function fetchIndices() {
    try {
        const response = await fetch('/api/qindices');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const indices = await response.json();
        
        // Clear existing options except the placeholder
        indexSelect.innerHTML = '<option value="">Choose an index...</option>';
        
        // Add new options
        indices.forEach(index => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = index;
            indexSelect.appendChild(option);
        });
    } catch (error) {
        showError(`Failed to load indices: ${error.message}`);
    }
}

// Get prediction for selected index
async function getPrediction() {
  const selectedIndex = indexSelect.value;
  const days = predictionDays.value;
  
  if (!selectedIndex) {
      showError('Please select an index');
      return;
  }
  
  try {
      showLoading(true);
      hideError();
      
      // Run diagnostics first
      const diagnostics = await diagnoseAPIConnection();
      if (diagnostics.serverStatus === 'error') {
          throw new Error(`Server diagnostic failed: ${diagnostics.error}`);
      }
      
      const response = await fetch(`/api/predict/${selectedIndex}?days=${days}`, {
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
      console.error('Detailed error:', error);
      showError(`Prediction failed: ${error.message}`);
  } finally {
      showLoading(false);
  }
}

// Update chart with new data
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
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    fill: true
                },
                {
                    label: 'Predictions',
                    data: [...Array(data.values.length - data.predictions.length).fill(null), ...data.predictions],
                    borderColor: '#dc2626',
                    borderDash: [5, 5]
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
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// Update metrics display
function updateMetrics(metrics) {
    if (metrics) {
        document.getElementById('accuracyMetric').textContent = `${(metrics.accuracy).toFixed(2)}%`;
        document.getElementById('mapeMetric').textContent = `${(metrics.mape).toFixed(2)}%`;
        document.getElementById('r2Metric').textContent = metrics.r2.toFixed(3);
    }
}

// Update predictions table
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

// Toggle fullscreen chart
function toggleFullscreen() {
    const chartCard = document.querySelector('.chart-card');
    chartCard.classList.toggle('fullscreen');
    if (chartInstance) {
        chartInstance.resize();
    }
}

// Export predictions to CSV
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

// Utility functions
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

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    fetchIndices();
});