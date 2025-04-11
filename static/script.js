document.addEventListener('DOMContentLoaded', function() {
    const errorMessage = document.getElementById('error-message');
    const predictionResult = document.getElementById('prediction-result');
    const featureImportanceList = document.getElementById('feature-importance-list');
    const companySelect = document.getElementById('company-select');
    const companyTitle = document.getElementById('company-title');
    const currentPriceValue = document.getElementById('current-price-value');
    const predictedPriceValue = document.getElementById('predicted-price-value');
    const priceChange = document.getElementById('price-change');
    
    let priceChart = null;
    let rsiChart = null;
    let macdChart = null;
    
    // Şirket listesini yükle
    fetch('/get_companies')
        .then(response => response.json())
        .then(companies => {
            companySelect.innerHTML = '<option value="">Şirket Seçin</option>';
            Object.entries(companies).forEach(([code, name]) => {
                const option = document.createElement('option');
                option.value = code;
                option.textContent = `${name} (${code})`;
                companySelect.appendChild(option);
            });
        })
        .catch(error => {
            console.error('Şirket listesi yüklenirken hata:', error);
            errorMessage.textContent = 'Şirket listesi yüklenirken bir hata oluştu.';
        });
    
    // Şirket seçildiğinde verileri yükle
    companySelect.addEventListener('change', function() {
        const selectedCompany = this.value;
        if (!selectedCompany) return;
        
        loadCompanyData(selectedCompany);
    });
    
    function loadCompanyData(company) {
        // Grafikleri temizle ve yükleniyor mesajı göster
        const containers = document.querySelectorAll('.chart-container');
        containers.forEach(container => {
            const canvas = container.querySelector('canvas');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.font = '14px Arial';
            ctx.fillStyle = '#666';
            ctx.textAlign = 'center';
            ctx.fillText('Veriler yükleniyor...', canvas.width/2, canvas.height/2);
        });
        
        // Mevcut grafikleri temizle
        if (priceChart) priceChart.destroy();
        if (rsiChart) rsiChart.destroy();
        if (macdChart) macdChart.destroy();
        
        // Verileri getir
        fetch(`/get_stock_data/${company}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Şirket başlığını güncelle
                companyTitle.textContent = `${data.company_name} - Yapay Zeka Tahmini`;
                
                // Fiyat bilgilerini güncelle
                currentPriceValue.textContent = `${data.current_price} ₺`;
                predictedPriceValue.textContent = `${data.predicted_price} ₺`;
                
                // Fiyat değişimini güncelle
                const changeClass = data.price_change_percent >= 0 ? 'positive' : 'negative';
                const changeSymbol = data.price_change_percent >= 0 ? '+' : '';
                priceChange.textContent = `${changeSymbol}${data.price_change_percent}%`;
                priceChange.className = changeClass;
                
                // Tahmin sonuçlarını göster
                const prediction = data.prediction;
                const probability = data.probability;
                const direction = prediction === 1 ? "YÜKSELİŞ" : "DÜŞÜŞ";
                const prob = (probability[prediction] * 100).toFixed(2);
                
                predictionResult.innerHTML = `
                    <strong>Yarın için Tahmin:</strong> ${direction}<br>
                    <strong>Tahmin Olasılığı:</strong> %${prob}
                `;
                predictionResult.className = prediction === 1 ? 'up' : 'down';
                
                // Özellik önemlerini göster
                const features = Object.entries(data.feature_importance)
                    .sort((a, b) => b[1] - a[1]);
                
                featureImportanceList.innerHTML = features.map(([feature, importance]) => `
                    <div class="feature-item">
                        <div>${feature}</div>
                        <div class="value">${(importance * 100).toFixed(2)}%</div>
                    </div>
                `).join('');
                
                // Fiyat Grafiği
                const priceCtx = document.getElementById('priceChart').getContext('2d');
                priceChart = new Chart(priceCtx, {
                    type: 'line',
                    data: {
                        labels: data.dates,
                        datasets: [
                            {
                                label: 'Kapanış Fiyatı',
                                data: data.close,
                                borderColor: 'rgb(75, 192, 192)',
                                tension: 0.1
                            },
                            {
                                label: 'SMA 10',
                                data: data.sma_10,
                                borderColor: 'rgb(255, 99, 132)',
                                tension: 0.1
                            },
                            {
                                label: 'SMA 50',
                                data: data.sma_50,
                                borderColor: 'rgb(54, 162, 235)',
                                tension: 0.1
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: `${data.company_name} Fiyat Grafiği ve Hareketli Ortalamalar`
                            }
                        }
                    }
                });
                
                // RSI Grafiği
                const rsiCtx = document.getElementById('rsiChart').getContext('2d');
                rsiChart = new Chart(rsiCtx, {
                    type: 'line',
                    data: {
                        labels: data.dates,
                        datasets: [{
                            label: 'RSI 14',
                            data: data.rsi,
                            borderColor: 'rgb(153, 102, 255)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'RSI Göstergesi'
                            }
                        },
                        scales: {
                            y: {
                                min: 0,
                                max: 100,
                                grid: {
                                    color: function(context) {
                                        if (context.tick.value === 30 || context.tick.value === 70) {
                                            return 'rgba(255, 0, 0, 0.2)';
                                        }
                                        return 'rgba(0, 0, 0, 0.1)';
                                    }
                                }
                            }
                        }
                    }
                });
                
                // MACD Grafiği
                const macdCtx = document.getElementById('macdChart').getContext('2d');
                macdChart = new Chart(macdCtx, {
                    type: 'line',
                    data: {
                        labels: data.dates,
                        datasets: [
                            {
                                label: 'MACD',
                                data: data.macd,
                                borderColor: 'rgb(255, 159, 64)',
                                tension: 0.1
                            },
                            {
                                label: 'Sinyal',
                                data: data.macd_signal,
                                borderColor: 'rgb(201, 203, 207)',
                                tension: 0.1
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'MACD Göstergesi'
                            }
                        }
                    }
                });
                
                // Hata mesajını temizle
                errorMessage.style.display = 'none';
            })
            .catch(error => {
                console.error('Error:', error);
                errorMessage.textContent = `Hata: Veriler yüklenirken bir sorun oluştu. ${error.message}`;
                errorMessage.style.display = 'block';
                
                // Grafik containerlarını temizle
                containers.forEach(container => {
                    container.innerHTML = `<canvas></canvas>`;
                });
            });
    }
}); 