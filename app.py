from flask import Flask, render_template, jsonify, request, session, redirect, url_for, flash
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # GUI olmayan backend kullan
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import json
import os
import joblib
import logging
import traceback
from functools import wraps
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Güvenli bir secret key kullanın
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Paket tipleri
PACKAGE_TYPES = {
    'free': {
        'name': 'Ücretsiz Paket',
        'price': 0,
        'duration_days': 30,
        'features': ['Günlük 5 hisse analizi', 'Temel analiz özellikleri', 'E-posta bildirimleri']
    },
    'starter': {
        'name': 'Başlangıç',
        'price': 20,
        'duration_days': 7,
        'features': ['Hisse senedi analizi', 'Teknik gösterge izleme', 'Temel fiyat tahminleri']
    },
    'professional': {
        'name': 'Profesyonel',
        'price': 50,
        'duration_days': 7,
        'features': ['Hisse analizi', 'Hedef bazlı portföy', 'Yapay zeka tahminleri', 'Altın/Gümüş analizi']
    },
    'premium': {
        'name': 'Premium',
        'price': 100,
        'duration_days': 30,
        'features': ['Tüm özellikler', 'Gelişmiş yapay zeka tahminleri', 'Öncelikli destek']
    }
}

# Kullanıcı modeli
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    subscriptions = db.relationship('Subscription', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Abonelik modeli
class Subscription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    package_type = db.Column(db.String(20), nullable=False)
    start_date = db.Column(db.DateTime, default=datetime.utcnow)
    end_date = db.Column(db.DateTime, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    auto_renew = db.Column(db.Boolean, default=False)

    def get_remaining_days(self):
        if not self.is_active:
            return 0
        remaining = (self.end_date - datetime.utcnow()).days
        return max(0, remaining)

    def get_package_info(self):
        return PACKAGE_TYPES.get(self.package_type, PACKAGE_TYPES['free'])

# Veritabanını oluştur
with app.app_context():
    db.create_all()

# Login route'u
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember')
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            if remember:
                session.permanent = True
            flash('Başarıyla giriş yaptınız!', 'success')
            return redirect(url_for('index'))
        else:
            flash('E-posta veya şifre hatalı!', 'danger')
    
    return render_template('login.html')

# Register route'u
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Şifreler eşleşmiyor!', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Bu e-posta adresi zaten kullanılıyor!', 'danger')
            return redirect(url_for('register'))
        
        user = User(name=name, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Kayıt başarılı! Giriş yapabilirsiniz.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

# Logout route'u
@app.route('/logout')
def logout():
    session.clear()
    flash('Başarıyla çıkış yaptınız!', 'success')
    return redirect(url_for('login'))

# Kullanıcı girişi kontrolü için decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Bu sayfayı görüntülemek için giriş yapmalısınız!', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Ana sayfa route'u
@app.route('/')
def index():
    user_name = None
    if 'user_name' in session:
        user_name = session.get('user_name')
    return render_template('index.html', user_name=user_name)

# BIST şirketleri listesi
BIST_COMPANIES = {
    'ASELS': 'ASELSAN',
    'THYAO': 'Türk Hava Yolları',
    'GARAN': 'Garanti Bankası',
    'AKBNK': 'Akbank',
    'ISCTR': 'İş Bankası',
    'KCHOL': 'Koç Holding',
    'SISE': 'Şişe Cam',
    'EREGL': 'Ereğli Demir Çelik',
    'TUPRS': 'Tüpraş',
    'SAHOL': 'Sabancı Holding',
    'ARCLK': 'Arçelik',
    'BIMAS': 'BİM',
    'FROTO': 'Ford Otosan',
    'TOASO': 'Tofaş',
    'VAKBN': 'Vakıfbank',
    'YKBNK': 'Yapı Kredi',
    'PETKM': 'Petkim',
    'TAVHL': 'TAV Havalimanları',
    'ENJSA': 'Enerjisa',
    'SASA': 'Sasa Polyester',
    'TCELL': 'Turkcell',
    'HALKB': 'Halkbank',
    'PGSUS': 'Pegasus',
    'SOKM': 'Sok Marketler',
    'MGROS': 'Migros',
    'KRDMD': 'Kardemir',
    'TTKOM': 'Türk Telekom',
    'ULKER': 'Ülker',
    'CCOLA': 'Coca-Cola İçecek',
    'DOAS': 'Doğuş Otomotiv',
    'EKGYO': 'Emlak Konut GYO',
    'KOZAL': 'Koza Altın',
    'KOZAA': 'Koza Madencilik',
    'OYAKC': 'OYAK Çimento',
    'ALARK': 'Alarko Holding',
    'ALBRK': 'Albaraka Türk',
    'AKSEN': 'Aksa Enerji',
    'AKSA': 'Aksa Akrilik',
    'AEFES': 'Anadolu Efes',
    'ANACM': 'Anadolu Cam',
    'AYGAZ': 'Aygaz',
    'BAGFS': 'Bagfaş',
    'BRISA': 'Brisa',
    'CIMSA': 'Çimsa',
    'DOHOL': 'Doğan Holding',
    'EGEEN': 'Ege Endüstri',
    'ENKAI': 'ENKA İnşaat',
    'ERBOS': 'Erbosan',
    'GESAN': 'Gedik Enerji',
    'GLYHO': 'Global Yatırım',
    'GUBRF': 'Gübre Fabrik.',
    'IPEKE': 'İpek Enerji',
    'IHLGM': 'İhlas Gayrimenkul',
    'ISDMR': 'İskenderun Demir',
    'ISFIN': 'İş Fin.Kir.',
    'ISGYO': 'İş GYO',
    'KAREL': 'Karel Elektronik',
    'KARSN': 'Karsan Otomotiv',
    'KONTR': 'Kontrolmatik',
    'KORDS': 'Kordsa Teknik',
    'LOGO': 'Logo Yazılım',
    'MAVI': 'Mavi Giyim',
    'METRO': 'Metro Holding',
    'NETAS': 'Netaş Telekom.',
    'NUHCM': 'Nuh Çimento',
    'ODAS': 'Odaş Elektrik',
    'OTKAR': 'Otokar',
    'PRKME': 'Park Elek.Madencilik',
    'QUAGR': 'QUA Granite',
    'SAFKR': 'Şafak Küresel',
    'SELEC': 'Selçuk Ecza',
    'SMRTG': 'Smart Güneş',
    'SNGYO': 'Sinpaş GYO',
    'TATGD': 'Tat Gıda',
    'TKFEN': 'Tekfen Holding',
    'TKNSA': 'Teknosa',
    'TMSN': 'Tümosan Motor',
    'TRGYO': 'Torunlar GYO',
    'TSKB': 'T.S.K.B.',
    'TTRAK': 'Türk Traktör',
    'TURSG': 'Türkiye Sigorta',
    'USAK': 'Uşak Seramik',
    'VAKKO': 'Vakko Tekstil',
    'VESBE': 'Vestel Beyaz Eşya',
    'VESTL': 'Vestel',
    'YATAS': 'Yataş',
    'ZOREN': 'Zorlu Enerji',
    'GWIND': 'Galata Wind',
    'GENIL': 'Gen İlaç',
    'ISMEN': 'İş Yatırım',
    'MPARK': 'MLP Sağlık',
    'SKBNK': 'Şekerbank',
    'AKFGY': 'Akfen GYO',
    'AKGRT': 'Aksigorta',
    'ALCTL': 'Alcatel Lucent',
    'ANELE': 'Anel Elektrik',
    'ARSAN': 'Arsan Tekstil',
    'ASTOR': 'Astor Enerji',
    'AYDEM': 'Aydem Enerji',
    'BASGZ': 'Başkent Doğalgaz',
    'BERA': 'Bera Holding',
    'BIOEN': 'Biotrend Enerji',
    'BRYAT': 'Borusan Yat.Paz.',
    'BUCIM': 'Bursa Çimento',
    'CANTE': 'Çan2 Termik',
    'CLEBI': 'Çelebi Hava Servisi',
    'DEVA': 'Deva Holding',
    'DGKLB': 'Doğtaş Kelebek',
    'ECZYT': 'Eczacıbaşı Yatırım',
    'GOLD': 'Altın',
    'SILVER': 'Gümüş'
}

def teknik_gostergeleri_ekle(df):
    df = df.copy()
    
    # Günlük getiri
    df['Return'] = df['Close'].pct_change()
    
    # Hacim değişimi
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Fiyat değişim hızı
    df['Price_Velocity'] = df['Close'].diff() / df['Close'].shift(1)
    
    # Fiyat momentumu (5 ve 10 günlük)
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    
    # Hareketli ortalamalar (5, 10, 20, 50 günlük)
    for window in [5, 10, 20, 50]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False, min_periods=1).mean()
    
    # Bollinger Bantları
    df['BB_middle'] = df['SMA_20']
    df['BB_std'] = df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    
    # RSI (14 ve 7 günlük)
    for period in [7, 14]:
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Stochastic Oscillator
    low_min = df['Low'].rolling(window=14, min_periods=1).min()
    high_max = df['High'].rolling(window=14, min_periods=1).max()
    df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['%D'] = df['%K'].rolling(window=3, min_periods=1).mean()
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
    
    # Hedef değişkeni: 3 günlük fiyat hareketi
    df['Future_Return'] = df['Close'].shift(-3) / df['Close'] - 1
    df['Target'] = (df['Future_Return'] > 0).astype(int)
    
    # NaN değerleri 0 ile doldur
    df = df.fillna(0)
    
    return df

def prepare_lstm_data(df, sequence_length=10):
    # Veriyi normalize et
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(1 if df['Close'].iloc[i + sequence_length].item() > df['Close'].iloc[i + sequence_length - 1].item() else 0)
    
    return np.array(X), np.array(y), scaler

def create_lstm_model(sequence_length):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_hybrid_model(df):
    try:
        # LSTM için veriyi hazırla
        sequence_length = 10
        X_lstm, y_lstm, scaler = prepare_lstm_data(df, sequence_length)
        
        # LSTM modelini oluştur ve eğit
        lstm_model = create_lstm_model(sequence_length)
        lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)
        
        # LSTM tahminlerini al
        latest_sequence = scaler.transform(df[['Close']].iloc[-sequence_length:].values)
        lstm_prediction = lstm_model.predict(np.array([latest_sequence]))[0][0]
        lstm_probability = [1 - lstm_prediction, lstm_prediction]
        
        # XGBoost için veriyi hazırla
        X = df[['SMA_10', 'SMA_50', 'RSI_14', 'MACD', 'MACD_signal', 'Momentum_10']]
        y = df['Target']
        
        # XGBoost modelini oluştur ve eğit
        xgb_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8
        )
        xgb_model.fit(X, y)
        
        # XGBoost tahminlerini al
        latest_features = X.iloc[-1:].values
        xgb_probability = xgb_model.predict_proba(latest_features)[0]
        
        # Hibrit tahmin: LSTM ve XGBoost tahminlerinin ağırlıklı ortalaması
        lstm_weight = 0.6  # LSTM'e daha fazla ağırlık ver
        xgb_weight = 0.4
        
        final_probability = (np.array(lstm_probability) * lstm_weight + np.array(xgb_probability) * xgb_weight)
        final_prediction = 1 if final_probability[1] > 0.5 else 0
        
        # Özellik önemlerini hesapla
        feature_importance = dict(zip(X.columns, xgb_model.feature_importances_))
        
        return final_prediction, final_probability, feature_importance
    except Exception as e:
        print(f"Hibrit model eğitimi hatası: {str(e)}")
        raise

def analyze_model_performance(df):
    try:
        # Özellikler ve hedef
        features = [
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
            'RSI_7', 'RSI_14',
            'MACD', 'MACD_signal', 'MACD_hist',
            'Momentum_5', 'Momentum_10',
            'BB_upper', 'BB_lower',
            '%K', '%D',
            'ATR',
            'Volume_Change',
            'Price_Velocity'
        ]
        
        X = df[features]
        y = df['Target']
        
        # Eğitim ve test verisi ayır (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
        
        # Modeli oluştur ve eğit
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            max_depth=5,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
        
        # Early stopping ile eğitim
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Tahminler
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Olasılık tahminleri
        y_test_proba = model.predict_proba(X_test)
        
        # Performans metrikleri
        train_accuracy = float(accuracy_score(y_train, y_train_pred))
        test_accuracy = float(accuracy_score(y_test, y_test_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Sınıflandırma raporu
        report = classification_report(y_test, y_test_pred, output_dict=True)
        
        # Özellik önemleri
        feature_importance = {str(k): float(v) for k, v in zip(X.columns, model.feature_importances_)}
        
        # Grafikleri oluştur
        plt.figure(figsize=(15, 10))
        
        # 1. Confusion Matrix
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Tahmin')
        plt.ylabel('Gerçek')
        
        # 2. Özellik önemleri
        plt.subplot(2, 2, 2)
        importance_items = sorted(feature_importance.items(), key=lambda x: x[1])
        features, values = zip(*importance_items)
        plt.barh(features, values)
        plt.title('Özellik Önemleri')
        
        # 3. Doğruluk karşılaştırması
        plt.subplot(2, 2, 3)
        plt.bar(['Eğitim', 'Test'], [train_accuracy, test_accuracy])
        plt.title('Model Doğruluğu')
        plt.ylim(0, 1)
        for i, v in enumerate([train_accuracy, test_accuracy]):
            plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
        
        # 4. Tahmin dağılımı
        plt.subplot(2, 2, 4)
        plt.hist(y_test_proba[:, 1], bins=20)
        plt.title('Tahmin Olasılıkları Dağılımı')
        plt.xlabel('Yükseliş Olasılığı')
        plt.ylabel('Frekans')
        
        plt.tight_layout()
        
        # Grafiği kaydet
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close('all')
        
        performance_data = {
            'train_accuracy': f"{train_accuracy:.4f}",
            'test_accuracy': f"{test_accuracy:.4f}",
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'feature_importance': feature_importance,
            'plot': base64.b64encode(img.getvalue()).decode('utf-8')
        }
        
        return performance_data
        
    except Exception as e:
        print(f"Analiz hatası: {str(e)}")
        raise

@app.route('/stock_analysis')
def stock_analysis():
    user_name = None
    if 'user_name' in session:
        user_name = session.get('user_name')
    return render_template('stock_analysis.html', user_name=user_name)

@app.route('/get_stock_data/<company>')
def get_stock_data(company):
    try:
        print(f"\n{'='*50}")
        print(f"Veri indiriliyor... Şirket/Varlık: {company}")
        
        # Veriyi indir
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # Son 3 aylık veri
        
        # Altın ve gümüş için özel durum
        if company == 'GOLD':
            ticker = 'GC=F'  # Altın futures kodu (Yahoo Finance)
            print(f"Altın verisi indiriliyor... Ticker: {ticker}")
            
            # Altın için direkt veri indir, info kullanma
            df = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True,
                rounding=True
            )
            
            if not df.empty:
                current_price = round(float(df['Close'].iloc[-1]), 2)
                print(f"Güncel altın fiyatı: {current_price:.2f} USD/Ons")
            else:
                error_msg = "Altın verisi bulunamadı!"
                print(error_msg)
                return jsonify({'error': error_msg}), 404
                
        elif company == 'SILVER':
            ticker = 'SI=F'  # Gümüş futures kodu (Yahoo Finance)
            print(f"Gümüş verisi indiriliyor... Ticker: {ticker}")
            
            # Gümüş için direkt veri indir, info kullanma
            df = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True,
                rounding=True
            )
            
            if not df.empty:
                current_price = round(float(df['Close'].iloc[-1]), 2)
                print(f"Güncel gümüş fiyatı: {current_price:.2f} USD/Ons")
            else:
                error_msg = "Gümüş verisi bulunamadı!"
                print(error_msg)
                return jsonify({'error': error_msg}), 404
                
        else:
            # Normal hisseler için standart işlem
            ticker = f"{company}.IS"  # BIST hisseleri için normal format
            print(f"Tarih aralığı: {start_date} - {end_date}")
            print(f"Ticker: {ticker}")
            
            # Güncel fiyat bilgisini al
            stock = yf.Ticker(ticker)
            current_info = stock.info
            current_price = float(current_info.get('regularMarketPrice', 0))
            
            if current_price == 0:  # Eğer güncel fiyat alınamazsa son kapanışı kullan
                print("Güncel fiyat alınamadı, son kapanış fiyatı kullanılacak")
                # yfinance parametrelerini ayarla
                df = yf.download(
                    ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    progress=False,
                    auto_adjust=True,
                    rounding=True
                )
                if not df.empty:
                    current_price = round(float(df['Close'].iloc[-1]), 2)
            else:
                print(f"Güncel fiyat başarıyla alındı: {current_price}")
                df = yf.download(
                    ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    progress=False,
                    auto_adjust=True,
                    rounding=True
                )
        
        print(f"Güncel fiyat: {current_price:.2f} {'USD/Ons' if company in ['GOLD', 'SILVER'] else 'TL'}")
        
        if df.empty:
            print("İlk denemede veri bulunamadı, son 30 günlük veri deneniyor...")
            # Son 30 günlük veriyi dene
            start_date = end_date - timedelta(days=30)
            df = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True,
                rounding=True
            )
            
            if df.empty:
                error_msg = "Veri bulunamadı!"
                print(error_msg)
                return jsonify({'error': error_msg}), 404
        
        print("Teknik göstergeler ekleniyor...")
        # Teknik göstergeleri ekle
        df = teknik_gostergeleri_ekle(df)
        print(f"Teknik göstergeler eklendi. Veri boyutu: {len(df)}")
        print("Eklenen özellikler:", df.columns.tolist())
        
        if len(df) < 20:  # Minimum veri kontrolü
            error_msg = 'Yetersiz veri miktarı. En az 20 günlük veri gerekli.'
            print(error_msg)
            return jsonify({'error': error_msg}), 400
        
        print("Model eğitimi başlıyor...")
        # Daha az özellik kullan
        features = [
            'SMA_10', 'SMA_50',
            'RSI_14',
            'MACD', 'MACD_signal',
            'Momentum_10',
            'Volume_Change',
            'Price_Velocity'
        ]
        
        # Eksik sütunları kontrol et
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            error_msg = f"Eksik özellikler: {missing_features}"
            print(error_msg)
            return jsonify({'error': error_msg}), 500
        
        X = df[features]
        y = df['Target']
        
        print("Veri bölünüyor...")
        # Eğitim ve test verisi ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
        
        print("Model oluşturuluyor...")
        # Model eğitimi - daha basit parametreler
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        print("Model eğitiliyor...")
        model.fit(X_train, y_train)
        
        print("Model performansı hesaplanıyor...")
        # Model performansı
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        
        print(f"Eğitim doğruluğu: {train_accuracy:.4f}")
        print(f"Test doğruluğu: {test_accuracy:.4f}")
        
        # Özellik önemleri
        feature_importance = dict(zip(features, model.feature_importances_))
        print("Özellik önemleri:", feature_importance)
        
        # Son fiyat ve tahmin
        latest_features = X.iloc[-1:].values
        prediction = model.predict_proba(latest_features)[0]
        
        # Tahmin olasılığını -3% ile +3% arasında bir değişime dönüştür (daha gerçekçi)
        raw_change = (prediction[1] - 0.5) * 2  # -1 ile 1 arasında değer
        capped_change = np.clip(raw_change * 0.03, -0.03, 0.03)  # ±3% ile sınırla (daha gerçekçi)
        predicted_price = round(current_price * (1 + capped_change), 2)
        
        print(f"Mevcut fiyat: {current_price:.2f} {'USD/Ons' if company in ['GOLD', 'SILVER'] else 'TL'}")
        print(f"Tahmini fiyat: {predicted_price:.2f} {'USD/Ons' if company in ['GOLD', 'SILVER'] else 'TL'}")
        print(f"Tahmini değişim: {capped_change:.2%}")
        
        # Tarihleri string formatına çevir
        dates = df.index.strftime('%Y-%m-%d').tolist()
        
        print("Sonuçlar hazırlanıyor...")
        result = {
            'dates': dates,
            'close': [round(float(x), 2) for x in df['Close'].values],  # numpy değerlerini önce float'a çevir
            'sma_10': [round(float(x), 2) for x in df['SMA_10'].values],
            'sma_50': [round(float(x), 2) for x in df['SMA_50'].values],
            'rsi': [round(float(x), 2) for x in df['RSI_14'].values],
            'macd': [round(float(x), 3) for x in df['MACD'].values],
            'macd_signal': [round(float(x), 3) for x in df['MACD_signal'].values],
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'feature_importance': {k: float(v) for k, v in feature_importance.items()},
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': float(capped_change),
            'currency': 'USD' if company in ['GOLD', 'SILVER'] else 'TRY',
            'unit': 'Ons' if company in ['GOLD', 'SILVER'] else ''
        }
        
        print("İşlem başarılı!")
        print('='*50 + '\n')
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_msg = f"Hata: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        print('='*50 + '\n')
        return jsonify({'error': error_msg}), 500

@app.route('/get_companies')
def get_companies():
    # Sadece şirket kodlarını döndür
    return jsonify(list(BIST_COMPANIES.keys()))

@app.route('/analyze_performance')
def get_performance():
    try:
        # ASELSAN verilerini indir
        data = yf.download("ASELS.IS", 
                          start="2023-01-01",
                          end=datetime.now().strftime('%Y-%m-%d'),
                          progress=False)
        
        if data.empty:
            return jsonify({'error': 'Veri bulunamadı'}), 404
            
        # Teknik göstergeleri ekle
        data = teknik_gostergeleri_ekle(data)
        
        # Performans analizi
        performance = analyze_model_performance(data)
        
        return jsonify(performance)
    except Exception as e:
        print(f"Performans analizi hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/goals')
def goals():
    user_name = None
    if 'user_name' in session:
        user_name = session.get('user_name')
    return render_template('goals.html', user_name=user_name)

def calculate_portfolio_metrics(stock_data, risk_level):
    """Hisse senedi için risk ve getiri metriklerini hesapla"""
    try:
        if stock_data.empty:
            return {
                'annual_return': 0,
                'volatility': 0,
                'risk_score': 0,
                'ai_prediction': 0
            }
            
        # Teknik göstergeleri ekle
        df = teknik_gostergeleri_ekle(stock_data)
        
        # XGBoost modeli için özellikler
        features = [
            'SMA_10', 'SMA_50',
            'RSI_14',
            'MACD', 'MACD_signal',
            'Momentum_10',
            'Volume_Change',
            'Price_Velocity'
        ]
        
        # Model eğitimi
        X = df[features]
        y = df['Target']
        
        if len(X) < 30:  # Minimum veri kontrolü
            return {
                'annual_return': 0,
                'volatility': 0,
                'risk_score': 0,
                'ai_prediction': 0
            }
        
        # Eğitim ve test verisi ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
        
        # Model
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100
        )
        
        model.fit(X_train, y_train)
        
        # Son veriler için tahmin
        latest_features = X.iloc[-1:].values
        ai_prediction = float(model.predict_proba(latest_features)[0][1])  # Yükseliş olasılığı
        
        # Getiri ve volatilite hesapla
        returns = df['Close'].pct_change().dropna()
        annual_return = float(returns.mean() * 252)
        volatility = float(returns.std() * np.sqrt(252))
        
        # Risk seviyesine göre ağırlıkları ayarla
        risk_weights = {
            'low': 0.7,
            'medium': 1.0,
            'high': 1.3
        }
        
        risk_weight = risk_weights[risk_level]
        
        # AI tahminini getiriye dahil et
        adjusted_return = (abs(annual_return) * 0.7 + ai_prediction * 0.3) * risk_weight
        
        # Risk skorunu hesapla (0-100 arası)
        risk_score = min(100, volatility * 100 * risk_weight)
        
        return {
            'annual_return': adjusted_return,
            'volatility': volatility,
            'risk_score': risk_score,
            'ai_prediction': ai_prediction
        }
    except Exception as e:
        print(f"Metrik hesaplama hatası: {str(e)}")
        return {
            'annual_return': 0,
            'volatility': 0,
            'risk_score': 0,
            'ai_prediction': 0
        }

@app.route('/generate_portfolio', methods=['POST'])
def generate_portfolio():
    try:
        data = request.get_json()
        capital = float(data['capital'])
        target = float(data['target'])
        risk_level = data['riskLevel']
        
        print(f"Portföy oluşturuluyor... Sermaye: {capital}, Hedef: {target}, Risk: {risk_level}")
        
        # Minimum sermaye kontrolü
        if capital < 1000:
            return jsonify({'error': 'Minimum sermaye 1.000 TL olmalıdır'}), 400
            
        # Hedef kontrolü
        if target <= capital:
            return jsonify({'error': 'Hedef tutar sermayeden büyük olmalıdır'}), 400
        
        # Risk seviyesine göre hisse seçimi
        stock_weights = {
            'low': [
                'THYAO', 'GARAN', 'AKBNK', 'EREGL', 'BIMAS',  # Ana endeks
                'TUPRS', 'SAHOL', 'KCHOL', 'ISCTR', 'VAKBN',  # Büyük bankalar ve holdingler
                'TTKOM', 'TCELL', 'AEFES', 'ULKER', 'CCOLA',  # Defansif sektörler
                'GOLD'  # Altın (düşük risk grubuna)
            ],
            'medium': [
                'ASELS', 'SISE', 'ARCLK', 'FROTO', 'TOASO',   # Sanayi ve otomotiv
                'TKFEN', 'ENKAI', 'OYAKC', 'KORDS', 'OTKAR',  # İnşaat ve sanayi
                'VESTL', 'VESBE', 'LOGO', 'KAREL', 'MAVI',    # Teknoloji ve perakende
                'GOLD', 'SILVER'  # Altın ve gümüş (orta risk grubuna)
            ],
            'high': [
                'SASA', 'PGSUS', 'TAVHL', 'KOZAL', 'KOZAA',   # Yüksek beta
                'GESAN', 'KONTR', 'ODAS', 'GWIND', 'SMRTG',   # Enerji ve yenilenebilir
                'IPEKE', 'PRKME', 'QUAGR', 'BIOEN', 'ASTOR',  # Madencilik ve enerji
                'SILVER'  # Gümüş (yüksek risk grubuna)
            ]
        }
        
        selected_stocks = stock_weights[risk_level]
        portfolio = []
        total_score = 0
        stock_metrics = {}
        
        print("Seçilen hisseler:", selected_stocks)
        
        # Her hisse için metrikleri hesapla
        for stock in selected_stocks:
            print(f"\nHisse analizi: {stock}")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)  # Son 3 aylık veri
            
            try:
                # Hisse verilerini indir
                ticker = ""
                is_precious_metal = False
                
                if stock == 'GOLD':
                    ticker = 'GC=F'  # Altın futures kodu
                    is_precious_metal = True
                    print(f"Altın verisi indiriliyor... Ticker: {ticker}")
                elif stock == 'SILVER':
                    ticker = 'SI=F'  # Gümüş futures kodu
                    is_precious_metal = True
                    print(f"Gümüş verisi indiriliyor... Ticker: {ticker}")
                else:
                    ticker = f"{stock}.IS"
                
                df = yf.download(
                    ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    progress=False
                )
                
                print(f"{stock} veri boyutu: {len(df)}")
                
                # Altın ve gümüş için USD/TRY kurunu kullanarak TL'ye çevir
                if is_precious_metal and not df.empty:
                    try:
                        # USD/TRY kuru için veri indir
                        usd_try = yf.download(
                            'USDTRY=X',
                            start=start_date.strftime('%Y-%m-%d'),
                            end=end_date.strftime('%Y-%m-%d'),
                            progress=False
                        )
                        
                        if not usd_try.empty:
                            # Son kuru al
                            usd_try_rate = usd_try['Close'].iloc[-1]
                            print(f"Güncel USD/TRY kuru: {usd_try_rate:.4f}")
                            
                            # Fiyatları TL'ye çevir
                            df['Close_TRY'] = df['Close'] * usd_try_rate
                            df['Open_TRY'] = df['Open'] * usd_try_rate
                            df['High_TRY'] = df['High'] * usd_try_rate
                            df['Low_TRY'] = df['Low'] * usd_try_rate
                            
                            # TRY bazlı sütunları kullan
                            df['Close_Original'] = df['Close'].copy()
                            df['Open_Original'] = df['Open'].copy()
                            df['High_Original'] = df['High'].copy()
                            df['Low_Original'] = df['Low'].copy()
                            
                            df['Close'] = df['Close_TRY']
                            df['Open'] = df['Open_TRY']
                            df['High'] = df['High_TRY']
                            df['Low'] = df['Low_TRY']
                            
                            print(f"{stock} fiyatları TL'ye çevrildi.")
                    except Exception as e:
                        print(f"USD/TRY dönüşümü yapılamadı: {str(e)}")
                
                if not df.empty and len(df) > 30:  # En az 30 günlük veri olmalı
                    metrics = calculate_portfolio_metrics(df, risk_level)
                    if metrics['annual_return'] > 0:  # Sadece pozitif getirisi olan hisseleri ekle
                        stock_metrics[stock] = metrics
                        # AI tahminini de skora dahil et
                        total_score += metrics['annual_return'] * (1 + metrics['ai_prediction'])
                        print(f"{stock} metrikleri hesaplandı: {metrics}")
                else:
                    print(f"{stock} için yeterli veri bulunamadı")
            except Exception as e:
                print(f"{stock} veri indirme hatası: {str(e)}")
                continue
        
        if not stock_metrics:
            return jsonify({'error': 'Hiçbir hisse için yeterli veri bulunamadı'}), 500
        
        print("\nPortföy dağılımı hesaplanıyor...")
        # Portföy dağılımını hesapla
        min_investment = 100  # Minimum yatırım tutarı
        portfolio = []
        valid_stocks = []
        
        # İlk aşama: Geçerli hisseleri belirle
        for stock in selected_stocks:
            if stock in stock_metrics:
                metrics = stock_metrics[stock]
                # AI tahminini ağırlık hesaplamada kullan
                weight = (metrics['annual_return'] * (1 + metrics['ai_prediction'])) / total_score if total_score > 0 else 1.0 / len(stock_metrics)
                amount = round(capital * weight, 2)
                
                # Beklenen getiri hesaplama
                base_return = metrics['annual_return'] * 100  # Yıllık getiri
                ai_confidence = metrics['ai_prediction'] * 100  # AI tahmini
                
                # AI tahmini ve tarihsel getiriyi birleştir
                expected_return = base_return * 0.7 + ai_confidence * 0.3
                
                # Risk seviyesine göre ayarla
                risk_multipliers = {'low': 0.8, 'medium': 1.0, 'high': 1.2}
                expected_return *= risk_multipliers[risk_level]
                
                # Sadece pozitif beklenen getirisi olan hisseleri ekle
                if expected_return > 0:
                    valid_stocks.append({
                        'code': stock,
                        'name': BIST_COMPANIES[stock],
                        'weight': weight,
                        'expectedReturn': round(expected_return, 1),
                        'risk': 'Düşük' if metrics['risk_score'] < 33 else 'Orta' if metrics['risk_score'] < 66 else 'Yüksek',
                        'aiConfidence': round(metrics['ai_prediction'] * 100, 1)
                    })
        
        if not valid_stocks:
            return jsonify({'error': 'Uygun hisse bulunamadı'}), 500
        
        # Hisseleri beklenen getirisine göre sırala
        valid_stocks.sort(key=lambda x: x['expectedReturn'], reverse=True)
        
        # En iyi performans gösteren maksimum 8 hisseyi seç
        valid_stocks = valid_stocks[:8]
        
        # Minimum yatırım tutarını kontrol et
        total_min_investment = len(valid_stocks) * min_investment
        if total_min_investment > capital:
            # Eğer sermaye yetmiyorsa, hisse sayısını azalt
            max_stocks = int(capital // min_investment)
            valid_stocks = valid_stocks[:max_stocks]
        
        # Kalan sermayeyi hesapla
        remaining_capital = capital - (len(valid_stocks) * min_investment)
        
        # Ağırlıkları yeniden hesapla
        total_weight = sum(stock['expectedReturn'] for stock in valid_stocks)
        
        # Her hisse için final dağılımı hesapla
        for stock in valid_stocks:
            # Minimum tutarı garantile
            base_amount = min_investment
            
            # Kalan sermayeyi performansa göre dağıt
            if total_weight > 0:
                additional_amount = remaining_capital * (stock['expectedReturn'] / total_weight)
            else:
                additional_amount = remaining_capital / len(valid_stocks)
            
            total_amount = round(base_amount + additional_amount, 2)
            percentage = round((total_amount / capital) * 100, 1)
            
            portfolio.append({
                'code': stock['code'],
                'name': stock['name'],
                'amount': total_amount,
                'percentage': percentage,
                'expectedReturn': stock['expectedReturn'],
                'risk': stock['risk'],
                'aiConfidence': stock['aiConfidence']
            })
            print(f"{stock['code']} portföy bilgileri eklendi: {total_amount} TL ({percentage}%)")
        
        if not portfolio:
            return jsonify({'error': 'Portföy oluşturulamadı'}), 500
        
        # Hedef süresini hesapla (ay olarak)
        weighted_return = sum(stock['expectedReturn'] * (stock['percentage'] / 100) for stock in portfolio)
        monthly_return = weighted_return / 12
        target_ratio = target / capital
        
        # Hedef süre hesaplama
        if monthly_return <= 0:
            estimated_months = 36  # Varsayılan maksimum süre
        else:
            try:
                estimated_months = round(np.log(target_ratio) / np.log(1 + monthly_return/100))
                estimated_months = min(max(estimated_months, 1), 36)  # 1-36 ay arası sınırla
            except:
                estimated_months = 36
        
        print(f"\nHesaplama tamamlandı. Tahmini süre: {estimated_months} ay")
        
        return jsonify({
            'portfolio': portfolio,
            'estimatedMonths': estimated_months
        })
        
    except Exception as e:
        import traceback
        error_msg = f"Portföy oluşturma hatası: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

# Abonelik API endpoint'leri
@app.route('/api/subscription', methods=['GET'])
@login_required
def get_subscription():
    user = User.query.get(session['user_id'])
    current_subscription = Subscription.query.filter_by(
        user_id=user.id,
        is_active=True
    ).first()

    if not current_subscription:
        return jsonify({
            'currentPackage': {
                'name': 'Ücretsiz Paket',
                'remainingTime': '30 gün',
                'features': PACKAGE_TYPES['free']['features']
            },
            'history': []
        })

    package_info = current_subscription.get_package_info()
    remaining_days = current_subscription.get_remaining_days()

    # Süresi dolan abonelikleri kontrol et
    if remaining_days == 0:
        current_subscription.is_active = False
        db.session.commit()
        return jsonify({
            'currentPackage': {
                'name': 'Ücretsiz Paket',
                'remainingTime': '30 gün',
                'features': PACKAGE_TYPES['free']['features']
            },
            'history': get_subscription_history(user.id)
        })

    # Abonelik geçmişini al
    history = get_subscription_history(user.id)

    return jsonify({
        'currentPackage': {
            'name': package_info['name'],
            'remainingTime': f'{remaining_days} gün',
            'features': package_info['features']
        },
        'history': history
    })

@app.route('/api/subscription/cancel', methods=['POST'])
@login_required
def cancel_subscription():
    user = User.query.get(session['user_id'])
    current_subscription = Subscription.query.filter_by(
        user_id=user.id,
        is_active=True
    ).first()

    if current_subscription:
        current_subscription.is_active = False
        current_subscription.auto_renew = False
        db.session.commit()

    return jsonify({'status': 'success'})

def get_subscription_history(user_id):
    subscriptions = Subscription.query.filter_by(user_id=user_id).order_by(Subscription.start_date.desc()).all()
    history = []
    
    for sub in subscriptions:
        package_info = sub.get_package_info()
        history.append({
            'name': package_info['name'],
            'startDate': sub.start_date.strftime('%d.%m.%Y'),
            'endDate': sub.end_date.strftime('%d.%m.%Y'),
            'status': 'Aktif' if sub.is_active else 'Sona Erdi'
        })
    
    return history

# Hesap ayarları sayfası
@app.route('/accounts_settings')
@login_required
def accounts_settings():
    return render_template('accounts_settings.html', user_name=session.get('user_name'))

# Paket satın alma sayfası
@app.route('/purchase')
@login_required
def purchase():
    package_type = request.args.get('package')
    if not package_type or package_type not in PACKAGE_TYPES:
        flash('Geçersiz paket seçimi!', 'danger')
        return redirect(url_for('index'))
    return render_template('purchase.html', user_name=session.get('user_name'))

# Paket bilgilerini getiren API endpoint'i
@app.route('/api/packages/<package_type>')
@login_required
def get_package_info(package_type):
    if package_type not in PACKAGE_TYPES:
        return jsonify({'error': 'Geçersiz paket tipi'}), 400
    return jsonify(PACKAGE_TYPES[package_type])

# Zamanlayıcıyı başlat
scheduler = BackgroundScheduler()

# Abonelik sürelerini güncelleyen işlev
@scheduler.scheduled_job('interval', days=1)
def update_subscription_days():
    with app.app_context():
        subscriptions = Subscription.query.filter_by(is_active=True).all()
        for subscription in subscriptions:
            if subscription.get_remaining_days() <= 0:
                subscription.is_active = False
        db.session.commit()

scheduler.start()

# Satın alma işlemini gerçekleştiren API endpoint'i
@app.route('/api/purchase', methods=['POST'])
@login_required
def purchase_package():
    data = request.get_json()
    package_type = data.get('package_type')
    payment_method = data.get('payment_method')

    if not package_type or package_type not in PACKAGE_TYPES:
        return jsonify({'success': False, 'message': 'Geçersiz paket tipi'})

    if not payment_method:
        return jsonify({'success': False, 'message': 'Ödeme yöntemi seçilmedi'})

    user = User.query.get(session['user_id'])
    
    # Mevcut aktif aboneliği kontrol et
    current_subscription = Subscription.query.filter_by(
        user_id=user.id,
        is_active=True
    ).first()

    if current_subscription:
        # Mevcut aboneliği pasif yap
        current_subscription.is_active = False
        current_subscription.auto_renew = False

    # Yeni abonelik oluştur
    new_subscription = Subscription(
        user_id=user.id,
        package_type=package_type,
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow() + timedelta(days=PACKAGE_TYPES[package_type]['duration_days']),
        is_active=True,
        auto_renew=False
    )

    db.session.add(new_subscription)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': 'Paket başarıyla satın alındı',
        'subscription': {
            'id': new_subscription.id,
            'package_type': new_subscription.package_type,
            'start_date': new_subscription.start_date.strftime('%Y-%m-%d'),
            'end_date': new_subscription.end_date.strftime('%Y-%m-%d')
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 