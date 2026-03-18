import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
import joblib
import json
import os
import warnings
import re
import glob
import gc
import time
import io
import shutil

# --- [FIX CRASH TENSORFLOW] ---
# 1. Matikan optimasi oneDNN (Penyebab utama crash "Instructions AVX/SSE")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. Paksa pakai CPU biasa (biar gak bentrok cari GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 3. Hilangkan warning cerewet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ------------------------------

# Library AI
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
from sklearn.preprocessing import MinMaxScaler  # <--- INI YANG HILANG TADI
from sklearn.metrics import mean_absolute_error

# =========================================================
# 🧠 RESEP RAHASIA DARI COLAB (FIXED PARAMETERS)
# =========================================================
MODEL_CONFIG = {
    "Aksesoris": {
        "XGBoost": {
            "p": 1.6140704776430503,
            "n_est": 299,
            "lr": 0.0886222307099048,
            "depth": 2,
            "min_child": 6
        },
        "LSTM": {
            "units": 50,
            "dropout": 0.20617534828874073,
            "lr": 0.008706020878304856
        },
        "SARIMA": {
            "p": 1,
            "d": 1,
            "q": 2,
            "P": 1,
            "D": 0,
            "Q": 0
        }
    },
    "Fashion Bayi & Anak lainnya": {
        "XGBoost": {
            "p": 1.1116793122799113,
            "n_est": 275,
            "lr": 0.04186007623689508,
            "depth": 3,
            "min_child": 20
        },
        "LSTM": {
            "units": 47,
            "layers": 1,
            "dropout": 0.1116167224336399,
            "lr": 0.0029621516588303515
        },
        "SARIMA": {
            "p": 1,
            "d": 1,
            "q": 2,
            "P": 1,
            "D": 0,
            "Q": 0
        }
    },
    "Fashion Muslim": {
        "XGBoost": {
            "p": 1.2681318117695117,
            "n_est": 377,
            "lr": 0.07183315985612762,
            "depth": 6
        },
        "LSTM": {
            "units": 35,
            "layers": 2,
            "dropout": 0.2991691172634857,
            "lr": 0.001987596639513196
        },
        "SARIMA": {
            "p": 0,
            "d": 1,
            "q": 2,
            "P": 0,
            "Q": 0
        }
    },
    "Pakaian Laki-laki": {
        "XGBoost": {
            "p": 1.7094591933409686,
            "n_est": 448,
            "lr": 0.04690954604268307,
            "depth": 5
        },
        "LSTM": {
            "units": 61,
            "layers": 1,
            "dropout": 0.23684660530243137,
            "lr": 0.0005595074635794797
        },
        "SARIMA": {
            "p": 1,
            "d": 0,
            "q": 0,
            "P": 0,
            "Q": 0
        }
    },
    "Pakaian Perempuan": {
        "XGBoost": {
            "n_est": 180,
            "lr": 0.0752691664580665,
            "depth": 3
        },
        "LSTM": {
            "units": 63,
            "dropout": 0.20179915488674519,
            "lr": 0.008691089486124988
        },
        "SARIMA": {
            "p": 1,
            "d": 1,
            "q": 2,
            "P": 1,
            "D": 0,
            "Q": 0
        }
    }
}

# ✅ STABILISASI MEMORI
import sys
sys.setrecursionlimit(3000)  # Cegah stack overflow

# ✅ LIMIT TENSORFLOW MEMORY
import tensorflow as tf
tf.config.set_soft_device_placement(True)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['OMP_NUM_THREADS'] = '1'

# ✅ LIMIT MEMORY USAGE
import platform
if platform.system() != 'Windows':
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, -1))
    except:
        pass

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Dashboard UMKM Pro", layout="wide", page_icon="🛍️")
if 'reset_done' not in st.session_state:
    st.cache_resource.clear()
    st.session_state['reset_done'] = True
    print("🧹 MEMORI DIBERSIHKAN PAKSA!")

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Mencegah retracing berlebih
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if 'page' not in st.session_state: st.session_state['page'] = "Ringkasan Prediksi"

VALID_CATEGORIES = [
    'Fashion Muslim', 'Pakaian Laki-laki', 'Pakaian Perempuan',
    'Aksesoris', 'Fashion Bayi & Anak lainnya'
]

STRATEGY_MAP = {
    'Fashion Muslim': 'SEASONAL_DEEP',
    'Pakaian Laki-laki': 'SEASONAL_DEEP',
    'Pakaian Perempuan': 'MAE_STABLE',
    'Aksesoris': 'SMOOTH_OPERATOR',
    'Fashion Bayi & Anak lainnya': 'SAFETY_MAX'
}

# --- 0. CSS AGAR KOTAK SEJAJAR & RAPI ---
st.markdown("""
    <style>
    div.kpi-container {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        height: 140px; /* Tinggi fix biar sejajar */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .kpi-label { 
        font-size: 13px; 
        color: #6c757d; 
        font-weight: 600; 
        text-transform: uppercase; 
        margin-bottom: 8px;
    }
    .kpi-value-base {
        font-weight: 700; 
        color: #212529; 
        margin: 0;
        line-height: 1.2;
    }
    .kpi-sub { font-size: 13px; margin-top: 8px; }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

try:
    from hijridate import Gregorian
    HIJRI_AVAILABLE = True
    def get_hijri_bell_curve(date_val):
        try:
            h = Gregorian(date_val.year, date_val.month, date_val.day).to_hijri()
            if h.month == 8: return 0.2 + (h.day / 30) * 0.3
            elif h.month == 9: return 0.5 + (h.day / 30) * 0.5
            elif h.month == 10:
                if h.day <= 7: return 0.5 - (h.day / 7) * 0.5
                return 0.0
            return 0.0
        except: return 0.0
except ImportError:
    HIJRI_AVAILABLE = False
    def get_hijri_bell_curve(d): return 0.0
import psutil

def check_memory_usage():
    """Return True jika memory usage > 80%"""
    try:
        mem = psutil.virtual_memory()
        return mem.percent > 80
    except:
        return False



@st.cache_resource(show_spinner=False, max_entries=5, ttl=600)
def load_saved_model(path, model_type):
    try:
        if model_type == 'keras': return load_model(path, compile=False) 
        elif model_type == 'joblib': return joblib.load(path)
        elif model_type == 'json':
            with open(path, 'r') as f: return json.load(f)
        elif model_type == 'sarima_pkl':
            try: return SARIMAXResultsWrapper.load(path)
            except: return joblib.load(path)
    except Exception as e:
        return None
    return None

def smape(y_true, y_pred):
    y_true = np.nan_to_num(np.array(y_true).astype(float), nan=0.0)
    y_pred = np.nan_to_num(np.array(y_pred).astype(float), nan=0.0)
    if len(y_true) == 0: return 0.0
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-10
    return np.mean(numerator / denominator) * 100

import math

def calculate_steps_to_date(start_date, end_date):
    """Menghitung butuh berapa step (2 mingguan) untuk sampai ke end_date"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    if end <= start:
        return 0
        
    # Hitung selisih hari
    days_diff = (end - start).days
    
    # Dibagi 14 hari (karena frekuensi datamu 2 Mingguan / 2W)
    # math.ceil untuk membulatkan ke atas agar target tanggal pasti tercover
    steps = math.ceil(days_diff / 14) 
    
    return int(steps)

import psutil
import tracemalloc

def get_memory_usage():
    """Cek memory usage saat ini"""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024  # Convert to MB
    return mem_mb

def log_memory(label):
    """Print memory usage dengan label"""
    mem = get_memory_usage()
    print(f"💾 [{label}] Memory: {mem:.1f} MB")
    return mem

def find_model_file(prefix, category, extension, use_temp=False):
    # Tentukan folder sasaran
    target_folder = "models_temp" if use_temp else "models"
    
    # Bersihkan nama file
    safe_cat = category.replace("/", "_").replace("&", "dan").replace(" ", "_")
    
    # Coba cari file persis
    exact_path = f"{target_folder}/{prefix}_{safe_cat}{extension}"
    if os.path.exists(exact_path): return exact_path
    
    # Kalau gak ketemu, cari yang mirip (fuzzy search)
    # HANYA JIKA BUKAN TEMP (Kalau temp harus exact biar ga salah ambil)
    if not use_temp:
        files = glob.glob(f"{target_folder}/*{extension}")
        clean_cat_query = category.lower().replace("&", "").replace("/", "")
        best_match = None
        max_score = 0
        for f in files:
            filename = os.path.basename(f).lower()
            if prefix.lower() not in filename: continue
            score = 0
            parts = clean_cat_query.split()
            for p in parts:
                if p in filename: score += 1
            if score > max_score:
                max_score = score
                best_match = f
        return best_match
    
    return None

def categorize_item(item_name):
    if pd.isna(item_name): return 'Unknown'
    item_name = f" {str(item_name).lower()} "
    def has_word(keywords, target_string):
        pattern = r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b'
        return bool(re.search(pattern, target_string))
    clothing_guard = ['dress', 'gamis', 'tunik', 'set', 'oneset', 'baju', 'kaos', 'shirt', 't-shirt', 'pants', 'celana', 'rok', 'skirt', 'blouse', 'jacket', 'jaket', 'hoodie', 'sweater', 'knitwear', 'longsleeve', 'cargo', 'jeans', 'chinos', 'suit', 'overall', 'jumpsuit', 'piyama', 'pajamas', 'koko', 'couple', 'sarimbit', 'leging', 'legging', 'shortpant', 'kulot', 'cullote', 'vest', 'outer', 'cardigan', 'crewneck', 'robe', 'skort', 'top', 'bottom', 'sholawat', 'ngaji']
    kw_food = ['coklat', 'chocolate', 'silverqueen', 'lapis legit', 'makanan', 'snack', 'kue', 'delfi', 'minyak kutus', 'parfum', 'skincare', 'sabun', 'kutek', 'body mist', 'lotion', 'shampoo', 'odol', 'sikat gigi', 'masker wajah', 'tissue', 'skintific', 'serum', 'kado', 'hampers', 'gift', 'paper bag', 'kardus', 'plastik packing', 'lollipop', 'permen', 'maltitos', 'lip stick', 'lip balm', 'lip gloss', 'lip tint', 'lip cream', 'lip glow']
    if any(k in item_name for k in kw_food):
        if not any(cg in item_name for cg in clothing_guard): return 'Kebutuhan Harian & Makanan'
    kw_home = ['sprei', 'bedcover', 'bantal', 'guling', 'handuk', 'towel', 'selimut', 'blanket', 'karpet', 'keset', 'gorden', 'wajan', 'panci', 'frypan', 'teflon', 'pisau dapur', 'gelas', 'piring', 'mangkok', 'sapu', 'pel ', 'botol minum', 'tumbler', 'termos', 'lunch box', 'kotak makan', 'misting', 'sendok', 'garpu', 'chopper', 'blender', 'drink bottle', 'stainless steel']
    if has_word(['rak'], item_name) and 'sepatu' in item_name: return 'Perlengkapan Rumah'
    if any(k in item_name for k in kw_home):
        if 'piyama' not in item_name and 'pajamas' not in item_name: return 'Perlengkapan Rumah'
    kw_school = ['buku', 'pulpen', 'pensil', 'pencil', 'penghapus', 'penggaris', 'sampul', 'map', 'binder', 'atk', 'stationery', 'smiggle', 'back to school', 'crayon']
    if any(k in item_name for k in kw_school):
        if not any(cg in item_name for cg in clothing_guard): return 'Perlengkapan Sekolah'
    if 'stiker' in item_name or 'label' in item_name:
        if not any(cg in item_name for cg in clothing_guard): return 'Perlengkapan Sekolah'
    kw_gear_general = ['sepatu', 'sandal', 'sendal', 'shoes', 'heels', 'wedges', 'flat shoes', 'sneaker', 'boots', 'selop', 'slipper', 'slip on', 'slipon', 'alas kaki', 'kaus kaki', 'sock', 'footwear', 'ransel', 'backpack', 'selempang', 'sling bag', 'tote', 'waist', 'dompet', 'wallet', 'pouch', 'clutch', 'koper', 'kacamata', 'jam tangan', 'ikat pinggang', 'bros', 'jepit', 'bando', 'anting', 'kalung', 'gelang']
    kw_gear_strict = ['helm', 'pin', 'cap', 'hat', 'topi', 'bucket', 'belt']
    if any(k in item_name for k in kw_gear_general): return 'Aksesoris'
    if has_word(['tas', 'bag'], item_name):
        if 'kertas' not in item_name and 'paper' not in item_name: return 'Aksesoris'
    if has_word(kw_gear_strict, item_name):
        is_clothing_bonus = False
        if any(cg in item_name for cg in clothing_guard):
            if 'free' in item_name or 'plus' in item_name: is_clothing_bonus = True
        if not is_clothing_bonus: return 'Aksesoris'
    kw_muslim = ['mukena', 'tunik', 'hijab', 'abaya', 'gamis', "syar'i", 'jilbab', 'koko', 'sarung', 'kurta', 'shaleha', 'muslimah', 'peci', 'sajadah', 'khimar', 'bergo', 'kemko', 'jubah', 'turban', 'ciput', 'iqro', 'quran', 'ngaji', 'kerudung', 'sarcel', 'sarceko', 'dakwah']
    if any(k in item_name for k in kw_muslim): return 'Fashion Muslim'
    kw_kids = ['bayi', 'baby', 'anak', 'kids', 'kid', 'junior', 'jun', 'teen', 'newborn', 'balita', 'toodler', 'libby', 'miyo', 'velvet', 'bohopana', 'little', 'cuit', 'vitto', 'paddlekids', 'kakadede', 'oshkosh', 'senshukei', 'baju renang']
    if any(k in item_name for k in kw_kids): return 'Fashion Bayi & Anak lainnya'
    kw_men = ['laki', 'pria', 'cowok', 'cowo', 'men', 'man', 'male', 'bapak', 'father', 'dad', 'ayah', 'kemeja', 'flanel', 'boxer', 'boy']
    kw_women = ['wanita', 'perempuan', 'cewe', 'girl', 'woman', 'women', 'lady', 'nona', 'bunda', 'mom', 'ibu', 'dress', 'blouse', 'rok', 'skirt', 'kulot', 'cullote', 'legging', 'leging', 'bra', 'cd wanita', 'celana dalam wanita', 'tanktop', 'kemben', 'daster', 'homedress', 'mididress', 'inner', 'manset', 'cardigan', 'longdress', 'hotpants', 'busui', 'cutbray', 'scraves', 'pashmina']
    kw_general_clothing = ['shirt', 't-shirt', 'kaos', 'polo', 'hem', 'raglan', 'chino', 'cargo', 'jeans', 'distro', 'denim', 'levis', 'celana', 'pants', 'pant', 'underwear', 'celdam', 'joger', 'jogger', 'sweater', 'hoodie', 'jacket', 'jaket', 'outer', 'knit', 'vest', 'coat', 'jersey', 'sport', 'training', 'oversize']
    if any(k in item_name for k in kw_men) or has_word(['boxer'], item_name): return 'Pakaian Laki-laki'
    if any(k in item_name for k in kw_women): return 'Pakaian Perempuan'
    if any(k in item_name for k in kw_general_clothing): return 'Pakaian Laki-laki'
    return 'Fashion Bayi & Anak lainnya'

def smart_date_parser(df, date_col, id_col=None):
    col_data = df[date_col].astype(str)
    dates = pd.to_datetime(col_data, errors='coerce')
    if id_col and id_col in df.columns:
        mask_fail = dates.isna()
        if mask_fail.sum() > 0:
            id_data = df.loc[mask_fail, id_col].astype(str)
            extracted = id_data.str.extract(r'^(\d{6})')[0]
            dates_from_id = pd.to_datetime(extracted, format='%y%m%d', errors='coerce')
            dates.loc[mask_fail] = dates_from_id
    return dates

def calculate_metrics(df_full, model_name, category=None):
    if category is None:
        scores = []
        for cat in df_full['Kategori'].unique():
            s = calculate_metrics(df_full, model_name, cat)
            scores.append(s)
        return np.mean(scores) if scores else 0

    df_pred = df_full[(df_full['Kategori'] == category) & (df_full['Model'] == model_name) & (df_full['Jenis'] == 'Test')]
    
    # ✅ FIX FINAL: KARENA APPLE-TO-APPLE, SEMUA MODEL DINILAI PAKAI SMOOTH!
    target_label = 'Aktual (Smooth)'
    
    df_act = df_full[(df_full['Kategori'] == category) & (df_full['Model'] == target_label) & (df_full['Jenis'] == 'Aktual')]
    
    # Fallback
    if df_act.empty:
        df_act = df_full[(df_full['Kategori'] == category) & (df_full['Model'] == model_name) & (df_full['Jenis'] == 'Aktual')]
    if df_act.empty:
        df_act = df_full[(df_full['Kategori'] == category) & (df_full['Jenis'] == 'Aktual')]

    if df_pred.empty or df_act.empty: return 100

    df_pred = df_pred.sort_values('Tanggal')
    df_act = df_act.sort_values('Tanggal')
    
    # Samakan irisan waktu (Penting untuk anti-miss tanggal)
    df_act = df_act.drop_duplicates(subset=['Tanggal'])
    df_act = df_act[pd.to_datetime(df_act['Tanggal']).dt.date.isin(pd.to_datetime(df_pred['Tanggal']).dt.date.values)]

    val_pred = df_pred['Nilai'].values
    val_act = df_act['Nilai'].values
    
    min_len = min(len(val_pred), len(val_act))
    if min_len == 0: return 100
    
    return smape(val_act[-min_len:], val_pred[-min_len:])

def apply_post_processing(preds_array, df_test, strategy):
    """Fungsi Rem Tangan (Safety Net) untuk Data Uji 10%"""
    final_preds = []
    bell_vals = df_test['Bell_Curve'].values
    
    # Ambil median dari data uji, kalau tidak ada, pakai 0
    if 'rolling_median_6' in df_test.columns:
        # KUNCI PERBAIKAN: Kembalikan median dari Log ke angka asli (Linear)!!!
        med_vals_log = df_test['rolling_median_6'].values
        med_vals = np.expm1(np.nan_to_num(med_vals_log, nan=0.0))
    else:
        med_vals = np.zeros(len(preds_array))
        
    for i, val in enumerate(preds_array):
        if strategy == 'SAFETY_MAX':
            med_val = med_vals[i]
            # Tekan tebakan jika anomali di bulan biasa
            if bell_vals[i] < 0.25 and val > (med_val * 1.3):
                val = med_val
            # Netralkan jika tren mati
            if med_val < 2:
                val = 0.0
        # Validasi absolut (Anti minus)
        final_preds.append(max(0, val))
        
    return np.array(final_preds)

# =========================================================
# 🤖 MESIN TRAINING OTOMATIS (ON-THE-FLY TRAINING)
# =========================================================

# 1. Helper: Feature Engineering (Wajib sama persis dengan Colab)
def apply_feature_engineering(df_raw, strategy):
    """
    Feature Engineering 100% SAMA dengan Colab.
    TIDAK ADA dropna() di sini! (Dilakukan di luar)
    """
    df = df_raw.sort_values("Bulan").copy()
    
    # STEP 1: SMOOTHING
    if strategy == 'SMOOTH_OPERATOR':
        df['Total_Jumlah'] = df['Total_Jumlah'].rolling(window=3, min_periods=1).mean()
    
    df['Smooth_Target'] = df['Total_Jumlah'].rolling(window=2, min_periods=1).mean()
    df['Target_Log'] = np.log1p(df['Smooth_Target'])
    
    # STEP 2: EXTERNAL FEATURES
    df["Week_Num"] = df["Bulan"].dt.isocalendar().week
    df["Sin_Week"] = np.sin(2 * np.pi * df["Week_Num"] / 52)
    df["Cos_Week"] = np.cos(2 * np.pi * df["Week_Num"] / 52)
    df["Bell_Curve"] = df["Bulan"].apply(get_hijri_bell_curve)
    
    # STEP 3: LAG FEATURES
    df["lag_1"] = df["Target_Log"].shift(1)
    df["lag_2"] = df["Target_Log"].shift(2)
    df["velocity"] = df["lag_1"] - df["lag_2"]
    df["rolling_mean_4"] = df["Target_Log"].shift(1).rolling(4, min_periods=1).mean()
    df["rolling_median_6"] = df["Target_Log"].shift(1).rolling(6, min_periods=1).median()
    
    # STEP 4: SEASONAL LAG
    if strategy == 'SEASONAL_DEEP':
        if len(df) > 30:
            df["lag_26"] = df["Target_Log"].shift(26)
        else:
            df["lag_26"] = 0
    
    # ✅ PERBAIKAN: JANGAN dropna() di sini!
    # Biarkan NaN ada, nanti di-handle saat split train/test
    return df.reset_index(drop=True)

# ==============================================================================
# 2. FUNGSI DATA PREP & DEFAULT TRAINING (Hanya XGBoost) - FIXED 90:10
# ==============================================================================
def train_models_on_upload(df_clean):
    print("🚀 START: TRAINING 90:10 ON UPLOAD...")
    st.session_state['cache_df_ml'] = {}
    all_results = []
    
    # 1. Filter
    df_clean = df_clean[~((df_clean['Kategori'] == 'Pakaian Perempuan') & (df_clean['Total_Jumlah'] > 1000))].copy()
    df_clean['Bulan'] = df_clean['Tanggal']
    agg_data = df_clean.groupby(['Kategori', pd.Grouper(key='Bulan', freq='2W')])['Total_Jumlah'].sum().reset_index()
    
    unique_cats = agg_data['Kategori'].unique()
    total_steps = len(unique_cats)
    prog_bar = st.progress(0, text="Menyiapkan data & Model Default...")
    
    # Buat folder models jika belum ada
    if not os.path.exists("models_temp"): os.makedirs("models_temp")
    
    for i, cat in enumerate(unique_cats):
        prog_bar.progress(int(((i + 1) / total_steps) * 100), text=f"Processing: {cat}")
        K.clear_session(); gc.collect()
        
        df_cat = agg_data[agg_data['Kategori'] == cat].copy()
        if len(df_cat) < 4: continue
        
        strategy = STRATEGY_MAP.get(cat, 'MAE_STABLE')
        params = MODEL_CONFIG[cat]
        
        # Feature Engineering
        df_ml = apply_feature_engineering(df_cat, strategy)
        # st.session_state['cache_df_ml'][cat] = df_ml
        
        # --- PERBAIKAN: SIMPAN KEDUA DATA AKTUAL ---
        # 1. Simpan Aktual Raw (Untuk background grafik abu-abu)
        all_results.append(pd.DataFrame({
            'Tanggal': df_cat['Bulan'], 
            'Nilai': df_cat['Total_Jumlah'],
            'Jenis': 'Aktual', 'Model': 'Aktual (Raw)', 'Kategori': cat, 'SMAPE': 0
        }))
        
        # 2. Simpan Aktual Smooth (Untuk kunci jawaban SMAPE & garis hitam)
        all_results.append(pd.DataFrame({
            'Tanggal': df_ml['Bulan'], 
            'Nilai': np.expm1(df_ml['Target_Log']),
            'Jenis': 'Aktual', 'Model': 'Aktual (Smooth)', 'Kategori': cat, 'SMAPE': 0
        }))
        
        # Drop NaN untuk Training
        df_ml_clean = df_ml.dropna().reset_index(drop=True)
        
        # ✂️ SPLIT 90:10 (INI LOGIKA KUNCINYA)
        n_total = len(df_ml_clean)
        n_test = max(1, int(n_total * 0.1)) # Minimal 1 data test
        
        df_train = df_ml_clean.iloc[:-n_test]
        df_test = df_ml_clean.iloc[-n_test:] # INI DATA UJIAN
        
        # Setup Features
        if strategy == 'SEASONAL_DEEP': feats = ["lag_1", "lag_2", "Bell_Curve", "Sin_Week"]
        elif strategy == 'MAE_STABLE': feats = ["lag_1", "lag_2", "Bell_Curve", "Sin_Week", "Cos_Week"]
        elif strategy == 'SAFETY_MAX': feats = ["lag_1", "Bell_Curve", "Sin_Week"]
        elif strategy == 'SMOOTH_OPERATOR': feats = ["lag_1", "rolling_mean_4", "velocity"]
        else: feats = ["lag_1", "Bell_Curve"]
        feats = [f for f in feats if f in df_ml_clean.columns]
        
        # --- TRAIN XGBOOST (Hanya pada df_train - 90%) ---
        try:
            p_xgb = params['XGBoost']
            xgb_params = {
                'n_estimators': int(p_xgb.get('n_est', 200)), 
                'learning_rate': p_xgb.get('lr', 0.1),
                'max_depth': int(p_xgb.get('depth', 3)),
                'random_state': 42,
                'verbosity': 0
            }
            
            # Pasang Objective sesuai Strategi (Persis Colab)
            if strategy in ['MAE_STABLE', 'SET_CLASSIC']: 
                xgb_params['objective'] = 'reg:absoluteerror'
            elif 'p' in p_xgb: 
                xgb_params['objective'] = 'reg:tweedie'
                xgb_params['tweedie_variance_power'] = p_xgb['p']
                
            if 'min_child' in p_xgb: 
                xgb_params['min_child_weight'] = p_xgb['min_child']

            xgb_model = xgb.XGBRegressor(**xgb_params)
            
            # Hitung Bobot Lebaran
            w_train = np.ones(len(df_train))
            if strategy == 'SAFETY_MAX': w_train = df_train['Bell_Curve'].apply(lambda x: 10.0 if x > 0.3 else 1.0).values
            elif strategy != 'SMOOTH_OPERATOR': w_train = df_train['Bell_Curve'].apply(lambda x: 5.0 if x > 0.3 else 1.0).values

            # FIT 90% DENGAN BOBOT
            xgb_model.fit(df_train[feats], df_train['Target_Log'], sample_weight=w_train)
            
            # 📝 UJIAN (PREDICT 10%) - UNTUK DAPAT ANGKA SMAPE
            preds_log = xgb_model.predict(df_test[feats])
            preds_real_raw = np.expm1(preds_log)
            preds_real = apply_post_processing(preds_real_raw, df_test, strategy)
            actuals_real = np.expm1(df_test['Target_Log'])
            score_smape = smape(actuals_real, preds_real)
            
            # Simpan Hasil Ujian (Test)
            all_results.append(pd.DataFrame({
                'Tanggal': df_test['Bulan'], 
                'Nilai': preds_real, 
                'Jenis': 'Test', # <-- Ini yang dibaca halaman KPI
                'Model': 'Model Pintar', 
                'Kategori': cat, 
                'SMAPE': score_smape
            }))
            
            # Simpan Model (Model ini "jujur" hanya tahu 90% data)
            safe_cat = cat.replace("/", "_").replace("&", "dan").replace(" ", "_")
            xgb_model.save_model(f"models_temp/XGBoost_{safe_cat}.json")
            
            # 🔮 FUTURE FORECAST (Walk-Forward)
            # Kita pakai model yg dilatih 90%, TAPI input lag-nya ambil dari data terbaru (100%)
            # Jadi prediksinya tetap start dari besok.
            _, _, fut_vals = run_xgboost_inference(df_ml, cat, strategy, steps=4, use_temp=True)
            fut_dates = pd.date_range(start=df_ml_clean['Bulan'].max(), periods=5, freq='2W')[1:]
            
            if fut_vals and len(fut_vals) > 0:
                all_results.append(pd.DataFrame({
                    'Tanggal': fut_dates, 'Nilai': fut_vals, 'Jenis': 'Future', 
                    'Model': 'Model Pintar', 'Kategori': cat, 'SMAPE': 0
                }))
            
            del xgb_model
        except Exception as e: 
            print(f"XGB Error {cat}: {e}")
    
    prog_bar.empty()
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()

# ==============================================================================
# 3. FUNGSI TRAINING ON-DEMAND (LSTM/SARIMA) - FIXED 90:10
# ==============================================================================
def train_specific_model(model_name):
    # [UBAH] Cek data mentah, bukan cache
    if 'data_clean' not in st.session_state: return pd.DataFrame()
    
    # [UBAH] Ambil data MENTAH & Agregasi Ulang (On-Demand)
    df_raw_source = st.session_state['data_clean'].copy()
    if 'Bulan' not in df_raw_source.columns:
        df_raw_source['Bulan'] = df_raw_source['Tanggal']

    agg_data = df_raw_source.groupby(['Kategori', pd.Grouper(key='Bulan', freq='2W')])['Total_Jumlah'].sum().reset_index()
    unique_cats = agg_data['Kategori'].unique()
    
    new_res = []
    total = len(unique_cats)
    prog = st.progress(0, text=f"Sedang melatih {model_name} (90:10 Split)...")
    
    if not os.path.exists("models_temp"): os.makedirs("models_temp")
    
    # [UBAH] Loop berdasarkan unique_cats, bukan cache.items()
    for i, cat in enumerate(unique_cats):
        prog.progress(int((i / total) * 100), text=f"Processing {cat}...")
        
        try: K.clear_session(); gc.collect()
        except: pass
        
        try:
            strategy = STRATEGY_MAP.get(cat, 'MAE_STABLE')
            params = MODEL_CONFIG[cat]
            safe_cat = cat.replace("/", "_").replace("&", "dan").replace(" ", "_")
            
            # [UBAH] Ambil data kategori & Hitung Fitur Dadakan
            df_cat = agg_data[agg_data['Kategori'] == cat].copy()
            if len(df_cat) < 4: continue

            # Feature Engineering (Dibuat -> Dipakai -> Dibuang otomatis)
            df_ml = apply_feature_engineering(df_cat, strategy)
            
            # --- MULAI DARI SINI LOGIC SAMA PERSIS DENGAN SEBELUMNYA ---
            
            # Bersihkan NaN & Split
            df_ml_clean = df_ml.dropna().reset_index(drop=True)
            if len(df_ml_clean) < 5: continue
            
            n_total = len(df_ml_clean)
            n_test = max(1, int(n_total * 0.1))
            
            df_train = df_ml_clean.iloc[:-n_test]
            df_test = df_ml_clean.iloc[-n_test:]
            
            # ==========================================
            # LSTM (Model Canggih)
            # ==========================================
            if model_name == "Model Canggih":
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
                from tensorflow.keras.optimizers import Adam
                
                if strategy == 'SEASONAL_DEEP': feats = ["lag_1", "lag_2", "Bell_Curve", "Sin_Week"]
                elif strategy == 'MAE_STABLE': feats = ["lag_1", "lag_2", "Bell_Curve", "Sin_Week", "Cos_Week"]
                elif strategy == 'SAFETY_MAX': feats = ["lag_1", "Bell_Curve", "Sin_Week"]
                elif strategy == 'SMOOTH_OPERATOR': feats = ["lag_1", "rolling_mean_4", "velocity"]
                else: feats = ["lag_1", "Bell_Curve"]
                
                for f in feats: 
                    if f not in df_ml_clean.columns: df_ml_clean[f] = 0.0
                    if f not in df_train.columns: df_train[f] = 0.0
                    if f not in df_test.columns: df_test[f] = 0.0
                
                scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
                X_train_raw = df_train[feats].values
                y_train_raw = df_train[['Target_Log']].values
                w_train_raw = df_train['Bell_Curve'].values # <-- Ambil bobot
                
                scaler_X.fit(X_train_raw); scaler_y.fit(y_train_raw)
                joblib.dump(scaler_X, f"models_temp/Scaler_X_{safe_cat}.pkl")
                joblib.dump(scaler_y, f"models_temp/Scaler_y_{safe_cat}.pkl")
                
                # REVISI ALIGNMENT WAKTU & WEIGHT
                def create_seq(X, y, w_arr, steps=3):
                    Xs, ys, ws = [], [], []
                    if len(X) < steps: return np.array([]), np.array([]), np.array([])
                    for i in range(len(X) - steps + 1):
                        Xs.append(X[i:(i+steps)])
                        ys.append(y[i+steps-1])
                        ws.append(w_arr[i+steps-1])
                    return np.array(Xs), np.array(ys), np.array(ws)
                
                # Bikin Pembobotan Lebaran (Sample Weights)
                sample_weights = np.ones(len(w_train_raw))
                for idx_w, val_w in enumerate(w_train_raw):
                    if strategy == 'SAFETY_MAX' and val_w > 0.3: sample_weights[idx_w] = 10.0
                    elif strategy != 'SMOOTH_OPERATOR' and val_w > 0.3: sample_weights[idx_w] = 5.0
                
                X_train_sc = scaler_X.transform(X_train_raw)
                y_train_sc = scaler_y.transform(y_train_raw)
                X_seq_train, y_seq_train, w_seq_train = create_seq(X_train_sc, y_train_sc, sample_weights)
                
                full_X_sc = scaler_X.transform(df_ml_clean[feats].values)
                full_y_sc = scaler_y.transform(df_ml_clean[['Target_Log']].values)
                full_w_sc = df_ml_clean['Bell_Curve'].values
                start_idx = max(0, len(full_X_sc) - len(df_test) - 3)
                X_seq_test, y_seq_test, _ = create_seq(full_X_sc[start_idx:], full_y_sc[start_idx:], full_w_sc[start_idx:])
                
                # --- FIX LSTM: MULTI-LAYER & EPOCH 60 ---
                tf.random.set_seed(42) # Kunci takdir
                p_lstm = params['LSTM']
                model = Sequential()
                model.add(Input(shape=(3, len(feats))))
                
                n_layers = p_lstm.get('layers', 1)
                for _layer_idx in range(n_layers):
                    return_seq = True if _layer_idx < n_layers - 1 else False
                    model.add(LSTM(p_lstm.get('units', 32), return_sequences=return_seq))
                    model.add(Dropout(p_lstm.get('dropout', 0.2)))
                    
                model.add(Dense(1))
                model.compile(optimizer=Adam(learning_rate=p_lstm.get('lr', 0.001)), loss='mae')
                
                if len(X_seq_train) > 0:
                    # Masukkan sample_weight ke proses fit, EPOCH 60, BATCH 16!
                    model.fit(X_seq_train, y_seq_train, sample_weight=w_seq_train, epochs=60, batch_size=16, verbose=0)
                    model.save(f"models_temp/LSTM_{safe_cat}.keras")
                    
                    if len(X_seq_test) > 0:
                        pred_sc = model.predict(X_seq_test, verbose=0)
                        pred_real = np.expm1(scaler_y.inverse_transform(pred_sc)).flatten()
                        min_len = min(len(pred_real), len(df_test))
                        
                        if min_len > 0:
                            y_pred_raw = pred_real[-min_len:]
                            df_test_aligned = df_test.iloc[-min_len:].copy()
                            y_pred_final = apply_post_processing(y_pred_raw, df_test_aligned, strategy)

                            y_true_final = np.expm1(df_test['Target_Log'].iloc[-min_len:].values)
                            dates_final = df_test['Bulan'].iloc[-min_len:]
                            smape_val = smape(y_true_final, y_pred_final)
                            
                            new_res.append(pd.DataFrame({
                                'Tanggal': dates_final, 'Nilai': y_pred_final, 'Jenis': 'Test',
                                'Model': 'Model Canggih', 'Kategori': cat, 'SMAPE': smape_val
                            }))
                    
                    _, _, fut = run_lstm_inference(df_ml, cat, strategy, steps=4, use_temp=True)
                    if fut:
                        dates = pd.date_range(start=df_ml['Bulan'].max(), periods=len(fut)+1, freq='2W')[1:]
                        new_res.append(pd.DataFrame({
                            'Tanggal': dates, 'Nilai': fut, 'Jenis': 'Future',
                            'Model': 'Model Canggih', 'Kategori': cat, 'SMAPE': 0
                        }))
                del model
            
            # ==========================================
            # SARIMA (Model Klasik)
            # ==========================================
            elif model_name == "Model Klasik":
                p_sar = params['SARIMA']
                final_D = 1 if strategy == 'SEASONAL_DEEP' else p_sar.get('D', 0)

                # Pakai data asli (df_ml)
                idx_split = len(df_ml) - len(df_test)
                y_train_sar = df_ml['Target_Log'].iloc[:idx_split]
                x_train_sar = df_ml[['Bell_Curve']].iloc[:idx_split]

                sar_model = SARIMAX(y_train_sar, exog=x_train_sar,
                                    order=(p_sar['p'], p_sar['d'], p_sar['q']),
                                    seasonal_order=(p_sar['P'], final_D, p_sar['Q'], 26),
                                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
                sar_model.save(f"models_temp/SARIMA_{safe_cat}.pkl")

                x_test_sar = df_test[['Bell_Curve']]
                preds_log = sar_model.forecast(steps=len(df_test), exog=x_test_sar)
                
                # --- FIX SARIMA: POST PROCESSING (REM TANGAN) ---
                preds_real_raw = np.expm1(preds_log.values)
                preds_real = apply_post_processing(preds_real_raw, df_test, strategy)
                actuals_real = np.expm1(df_test['Target_Log'].values)
                smape_val = smape(actuals_real, preds_real)
                
                new_res.append(pd.DataFrame({
                    'Tanggal': df_test['Bulan'], 'Nilai': preds_real, 'Jenis': 'Test',
                    'Model': 'Model Klasik', 'Kategori': cat, 'SMAPE': smape_val
                }))
                
                df_base_sar = pd.DataFrame({'Total_Jumlah': df_ml['Total_Jumlah'], 'Bulan': df_ml['Bulan']})
                df_base_sar['Bell_Curve'] = df_ml['Bell_Curve']
                _, _, fut_sar = run_sarima_inference(df_base_sar, cat, strategy, steps=4, use_temp=True)
                if fut_sar:
                    dates = pd.date_range(start=df_ml['Bulan'].max(), periods=len(fut_sar)+1, freq='2W')[1:]
                    new_res.append(pd.DataFrame({
                        'Tanggal': dates, 'Nilai': fut_sar, 'Jenis': 'Future',
                        'Model': 'Model Klasik', 'Kategori': cat, 'SMAPE': 0
                    }))
                del sar_model

        except Exception as e:
            print(f"Error {model_name} {cat}: {e}")
        
    prog.empty()
    try: K.clear_session(); gc.collect()
    except: pass
    
    if new_res: return pd.concat(new_res, ignore_index=True)
    return pd.DataFrame()

# =========================================================
# 3. PROCESSING ENGINE
# =========================================================

def repair_and_load_bytes(uploaded_file):
    content = uploaded_file.getvalue().decode('utf-8', errors='replace')
    lines = content.splitlines()
    if len(lines) < 2: return None
    header = lines[0].strip().split(',')
    expected_cols = len(header)
    repaired_lines = [lines[0].strip()]
    for i, line in enumerate(lines[1:]):
        clean_line = line.replace('"', '').strip()
        parts = clean_line.split(',')
        if len(parts) == expected_cols: repaired_lines.append(clean_line)
        elif len(parts) > expected_cols:
            head_part = parts[:3]
            tail_count = expected_cols - 4
            tail_part = parts[-tail_count:]
            middle_part = [" ".join(parts[3 : -tail_count])]
            repaired_lines.append(",".join(head_part + middle_part + tail_part))
        else:
            while len(parts) < expected_cols: parts.append('')
            repaired_lines.append(",".join(parts))
    return pd.read_csv(io.StringIO("\n".join(repaired_lines)))

def prepare_features_for_inference(df_raw, cat):
    strategy = STRATEGY_MAP.get(cat, 'MAE_STABLE')
    df = df_raw.sort_values("Bulan").copy()
    if strategy == 'SMOOTH_OPERATOR':
        df['Total_Jumlah'] = df['Total_Jumlah'].rolling(window=3, min_periods=1).mean()
    df['Smooth_Target'] = df['Total_Jumlah'].rolling(window=2, min_periods=1).mean()
    df['Target_Log'] = np.log1p(df['Smooth_Target'])
    df["Week_Num"] = df["Bulan"].dt.isocalendar().week
    df["Sin_Week"] = np.sin(2 * np.pi * df["Week_Num"] / 52)
    df["Cos_Week"] = np.cos(2 * np.pi * df["Week_Num"] / 52)
    df["Bell_Curve"] = df["Bulan"].apply(get_hijri_bell_curve)
    df_base = df.copy()
    df["lag_1"] = df["Target_Log"].shift(1)
    df["lag_2"] = df["Target_Log"].shift(2)
    df["velocity"] = df["lag_1"] - df["lag_2"]
    df["rolling_mean_4"] = df["Target_Log"].shift(1).rolling(4, min_periods=1).mean()
    df["rolling_median_6"] = df["Target_Log"].shift(1).rolling(6, min_periods=1).median()
    if strategy == 'SEASONAL_DEEP':
        if len(df) > 30: 
            df["lag_26"] = df["Target_Log"].shift(26).fillna(0)
        else: 
            df["lag_26"] = 0
    df_ml = df.dropna().reset_index(drop=True)
    return df_base, df_ml, strategy


# =========================================================
# 🛠️ HELPER & INFERENCE FUNCTIONS (PURE COLAB + WARM UP)
# =========================================================

def get_warmup_data(cat, model_name):
    if 'data_ready' not in st.session_state: return []
    df = st.session_state['data_ready']
    
    if 'XGBoost' in model_name: target = 'Model Pintar'
    elif 'LSTM' in model_name: target = 'Model Canggih'
    elif 'SARIMA' in model_name: target = 'Model Klasik'
    else: target = model_name
    
    # Ambil data CSV yang jenisnya 'Future' untuk kategori ini
    warmup = df[
        (df['Kategori'] == cat) & 
        (df['Model'] == target) & 
        (df['Jenis'] == 'Future')
    ].sort_values('Tanggal')
    
    if not warmup.empty: return warmup['Nilai'].tolist()
    return []

def run_sarima_inference(df_base, cat, strategy, steps=4, use_temp=False):
    # 1. Cari File (Cek Temp dulu)
    pkl_path = find_model_file("SARIMA", cat, ".pkl", use_temp=use_temp)
    
    # Fallback ke Asli
    if use_temp and not pkl_path:
        pkl_path = find_model_file("SARIMA", cat, ".pkl", use_temp=False)

    # Warmup kosong kalau mode upload/temp
    warmup_vals = [] if use_temp else get_warmup_data(cat, 'Model Klasik')
    
    series_data = df_base['Total_Jumlah'].fillna(0)
    exog_data = df_base[['Bell_Curve']].fillna(0)
    
    full_history = series_data.tolist()
    full_exog = exog_data['Bell_Curve'].tolist()
    
    if warmup_vals:
        full_history += warmup_vals
        last_date_hist = df_base['Bulan'].max()
        dates_warmup = pd.date_range(start=last_date_hist, periods=len(warmup_vals)+1, freq='2W')[1:]
        exog_warmup = [get_hijri_bell_curve(d) for d in dates_warmup]
        full_exog += exog_warmup

    remaining_steps = steps - len(warmup_vals)
    if remaining_steps <= 0:
        try:
            del loaded_model, new_model
            gc.collect()
        except:
            pass
        return pd.Series(dtype=float), 0, warmup_vals[:steps]

    last_date_total = df_base['Bulan'].max()
    if warmup_vals: last_date_total += pd.Timedelta(weeks=2*len(warmup_vals))
    future_dates = pd.date_range(start=last_date_total, periods=remaining_steps+1, freq='2W')[1:]
    exog_future = pd.DataFrame({'Bell_Curve': [get_hijri_bell_curve(d) for d in future_dates]}, index=future_dates)

    try:
        if pkl_path:
            loaded_model = load_saved_model(pkl_path, 'sarima_pkl')
            new_model = loaded_model.apply(full_history, exog=full_exog)
            new_preds = new_model.forecast(steps=remaining_steps, exog=exog_future)
            # Safety Net
            new_preds = [max(0, x) for x in new_preds]
            return pd.Series(dtype=float), 0, warmup_vals + new_preds
    except Exception as e:
        return None, 0, warmup_vals
    
    return None, 0, warmup_vals

# Tambahkan parameter use_temp=False di sini
def run_xgboost_inference(df_ml, cat, strategy, steps=4, use_temp=False):
    
    # Panggil pencari file dengan parameter use_temp
    pkl_path = find_model_file("XGBoost", cat, ".pkl", use_temp=use_temp)
    json_path = find_model_file("XGBoost", cat, ".json", use_temp=use_temp)
    
    model = None
    try:
        if json_path: model = xgb.XGBRegressor(); model.load_model(json_path)
        elif pkl_path: model = load_saved_model(pkl_path, 'joblib')
    except: pass
    
    # --- LOGIKA FALLBACK (PENTING!) ---
    # Kalau di folder TEMP gak ketemu modelnya (mungkin gagal training),
    # JANGAN CRASH. Tapi pinjam model dari folder ASLI (models/) sebentar.
    if not model and use_temp:
        # Coba cari di folder asli sebagai cadangan
        pkl_path = find_model_file("XGBoost", cat, ".pkl", use_temp=False)
        json_path = find_model_file("XGBoost", cat, ".json", use_temp=False)
        try:
            if json_path: model = xgb.XGBRegressor(); model.load_model(json_path)
            elif pkl_path: model = load_saved_model(pkl_path, 'joblib')
        except: pass

    # Kalau masih ga ketemu juga, baru nyerah
    if not model: 
        last_val = np.expm1(df_ml['Target_Log'].iloc[-1]) if not df_ml.empty else 0
        return pd.Series(dtype=float), 0, [last_val]*steps

    # ... (SISA KODE KE BAWAH SAMA PERSIS DENGAN YANG TADI) ...
    # ... (Pastikan Logika Safety Net Anti-0 TETAP ADA di bawah sini) ...
    
    # -----------------------------------------------------------
    # COPY PASTE BAGIAN LOOPING PREDIKSI DARI KODE SEBELUMNYA
    # YANG ADA "SAFETY NET" DAN "REM TANGAN"-NYA DI SINI
    # -----------------------------------------------------------
    
    # Biar ga bingung, ini saya tulis ulang bagian bawahnya sekalian:
    last_row = df_ml.iloc[-1].fillna(0).copy()
    curr_lag_1 = float(last_row.get('lag_1', 0))
    curr_lag_2 = float(last_row.get('lag_2', curr_lag_1))
    curr_roll_4 = float(last_row.get('rolling_mean_4', curr_lag_1))
    
    # KUNCI: Kalau pakai TEMP (Upload), jangan ambil Warmup
    warmup_vals = [] if use_temp else get_warmup_data(cat, 'Model Pintar')
    
    if strategy == 'SEASONAL_DEEP': feats = ["lag_1", "lag_2", "Bell_Curve", "Sin_Week"]
    elif strategy == 'SET_CLASSIC': feats = ["lag_1", "lag_2", "velocity", "Bell_Curve"]
    elif strategy == 'MAE_STABLE': feats = ["lag_1", "lag_2", "Bell_Curve", "Sin_Week", "Cos_Week"]
    elif strategy == 'SAFETY_MAX': feats = ["lag_1", "Bell_Curve", "Sin_Week"]
    elif strategy == 'SMOOTH_OPERATOR': feats = ["lag_1", "rolling_mean_4", "velocity"]
    else: feats = ["lag_1", "Bell_Curve"]
    
    last_date = df_ml['Bulan'].max()
    future_vals = []

    for i in range(steps):
        next_date = last_date + pd.Timedelta(weeks=2*(i+1))
        f_bell = get_hijri_bell_curve(next_date)
        if strategy == 'SEASONAL_DEEP' and f_bell > 0.3: curr_lag_1 += 0.15 
        wn = next_date.isocalendar().week
        f_sin = np.sin(2 * np.pi * wn / 52)
        f_cos = np.cos(2 * np.pi * wn / 52)
        curr_velocity = curr_lag_1 - curr_lag_2
        
        # Clipping Velocity
        curr_velocity = max(-0.5, min(0.5, curr_velocity))

        input_dict = {}
        for f in feats:
            if f == 'lag_1': input_dict[f] = curr_lag_1
            elif f == 'lag_2': input_dict[f] = curr_lag_2
            elif f == 'Bell_Curve': input_dict[f] = f_bell
            elif f == 'Sin_Week': input_dict[f] = f_sin
            elif f == 'Cos_Week': input_dict[f] = f_cos
            elif f == 'velocity': input_dict[f] = curr_velocity
            elif f == 'rolling_mean_4': input_dict[f] = curr_roll_4
            else: input_dict[f] = 0
            
        if i < len(warmup_vals):
            val_linear = warmup_vals[i]
            if val_linear < 1.0: val_linear = 1.0
            p_log_used = np.log1p(val_linear)
        else:
            in_df = pd.DataFrame([input_dict])[feats]
            p_log_raw = model.predict(in_df)[0]
            val_linear = max(0, np.expm1(p_log_raw))
            # Safety Net
            if val_linear < 1.0: val_linear = max(1.0, np.expm1(curr_lag_1) * 0.5)
            p_log_used = np.log1p(val_linear)

        future_vals.append(val_linear)
        curr_lag_2 = curr_lag_1
        curr_lag_1 = p_log_used
        curr_roll_4 = (curr_roll_4 * 3 + p_log_used) / 4

    try:
        del model
        gc.collect()
    except:
        pass
        
    return pd.Series(dtype=float), 0, future_vals

def run_lstm_inference(df_ml, cat, strategy, steps=4, use_temp=False):
    # 1. Cari File (Cek Temp dulu, kalau gak ada baru Asli)
    h5_path = find_model_file("LSTM", cat, ".h5", use_temp=use_temp)
    sx_path = find_model_file("Scaler_X", cat, ".pkl", use_temp=use_temp)
    sy_path = find_model_file("Scaler_y", cat, ".pkl", use_temp=use_temp)
    
    # Fallback Logic: Kalau di Temp gak lengkap, pinjam Asli
    if use_temp and not (h5_path and sx_path and sy_path):
        h5_path = find_model_file("LSTM", cat, ".h5", use_temp=False)
        sx_path = find_model_file("Scaler_X", cat, ".pkl", use_temp=False)
        sy_path = find_model_file("Scaler_y", cat, ".pkl", use_temp=False)

    if not (h5_path and sx_path and sy_path): return None, 0, None
    K.clear_session(); gc.collect()
    
    # Ambil Warmup (Kecuali mode temp/upload, kita kosongin biar fresh)
    warmup_vals = [] if use_temp else get_warmup_data(cat, 'Model Canggih')
    
    try:
        model = load_saved_model(h5_path, 'keras') 
        scaler_X = load_saved_model(sx_path, 'joblib')
        scaler_y = load_saved_model(sy_path, 'joblib')
        
        # Setup Features
        if strategy == 'SEASONAL_DEEP': feats = ["lag_1", "lag_2", "Bell_Curve", "Sin_Week"]
        elif strategy == 'SET_CLASSIC': feats = ["lag_1", "lag_2", "velocity", "Bell_Curve"]
        elif strategy == 'MAE_STABLE': feats = ["lag_1", "lag_2", "Bell_Curve", "Sin_Week", "Cos_Week"]
        elif strategy == 'SAFETY_MAX': feats = ["lag_1", "Bell_Curve", "Sin_Week"]
        elif strategy == 'SMOOTH_OPERATOR': feats = ["lag_1", "rolling_mean_4", "velocity"]
        else: feats = ["lag_1", "Bell_Curve"]
        
        for c in feats: 
            if c not in df_ml.columns: df_ml[c] = 0.0
            
        X_data = df_ml[feats].values.astype(np.float64)
        X_scaled = scaler_X.transform(X_data)
        
        sx_min = scaler_X.data_min_
        sx_range = scaler_X.data_range_
        idx_map = {name: i for i, name in enumerate(feats)}
        
        TIME_STEPS = 3
        if len(X_scaled) < TIME_STEPS:
            padding = np.tile(X_scaled[-1], (TIME_STEPS - len(X_scaled), 1))
            X_scaled = np.vstack([padding, X_scaled])

        current_seq = X_scaled[-TIME_STEPS:].copy()
        
        last_row = df_ml.iloc[-1].fillna(0)
        curr_lag_1 = float(last_row.get('lag_1', 0))
        curr_lag_2 = float(last_row.get('lag_2', curr_lag_1))
        curr_roll_4 = float(last_row.get('rolling_mean_4', curr_lag_1))
        
        last_date = df_ml['Bulan'].max()
        future_vals = []

        for i in range(steps):
            next_date = last_date + pd.Timedelta(weeks=2*(i+1))
            
            # --- ESTAFET ---
            if i < len(warmup_vals):
                val_linear = warmup_vals[i]
                if val_linear < 1.0: val_linear = 1.0 
                y_log_used = np.log1p(val_linear) # <-- INI ADA
            else:
                # --- FIX LSTM RAM LEAK ---
                input_tensor = current_seq.reshape(1, TIME_STEPS, len(feats))
                input_tf = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
                y_scaled_raw = float(model(input_tf, training=False)[0][0])
                
                y_log_real = scaler_y.inverse_transform([[y_scaled_raw]])[0][0]
                val_linear = max(0, np.expm1(y_log_real))
                
                # --- SAFETY NET (ANTI-0) ---
                if val_linear < 1.0: val_linear = max(1.0, np.expm1(curr_lag_1) * 0.5)
                
                # ✅ PERBAIKAN: HITUNG BALIK LOG-NYA DI SINI
                y_log_used = np.log1p(val_linear) 
            
            future_vals.append(val_linear)
            
            # Update State
            curr_lag_2 = curr_lag_1
            curr_lag_1 = y_log_used # <-- Error tadi di sini karena y_log_used belum ada
            curr_roll_4 = (curr_roll_4 * 3 + y_log_used) / 4
            
            new_row = current_seq[-1].copy()
            
            # Booster & Update State Features
            bell_check = get_hijri_bell_curve(next_date)
            boost_factor = 0.0
            if strategy == 'SEASONAL_DEEP' and bell_check > 0.3: boost_factor = 0.15

            if 'lag_1' in idx_map: new_row[idx_map['lag_1']] = ((curr_lag_1 + boost_factor) - sx_min[idx_map['lag_1']]) / (sx_range[idx_map['lag_1']] + 1e-9)
            if 'lag_2' in idx_map: new_row[idx_map['lag_2']] = (curr_lag_2 - sx_min[idx_map['lag_2']]) / (sx_range[idx_map['lag_2']] + 1e-9)
            if 'velocity' in idx_map:
                val_vel = curr_lag_1 - curr_lag_2
                new_row[idx_map['velocity']] = (val_vel - sx_min[idx_map['velocity']]) / (sx_range[idx_map['velocity']] + 1e-9)
            if 'rolling_mean_4' in idx_map:
                new_row[idx_map['rolling_mean_4']] = (curr_roll_4 - sx_min[idx_map['rolling_mean_4']]) / (sx_range[idx_map['rolling_mean_4']] + 1e-9)
            
            if 'Bell_Curve' in idx_map: new_row[idx_map['Bell_Curve']] = (bell_check - sx_min[idx_map['Bell_Curve']]) / (sx_range[idx_map['Bell_Curve']] + 1e-9)
            if 'Sin_Week' in idx_map:
                new_row[idx_map['Sin_Week']] = (np.sin(2*np.pi*next_date.isocalendar().week/52) - sx_min[idx_map['Sin_Week']]) / (sx_range[idx_map['Sin_Week']] + 1e-9)
            if 'Cos_Week' in idx_map:
                new_row[idx_map['Cos_Week']] = (np.cos(2*np.pi*next_date.isocalendar().week/52) - sx_min[idx_map['Cos_Week']]) / (sx_range[idx_map['Cos_Week']] + 1e-9)
            
            current_seq = np.vstack([current_seq[1:], new_row])

        try:
            del model, scaler_X, scaler_y
            K.clear_session()
            gc.collect()
        except:
            pass

        return pd.Series(dtype=float), 0, future_vals
        
    except Exception as e: 
        print(f"🔴 ERROR LSTM {cat}: {e}")
        return None, 0, []

def process_single_category(idx, cats, agg, steps_val=4, target_model=None):
    cat = cats[idx]
    result_buffer = []
    
    try:
        # Cek ketersediaan data raw
        c_data = agg[agg['Kategori'] == cat]
        if len(c_data) < 5:
            # print(f"⚠️ Skip {cat}: Data kurang (<5 baris)")
            return result_buffer
        
        # --- [FIX] PRIORITASKAN CACHE ---
        # Jangan build ulang df_ml dari nol, ambil yang sudah ada di session_state
        # agar konsisten dengan saat Load/Upload.
        df_ml = None
        df_base = None
        strat = STRATEGY_MAP.get(cat, 'MAE_STABLE')
        
        if 'cache_df_ml' in st.session_state and cat in st.session_state['cache_df_ml']:
            df_ml = st.session_state['cache_df_ml'][cat]
            # df_base turunan dari df_ml (untuk SARIMA)
            if 'Total_Jumlah' in df_ml.columns:
                df_base = df_ml[['Bulan', 'Total_Jumlah', 'Bell_Curve']].copy()
            else:
                # Fallback kalau kolom Total_Jumlah hilang (karena smoothing)
                # Kita rekonstruksi dari Target_Log atau ambil dari c_data
                df_base = c_data.copy() 
                # Pastikan Bell Curve ada
                df_base['Bell_Curve'] = df_base['Bulan'].apply(get_hijri_bell_curve)
                
            # print(f"✅ Pakai Cache: {cat}")
        else:
            # Fallback (Hanya jika cache rusak/hilang)
            # print(f"⚠️ Cache Miss: {cat} -> Rebuild")
            df_base, df_ml, strat = prepare_features_for_inference(c_data, cat)
        
        # Validasi Final sebelum masuk mesin
        if df_ml is None or df_ml.empty: return result_buffer
        if 'Target_Log' not in df_ml.columns: return result_buffer
        
        # --- LOGIKA HEMAT MEMORI ---
        run_all = target_model is None
        
        # 1. SARIMA (Model Klasik)
        if run_all or target_model == 'Model Klasik':
            # Gunakan df_base yang valid
            if df_base is not None and not df_base.empty:
                p_sarima, s_sarima, f_sarima = run_sarima_inference(df_base, cat, strat, steps=steps_val)
                
                # Generate tanggal masa depan
                last_date = df_base['Bulan'].max()
                # f_sarima di sini sudah FULL OUTPUT (Warmup + Baru)
                # Jadi panjang tanggal harus menyesuaikan panjang f_sarima
                # PENTING: f_sarima panjangnya = steps (karena sudah dipotong di fungsi inference)
                future_dates = pd.date_range(start=last_date, periods=len(f_sarima)+1, freq='2W')[1:]
                
                if f_sarima is not None and len(f_sarima) == len(future_dates):
                    result_buffer.append(pd.DataFrame({
                        'Tanggal': future_dates, 
                        'Nilai': f_sarima, 
                        'Model': 'Model Klasik', 
                        'Kategori': cat, 
                        'SMAPE': 0, 
                        'Jenis': 'Future'
                    }))
                try:
                    del p_sarima, s_sarima, f_sarima
                    gc.collect()
                except:
                    pass
        
        # 2. XGBOOST (Model Pintar)
        if run_all or target_model == 'Model Pintar':
            p_xgb, s_xgb, f_xgb = run_xgboost_inference(df_ml, cat, strat, steps=steps_val)
            
            last_date = df_ml['Bulan'].max()
            # f_xgb sudah dipotong sesuai steps
            future_dates = pd.date_range(start=last_date, periods=len(f_xgb)+1, freq='2W')[1:]
            
            if f_xgb is not None and len(f_xgb) == len(future_dates):
                result_buffer.append(pd.DataFrame({
                    'Tanggal': future_dates, 
                    'Nilai': f_xgb, 
                    'Model': 'Model Pintar', 
                    'Kategori': cat, 
                    'SMAPE': 0, 
                    'Jenis': 'Future'
                }))
            try:
                del p_xgb, s_xgb, f_xgb
                gc.collect()
            except:
                pass
        
        # 3. LSTM (Model Canggih)
        if run_all or target_model == 'Model Canggih':
            p_lstm, s_lstm, f_lstm = run_lstm_inference(df_ml, cat, strat, steps=steps_val)
            K.clear_session()
            
            last_date = df_ml['Bulan'].max()
            future_dates = pd.date_range(start=last_date, periods=len(f_lstm)+1, freq='2W')[1:]
            
            if f_lstm is not None and len(f_lstm) == len(future_dates):
                result_buffer.append(pd.DataFrame({
                    'Tanggal': future_dates, 
                    'Nilai': f_lstm, 
                    'Model': 'Model Canggih', 
                    'Kategori': cat, 
                    'SMAPE': 0, 
                    'Jenis': 'Future'
                }))
            try:
                del p_lstm, s_lstm, f_lstm
                K.clear_session()
                gc.collect()
            except:
                pass
        
    except Exception as e: 
        print(f"❌ ERROR processing {cat}: {str(e)}")
    finally:
        # ✅ FINAL CLEANUP (ALWAYS RUN)
        try:
            K.clear_session()
            gc.collect()
        except:
            pass
    return result_buffer

# =========================================================
# 4. SIDEBAR NAVIGASI
# =========================================================

with st.sidebar:
    # Membagi kolom dengan rasio 1:4
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.write("###") # Spacer tipis agar gambar agak turun sejajar teks
        st.image("https://cdn-icons-png.flaticon.com/512/2921/2921222.png", width=50)
        
    with col2:
        st.markdown("## KawanUMKM") # Pakai markdown ## biar marginnya tidak terlalu lebar
        st.caption("Asisten Bisnis Cerdas")
        
    st.markdown("---")
    
    if st.button("📊 Ringkasan Prediksi", use_container_width=True, type="primary" if st.session_state['page'] == "Ringkasan Prediksi" else "secondary"):
        st.session_state['page'] = "Ringkasan Prediksi"
        st.rerun()
    if st.button("📈 Detail Kategori", use_container_width=True, type="primary" if st.session_state['page'] == "Detail Kategori" else "secondary"):
        st.session_state['page'] = "Detail Kategori"
        st.rerun()
    if st.button("⚖️ Perbandingan Model", use_container_width=True, type="primary" if st.session_state['page'] == "Perbandingan Model" else "secondary"):
        st.session_state['page'] = "Perbandingan Model"
        st.rerun()
    if st.button("📂 Data Mentah", use_container_width=True, type="primary" if st.session_state['page'] == "Data Mentah" else "secondary"):
        st.session_state['page'] = "Data Mentah"
        st.rerun()
    if st.button("📘 Panduan Pengguna", use_container_width=True, type="primary" if st.session_state['page'] == "Panduan Pengguna" else "secondary"):
        st.session_state['page'] = "Panduan Pengguna"
        st.rerun()
    st.markdown("---")
    # ✅ TAMPILKAN MEMORY USAGE
    mem_usage = get_memory_usage()
    mem_percent = psutil.virtual_memory().percent

    # if mem_percent > 80:
    #     st.error(f"⚠️ Memory: {mem_usage:.0f}MB ({mem_percent:.0f}%)")
    # elif mem_percent > 60:
    #     st.warning(f"💾 Memory: {mem_usage:.0f}MB ({mem_percent:.0f}%)")
    # else:
    #     st.info(f"✅ Memory: {mem_usage:.0f}MB ({mem_percent:.0f}%)")
    # st.markdown("---")
    st.markdown("### 📥 Input Data")
    st.caption("⚠️ **Syarat:** Data minimal 3 tahun (untuk menangkap pola musiman & tren jangka panjang).")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")
    if st.button("🔄 Reset System"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.cache_resource.clear()
        st.rerun()

# =========================================================
# 5. LOGIKA DATA LOADER (4 KOLOM: TGL, ID, ITEM, QTY)
# =========================================================

# Fungsi parsing khusus upload (bytes)
def repair_and_load_bytes(uploaded_file):
    content = uploaded_file.getvalue().decode('utf-8', errors='replace')
    lines = content.splitlines()
    if len(lines) < 2: return None
    header = lines[0].strip().split(',')
    expected_cols = len(header)
    repaired_lines = [lines[0].strip()]
    for i, line in enumerate(lines[1:]):
        clean_line = line.replace('"', '').strip()
        parts = clean_line.split(',')
        if len(parts) == expected_cols: repaired_lines.append(clean_line)
        elif len(parts) > expected_cols:
            head_part = parts[:3]
            tail_count = expected_cols - 4
            tail_part = parts[-tail_count:]
            middle_part = [" ".join(parts[3 : -tail_count])]
            repaired_lines.append(",".join(head_part + middle_part + tail_part))
        else:
            while len(parts) < expected_cols: parts.append('')
            repaired_lines.append(",".join(parts))
    return pd.read_csv(io.StringIO("\n".join(repaired_lines)))

# Inisialisasi
df_clean = None
default_exists = os.path.exists("database_prediksi_final.csv")

# A. CEK UPLOAD USER
if uploaded_file:
    # --- LOGIKA BARU: CEK STATUS PROSES ---
    # Buat ID unik untuk file ini (Gabungan Nama + Ukuran)
    file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
    
    # Cek apakah file ini SUDAH selesai diproses sebelumnya?
    is_processed = False
    if 'processed_file_id' in st.session_state:
        if st.session_state['processed_file_id'] == file_signature:
            is_processed = True

    # --- SKENARIO 1: SUDAH DIPROSES (TAMPILKAN DASHBOARD) ---
    if is_processed:
        st.success(f"✅ Data '{uploaded_file.name}' aktif digunakan.")
        
        # Tombol kecil jika ingin membatalkan/upload ulang
        if st.button("🔄 Ganti File / Setting Ulang Kolom"):
            del st.session_state['processed_file_id'] # Hapus tanda lunas
            st.rerun()
            
        # PENTING: Tidak ada st.stop() di sini, jadi kode lanjut ke bawah (Dashboard)

    # --- SKENARIO 2: BELUM DIPROSES (TAMPILKAN SETTINGAN & STOP) ---
    else:
        # Load Raw Data
        if 'raw_df_v20' not in st.session_state or st.session_state.get('current_file_ref') != file_signature:
            st.session_state['raw_df_v20'] = repair_and_load_bytes(uploaded_file)
            st.session_state['current_file_ref'] = file_signature
            
        df_raw = st.session_state['raw_df_v20']
        
        # Deteksi Kolom Otomatis
        all_cols = list(df_raw.columns)
        guess_date = next((c for c in all_cols if 'tgl' in c.lower() or 'date' in c.lower() or 'time' in c.lower()), all_cols[0])
        guess_id = next((c for c in all_cols if 'kode' in c.lower() or 'id' in c.lower() or 'inv' in c.lower() or 'no' in c.lower() or 'transaksi' in c.lower()), all_cols[1] if len(all_cols)>1 else None)
        guess_item = next((c for c in all_cols if ('nama' in c.lower() or 'prod' in c.lower() or 'desc' in c.lower()) and not ('kode' in c.lower() or 'id' in c.lower() or 'sku' in c.lower())), None)
        if not guess_item: guess_item = next((c for c in all_cols if ('barang' in c.lower() or 'item' in c.lower()) and not ('kode' in c.lower() or 'id' in c.lower() or 'sku' in c.lower())), None)
        if not guess_item: guess_item = next((c for c in all_cols if 'nama' in c.lower() or 'barang' in c.lower() or 'prod' in c.lower() or 'item' in c.lower()), all_cols[2] if len(all_cols)>2 else None)
        guess_qty = next((c for c in all_cols if 'jumlah' in c.lower() or 'qty' in c.lower() or 'kuantitas' in c.lower()), all_cols[3] if len(all_cols)>3 else None)
        
        # UI Pengaturan Kolom
        with st.expander("🛠️ Pengaturan Kolom Upload", expanded=True):
            st.info("Sistem mencoba menebak kolom Anda. Mohon koreksi jika salah.")
            c1, c2, c3, c4 = st.columns(4)
            with c1: sel_date = st.selectbox("📅 Kolom Tanggal", all_cols, index=all_cols.index(guess_date) if guess_date in all_cols else 0)
            with c2: sel_id = st.selectbox("🔢 Kode Transaksi", all_cols, index=all_cols.index(guess_id) if guess_id in all_cols else 1)
            with c3: sel_item = st.selectbox("📦 Nama Barang", all_cols, index=all_cols.index(guess_item) if guess_item in all_cols else 2)
            with c4: sel_qty = st.selectbox("📊 Jumlah/Qty", all_cols, index=all_cols.index(guess_qty) if guess_qty in all_cols else 3)
            
            if st.button("✅ Proses & Latih Ulang", type="primary", use_container_width=True):
                # 1. Bersihkan Data
                df_proc = df_raw.copy()
                df_proc['Tanggal'] = smart_date_parser(df_proc, sel_date, id_col=sel_id)
                df_proc = df_proc.dropna(subset=['Tanggal'])
                df_proc['Nama Barang'] = df_proc[sel_item].astype(str)
                df_proc['Total_Jumlah'] = pd.to_numeric(df_proc[sel_qty], errors='coerce').fillna(0)
                df_proc['Kategori'] = df_proc['Nama Barang'].apply(categorize_item)
                df_proc = df_proc[df_proc['Kategori'].isin(VALID_CATEGORIES)]
                
                if df_proc.empty:
                    st.error("Data tidak valid atau kategori tidak dikenali.")
                else:
                    # Satpam Data
                    temp_agg = df_proc.groupby([pd.Grouper(key='Tanggal', freq='2W')])['Total_Jumlah'].sum()
                    total_periode = len(temp_agg)
                    durasi_tahun = total_periode / 26 
                    st.write(f"📊 **Analisa Data:** Terdeteksi **{total_periode} periode** (±{durasi_tahun:.1f} tahun).")
                    
                    if total_periode < 60:
                        st.error(f"❌ **DATA TIDAK CUKUP** (Butuh Min. 60 Periode / 3 Tahun). Terdeteksi hanya {total_periode}.")
                    else:
                        # Lanjut Training
                        st.session_state['data_clean'] = df_proc
                        with st.spinner("⏳ Sedang mempelajari pola data baru Anda... (1-2 menit)"):
                            if not os.path.exists("models"): os.makedirs("models")
                            df_result_new = train_models_on_upload(df_proc)
                            
                            if not df_result_new.empty:
                                st.session_state['data_ready'] = df_result_new
                                
                                # ✅ KUNCI PENTING: TANDAI BAHWA FILE INI SUDAH DIPROSES
                                st.session_state['processed_file_id'] = file_signature 
                                
                                st.success("✅ Selesai!")
                                time.sleep(1)
                                st.rerun() # Refresh agar masuk ke blok 'is_processed'
                            else:
                                st.error("Gagal melatih model.")

        # REM TANGAN: Hentikan dashboard jika belum diproses
        st.warning("👆 Silakan sesuaikan kolom di atas lalu klik tombol **'Proses'**.")
        st.stop()
# B. # B. LOAD DEFAULT (JIKA TIDAK UPLOAD)
elif default_exists:
    # Load 'data_ready' (Hasil Prediksi Jadi)
    if 'data_ready' not in st.session_state:
        try:
            df_def = pd.read_csv("database_prediksi_final.csv")
            df_def['Tanggal'] = pd.to_datetime(df_def['Tanggal'])
            st.session_state['data_ready'] = df_def
        except Exception as e:
            st.error(f"Gagal memuat database default: {e}")
            st.stop()
    
    # Load 'data_clean' untuk halaman data mentah
    if 'data_clean' not in st.session_state:
        df_full = st.session_state['data_ready']
        df_hist = df_full[df_full['Jenis'] == 'Aktual'].copy()
        
        # ✅ PERBAIKAN KRUSIAL: Hapus duplikat agar data smoothing tidak hancur!
        df_hist = df_hist.drop_duplicates(subset=['Kategori', 'Tanggal'])
        
        # Mapping nama kolom agar konsisten
        if 'Total_Jumlah' not in df_hist.columns and 'Nilai' in df_hist.columns:
            df_hist = df_hist.rename(columns={'Nilai': 'Total_Jumlah'})
            
        st.session_state['data_clean'] = df_hist
    
    # --- [FIX KRUSIAL] POPULATE CACHE UNTUK INFERENCE ---
    # Ini menjamin mesin forecast punya "bahan bakar" saat user minta extend
    if 'cache_df_ml' not in st.session_state:
        print("\n🔄 SYSTEM: Membangun Cache dari Data Default...")
        st.session_state['cache_df_ml'] = {}
        
        # Ambil data historis mentah dari 'data_clean'
        # Pastikan kita punya kolom yang benar
        df_source = st.session_state['data_clean'].copy()
        
        # Normalisasi nama kolom
        if 'Tanggal' in df_source.columns and 'Bulan' not in df_source.columns:
            df_source['Bulan'] = df_source['Tanggal']
            
        # Group by kategori
        unique_cats = df_source['Kategori'].unique()
        
        for cat in unique_cats:
            # Filter data kategori
            df_cat_raw = df_source[df_source['Kategori'] == cat].copy()
            
            # Pastikan urut & reset index
            df_cat_raw = df_cat_raw.sort_values('Bulan').reset_index(drop=True)
            
            # Jalankan Feature Engineering (Sama persis kayak upload)
            # Ini akan menghasilkan 'Target_Log', 'lag_1', 'Bell_Curve', dll.
            strategy = STRATEGY_MAP.get(cat, 'MAE_STABLE')
            
            try:
                df_ml = apply_feature_engineering(df_cat_raw, strategy)
                
                # Simpan ke cache session state
                st.session_state['cache_df_ml'][cat] = df_ml
                print(f"  ✅ Cache Built: {cat} ({len(df_ml)} rows)")
            except Exception as e:
                print(f"  ❌ Cache Failed: {cat} - {e}")

# Ambil data final untuk dipakai di halaman bawah
df_final = st.session_state.get('data_ready', None)

if df_final is not None:
    name_map = {
        'SARIMA': 'Model Klasik', 'XGBoost': 'Model Pintar', 'LSTM': 'Model Canggih',
        'Metode Statistik': 'Model Klasik', 'Machine Learning': 'Model Pintar', 'Deep Learning': 'Model Canggih'
    }
    df_final['Model'] = df_final['Model'].replace(name_map)
    st.session_state['data_ready'] = df_final

# =========================================================
# 6. ROUTING HALAMAN
# =========================================================

if df_final is None:
    st.markdown("### 👋 Selamat Datang")
    st.info("Silakan upload file CSV transaksi di sidebar sebelah kiri.")

# =========================================================
# 6.A PAGE 1: RINGKASAN PREDIKSI (FINAL: FIXED CALCULATION)
# =========================================================
elif st.session_state['page'] == "Ringkasan Prediksi":
    # log_memory("START Page 1")
    
    # --- 0. CSS STYLE ---
    st.markdown("""
    <style>
    div.kpi-container {
        background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px;
        text-align: center; height: 140px; display: flex; flex-direction: column; justify-content: center; align-items: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .kpi-label { font-size: 13px; color: #6c757d; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; }
    .kpi-value-base { font-weight: 700; color: #212529; margin: 0; line-height: 1.2; }
    .kpi-sub { font-size: 13px; margin-top: 8px; }
    </style>
    """, unsafe_allow_html=True)
    
    # 2. HEADER
    st.title("📊 Ringkasan Prediksi Bisnis")
    st.markdown("Gambaran performa toko untuk periode mendatang berdasarkan data historis.")
    st.divider()
    
    # 3. SIAPKAN BATAS DATA
    # Ambil max date dari data Aktual
    max_hist_date = df_final.loc[df_final['Jenis']=='Aktual', 'Tanggal'].max()
    
    # 4. UI CONTROLS
    c_filter_1, c_filter_2 = st.columns([1, 1])
    
    with c_filter_1:
        # --- PERBAIKAN FINAL: JANGAN FILTER BERDASARKAN DATA YANG ADA ---
        # Kita paksa semua opsi muncul agar user BISA memilihnya (untuk memicu training)
        model_opts = ['Model Pintar', 'Model Canggih', 'Model Klasik']
        
        sel_model = st.selectbox("🤖 Pilih Model", model_opts, index=0, key="sb_model_selector")

        # --- LOGIKA LAZY LOADING (PEMICU MASAK DADAKAN) ---
        # Cek: Jika User pilih Model Canggih/Klasik TAPI datanya belum ada -> LATIH SEKARANG!
        if sel_model in ["Model Canggih", "Model Klasik"]:
            
            # Cek di dataframe apakah data Future untuk model ini sudah ada?
            cek_data = df_final[(df_final['Model'] == sel_model) & (df_final['Jenis'] == 'Future')]
            
            # Kalo KOSONG, berarti belum dimasak. GAS MASAK SEKARANG!
            if cek_data.empty:
                with st.spinner(f"👩‍🍳 Sedang menyiapkan {sel_model}... Harap tunggu sebentar..."):
                    # Panggil fungsi training dadakan
                    new_results = train_specific_model(sel_model)
                    
                    if not new_results.empty:
                        # Gabungkan hasil baru ke data utama di session state
                        st.session_state['data_ready'] = pd.concat([st.session_state['data_ready'], new_results], ignore_index=True)
                        st.success(f"✅ {sel_model} Selesai!")
                        st.rerun() # Refresh halaman biar grafik nongol
                    else:
                        st.warning(f"Gagal memproses {sel_model}.")
    # ------------------------------------------------
    
    # ... (Lanjut ke kode visualisasi grafik) ...

    # Cek ketersediaan data masa depan khusus model ini
    df_fut_check = df_final[(df_final['Jenis']=='Future') & (df_final['Model']==sel_model)]
    if not df_fut_check.empty:
        max_future_avail = df_fut_check['Tanggal'].max()
    else:
        max_future_avail = max_hist_date

    with c_filter_2:
        def_start = max_hist_date + pd.Timedelta(days=1)
        def_end = max_hist_date + pd.Timedelta(weeks=2)
        date_range = st.date_input("📅 Pilih Rentang Periode", value=(def_start, def_end), min_value=df_final['Tanggal'].min(), key="dp_range_picker")

    # 5. LOGIKA AUTO-RUN (HEMAT MEMORI)
    u_start, u_end = def_start, def_end
    is_ready = False
    if isinstance(date_range, tuple):
        if len(date_range) == 2:
            u_start, u_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            is_ready = True
        elif len(date_range) == 1:
            st.caption("👇 *Klik tanggal akhir untuk memuat data.*")
            st.stop()
    else:
        u_start, u_end = pd.to_datetime(date_range), pd.to_datetime(date_range)
        is_ready = True

    # BLOK 1: BIKIN FLAG (HANYA 1 KALI)
    if 'processing_forecast' not in st.session_state:
        st.session_state['processing_forecast'] = False

    # BLOK 2: SIMPAN INPUT TERAKHIR (HANYA 1 KALI)
    if 'last_model' not in st.session_state:
        st.session_state['last_model'] = sel_model
    if 'last_date' not in st.session_state:
        st.session_state['last_date'] = u_end

    # BLOK 3: CEK APAKAH USER GANTI INPUT
    if st.session_state['last_model'] != sel_model or st.session_state['last_date'] != u_end:
        # User ganti input, reset flag biar bisa proses lagi
        st.session_state['processing_forecast'] = False
        st.session_state['last_model'] = sel_model
        st.session_state['last_date'] = u_end

    if is_ready and u_end > max_future_avail and not st.session_state['processing_forecast']:
        # log_memory("START Extended Forecast")
        st.session_state['processing_forecast'] = True
        steps_needed = calculate_steps_to_date(max_hist_date, u_end)
        # if steps_needed > 52: steps_needed = 52 # Safety limit

        import time
        start_time = time.time()
        TIMEOUT_SECONDS = 180
            
        progress_text = f"Menghitung prediksi {sel_model}..."
        my_bar = st.progress(0, text=progress_text)
        
        # Ambil base data (Pilih Aktual yang sesuai dengan Model)
        # Ini mencegah pencampuran Raw & Smooth saat training ulang
        req_type = 'Aktual (Smooth)' if sel_model in ['Model Pintar', 'Model Canggih'] else 'Aktual (Raw)'
        df_hist_base = df_final[(df_final['Jenis']=='Aktual') & (df_final['Model']==req_type)].copy()
        
        # Fallback jika data spesifik belum ada, ambil apa saja yang Aktual
        if df_hist_base.empty:
             df_hist_base = df_final[df_final['Jenis']=='Aktual'].copy()

        # Rename & Aggregate
        df_hist_base = df_hist_base[['Kategori', 'Tanggal', 'Nilai']].rename(columns={'Nilai': 'Total_Jumlah', 'Tanggal': 'Bulan'})
        agg_re = df_hist_base.groupby(['Kategori', 'Bulan'])['Total_Jumlah'].mean().reset_index()
        unique_cats = agg_re['Kategori'].unique()
        
        new_results = []
        try:
            for idx, cat in enumerate(unique_cats):
                # log_memory(f"  Process {cat}")
                if time.time() - start_time > TIMEOUT_SECONDS:
                    my_bar.empty()
                    st.warning("⏱️ Proses terlalu lama, menampilkan hasil parsial...")
                    break
                my_bar.progress(int(((idx + 1) / len(unique_cats)) * 100), text=f"Memproses {cat} ({idx+1}/{len(unique_cats)})")
                # --- PENTING: Pass target_model agar tidak menjalankan semua model ---
                res = process_single_category(idx, unique_cats, agg_re, steps_val=steps_needed, target_model=sel_model)
                
                # Ambil future saja
                res_future = [d for d in res if not d.empty and d['Jenis'].iloc[0] == 'Future']
                new_results.extend(res_future)
                
                del res
                gc.collect()

            my_bar.empty()

            if new_results:
                with st.spinner("Menyimpan data..."):
                    df_calc_result = pd.concat(new_results, ignore_index=True)
                    
                    # --- LOGIKA CERDAS: PRESERVE CSV DATA ---
                    # 1. Ambil data lama SELAIN Future model ini (aman)
                    df_base_others = df_final[~((df_final['Jenis'] == 'Future') & (df_final['Model'] == sel_model))].copy()
                    
                    # 2. Ambil data Future model ini yang SUDAH ADA (Data CSV/Colab)
                    df_existing_future = df_final[(df_final['Jenis'] == 'Future') & (df_final['Model'] == sel_model)].copy()
                    
                    # 3. Filter hasil baru: Hanya ambil tanggal yang BELUM ADA di data CSV
                    #    Ini kuncinya! Data periode 1-4 dari CSV tidak akan tertimpa.
                    existing_dates = df_existing_future['Tanggal'].unique()
                    df_new_only = df_calc_result[~df_calc_result['Tanggal'].isin(existing_dates)]
                    
                    # 4. Gabungkan: Base + Future Lama (CSV) + Future Baru (Extended)
                    df_updated = pd.concat([df_base_others, df_existing_future, df_new_only], ignore_index=True)
                    
                    # Clean up & Save
                    del st.session_state['data_ready']
                    del df_final
                    gc.collect()
                    
                    st.session_state['data_ready'] = df_updated.sort_values('Tanggal')
                    st.session_state['processing_forecast'] = False
                    st.rerun()
            else:
                # ✅ TURUNKAN FLAG KALAU GAGAL
                st.session_state['processing_forecast'] = False 
        except Exception as e:
            st.error(f"Error memori: {str(e)}")
            st.stop()
        # log_memory("END Extended Forecast")

    # 6. FILTER VISUALISASI
    df_filtered = df_final[(df_final['Tanggal'] >= u_start) & (df_final['Tanggal'] <= u_end)].copy()
    df_view = df_filtered[(df_filtered['Jenis'] == 'Future') & (df_filtered['Model'] == sel_model)].copy()
    
    if not df_view.empty:
        df_view = df_view.sort_values('Tanggal')
        df_view['Start Date'] = df_view['Tanggal'] - pd.Timedelta(days=13)
        df_view['Periode Label'] = (df_view['Start Date'].dt.strftime('%d %b') + " - " + df_view['Tanggal'].dt.strftime('%d %b %Y'))

    # 7. KPI CALCULATION (FIX: ANTI DOUBLE COUNTING)
    total_forecast = df_view['Nilai'].sum() if not df_view.empty else 0
    
    # Pilih data history yang spesifik
    target_hist_model = 'Aktual (Smooth)' if sel_model in ['Model Pintar', 'Model Canggih'] else 'Aktual (Raw)'
    df_benchmark = df_final[(df_final['Jenis'] == 'Aktual') & (df_final['Model'] == target_hist_model)].copy()
    
    # Fallback
    if df_benchmark.empty: 
        df_benchmark = df_final[df_final['Jenis'] == 'Aktual'].copy()
    
    # --- FIX UTAMA: HAPUS DUPLIKAT SEBELUM DIHITUNG ---
    # Ini mencegah history terhitung 2x atau 3x
    df_benchmark = df_benchmark.drop_duplicates(subset=['Kategori', 'Tanggal'])
    
    total_past_benchmark = 0
    if not df_benchmark.empty:
        # Hitung rata-rata volume per 2 minggu dari 3 bulan terakhir (6 periode)
        avg_vol_per_period = df_benchmark.groupby('Tanggal')['Nilai'].sum().tail(6).mean()
        
        # Benchmark = Rata-rata x Jumlah periode yang dilihat user
        num_periods_view = df_view['Tanggal'].nunique()
        if num_periods_view == 0: num_periods_view = 1
        
        total_past_benchmark = avg_vol_per_period * num_periods_view
    
    growth = 0
    if total_past_benchmark > 0: 
        growth = ((total_forecast - total_past_benchmark) / total_past_benchmark) * 100
    
    trend_arrow, trend_color = ("▲ Naik", "#16a34a") if growth >= 0 else ("▼ Turun", "#dc2626")

    best_cat_name, best_cat_vol = "-", 0
    if not df_view.empty:
        # Gunakan nlargest (lebih cepat dari sort)
        top_cat_row = df_view.groupby('Kategori')['Nilai'].sum().nlargest(1)
        if not top_cat_row.empty:
            best_cat_name, best_cat_vol = top_cat_row.index[0], top_cat_row.values[0]

    # Akurasi Statis
    error_val = calculate_metrics(df_final, sel_model)
    accuracy_score = max(0, 100 - error_val)
    if accuracy_score >= 80: stars, color_acc, msg_acc = "⭐⭐⭐⭐⭐", "#16a34a", "Sangat Tinggi"
    elif accuracy_score >= 60: stars, color_acc, msg_acc = "⭐⭐⭐⭐", "#16a34a", "Tinggi"
    elif accuracy_score >= 40: stars, color_acc, msg_acc = "⭐⭐⭐", "#d97706", "Cukup"
    else: stars, color_acc, msg_acc = "⭐⭐", "#dc2626", "Rendah"

    cat_font_size = "22px" 
    if len(best_cat_name) > 18: cat_font_size = "18px"
    if len(best_cat_name) > 28: cat_font_size = "16px"

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"""<div class='kpi-container'><div class='kpi-label'>Potensi Penjualan</div><div class='kpi-value-base' style='font-size:22px'>{total_forecast:,.0f} <span style='font-size:16px; font-weight:normal; color:#64748b'>Pcs</span></div><div class='kpi-sub' style='color:{trend_color}; font-weight:bold'>{trend_arrow} {abs(growth):.1f}% <span style='color:gray; font-weight:normal'>vs Rata-rata</span></div></div>""", unsafe_allow_html=True)
    with c2: st.markdown(f"""<div class='kpi-container'><div class='kpi-label'>Kategori Unggulan</div><div class='kpi-value-base' style='font-size:{cat_font_size}'>{best_cat_name}</div><div class='kpi-sub' style='color:#64748b'>Est. Volume: <b>{best_cat_vol:,.0f}</b> Pcs</div></div>""", unsafe_allow_html=True)
    with c3: st.markdown(f"""<div class='kpi-container'><div class='kpi-label'>Tingkat Akurasi</div><div class='kpi-value-base' style='font-size:22px'>{stars}</div><div class='kpi-sub' style='color:{color_acc}; font-weight:bold'>{msg_acc} ({accuracy_score:.1f}%)</div></div>""", unsafe_allow_html=True)

    st.markdown("""<div style='margin-top: 15px; margin-bottom: 0px; font-size: 14px; color: #6c757d; font-style: italic;'>ℹ️ Catatan: Tingkat akurasi dihitung berdasarkan pengujian data historis (Test Set). Prediksi jangka panjang memiliki tingkat ketidakpastian yang lebih tinggi.</div>""", unsafe_allow_html=True)
    st.markdown("---")
    
    if not df_view.empty:
        st.subheader("📊 Grafik Prediksi Penjualan")
        # Sampling data jika terlalu banyak (biar ringan)
        plot_data = df_view.iloc[::2, :] if len(df_view) > 500 else df_view
        
        cat_order = plot_data.groupby('Kategori')['Nilai'].sum().sort_values(ascending=False).index.tolist()
        fig = px.bar(plot_data, x='Kategori', y='Nilai', color='Periode Label', barmode='group', category_orders={'Kategori': cat_order}, text_auto='.0f', color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(height=380, xaxis_title=None, yaxis_title="Jumlah (Pcs)", legend_title="Periode", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Rincian Stok Barang")
        sorted_cols = df_view[['Start Date', 'Periode Label']].drop_duplicates().sort_values('Start Date')['Periode Label'].tolist()
        pivot_table = df_view.pivot_table(index='Kategori', columns='Periode Label', values='Nilai', aggfunc='sum').fillna(0)
        valid_cols = [c for c in sorted_cols if c in pivot_table.columns]
        pivot_table = pivot_table.reindex(columns=valid_cols)
        pivot_table['TOTAL'] = pivot_table.sum(axis=1)
        pivot_table = pivot_table.sort_values('TOTAL', ascending=False)
        try: st.dataframe(pivot_table.style.format("{:,.0f}").background_gradient(cmap="Blues", subset=['TOTAL']), use_container_width=True)
        except: st.dataframe(pivot_table.style.format("{:,.0f}"), use_container_width=True)
        csv = pivot_table.to_csv().encode('utf-8')
        st.download_button(label="📥 Download Laporan (CSV)", data=csv, file_name=f"Laporan_{sel_model}.csv", mime="text/csv", type='primary')
    else:
        st.info("ℹ️ Silakan pilih rentang tanggal untuk melihat data.")

    try:
        K.clear_session()
        gc.collect()
    except:
        pass

# =========================================================
# 6.B PAGE 2: DETAIL KATEGORI (SINGLE DATE - CONTINUOUS GRAPH)
# =========================================================
elif st.session_state['page'] == "Detail Kategori":
    
    # --- 0. CSS KPI (SAMA PERSIS PAGE 1) ---
    st.markdown("""
    <style>
    div.kpi-container {
        background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px;
        text-align: center; height: 140px; display: flex; flex-direction: column; justify-content: center; align-items: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .kpi-label { font-size: 13px; color: #6c757d; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; }
    .kpi-value-base { font-weight: 700; color: #212529; margin: 0; line-height: 1.2; }
    .kpi-sub { font-size: 13px; margin-top: 8px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("📈 Analisa Detail Kategori")
    st.markdown("Pilih tab kategori di bawah untuk melihat analisis mendalam per barang.")
    st.divider()
    
    # 1. SETUP BATAS DATA
    max_hist_date = df_final.loc[df_final['Jenis']=='Aktual', 'Tanggal'].max()
    min_future_start = max_hist_date + pd.Timedelta(days=1)
    
    # 2. UI CONTROLS
    c_filter_1, c_filter_2 = st.columns([1, 1])
    
    with c_filter_1:
        # --- [PERBAIKAN] TAMPILKAN SEMUA OPSI MODEL (JANGAN DI-FILTER) ---
        model_opts = ['Model Pintar', 'Model Canggih', 'Model Klasik']
        
        sel_mod = st.selectbox("🤖 Pilih Model", model_opts, index=0, key="sb_model_detail")
        
        # --- [LOGIKA BARU] AUTO-TRAINING DI PAGE 2 ---
        # Cek: Apakah data masa depan untuk model ini sudah ada?
        cek_data = df_final[(df_final['Model'] == sel_mod) & (df_final['Jenis'] == 'Future')]
        
        # Jika BELUM ADA, panggil fungsi training sekarang!
        if cek_data.empty:
             with st.spinner(f"👩‍🍳 Sedang menyiapkan {sel_mod}... Harap tunggu sebentar..."):
                new_results = train_specific_model(sel_mod)
                
                if not new_results.empty:
                    # Update database utama
                    st.session_state['data_ready'] = pd.concat([st.session_state['data_ready'], new_results], ignore_index=True)
                    st.success(f"✅ {sel_mod} Selesai!")
                    st.rerun() # Refresh halaman biar datanya muncul
                else:
                    st.warning(f"Gagal memproses {sel_mod}. Cek data Anda.")
        
        # Cek batas tanggal maksimal untuk model ini
        df_fut_check = df_final[(df_final['Jenis']=='Future') & (df_final['Model']==sel_mod)]
        if not df_fut_check.empty:
            max_future_avail = df_fut_check['Tanggal'].max()
        else:
            max_future_avail = max_hist_date

    with c_filter_2:
        # DEFAULT: 8 Minggu ke depan
        def_end = max_hist_date + pd.Timedelta(weeks=8)
        
        # HANYA PILIH END DATE (START DATE OTOMATIS NYAMBUNG)
        target_date = st.date_input(
            "📅 Prediksi Hingga Tanggal", 
            value=def_end, 
            min_value=min_future_start,
            key="dp_detail_single"
        )

    # 3. LOGIKA AUTO-RUN (HEMAT MEMORI & KONTINUITAS)
    u_end = pd.to_datetime(target_date)
    is_ready = True # Selalu ready karena single date pasti valid

    if is_ready and u_end > max_future_avail:
        steps_needed = calculate_steps_to_date(max_hist_date, u_end)
        if steps_needed > 52: steps_needed = 52
            
        progress_text = f"Menghitung prediksi detail {sel_mod}..."
        my_bar = st.progress(0, text=progress_text)
        
        # Ambil Data Base yang BENAR (Sesuai Model)
        req_type = 'Aktual (Smooth)' if sel_mod in ['Model Pintar', 'Model Canggih'] else 'Aktual (Raw)'
        df_hist_base = df_final[(df_final['Jenis']=='Aktual') & (df_final['Model']==req_type)].copy()
        
        if df_hist_base.empty: df_hist_base = df_final[df_final['Jenis']=='Aktual'].copy()

        df_hist_base = df_hist_base[['Kategori', 'Tanggal', 'Nilai']].rename(columns={'Nilai': 'Total_Jumlah', 'Tanggal': 'Bulan'})
        agg_re = df_hist_base.groupby(['Kategori', 'Bulan'])['Total_Jumlah'].mean().reset_index()
        unique_cats = agg_re['Kategori'].unique()
        
        new_results = []
        try:
            for idx, cat in enumerate(unique_cats):
                my_bar.progress(int(((idx + 1) / len(unique_cats)) * 100), text=f"Memproses {cat}...")
                
                # Hanya jalankan target model
                res = process_single_category(idx, unique_cats, agg_re, steps_val=steps_needed, target_model=sel_mod)
                
                res_future = [d for d in res if not d.empty and d['Jenis'].iloc[0] == 'Future']
                new_results.extend(res_future)
                del res
                gc.collect()

            my_bar.empty()

            if new_results:
                with st.spinner("Sinkronisasi data..."):
                    df_calc_result = pd.concat(new_results, ignore_index=True)
                    
                    # --- LOGIKA SMART MERGE (FIXED VARIABLE NAME) ---
                    # Perhatikan: Di sini kita pakai 'sel_mod', bukan 'sel_model'
                    
                    # 1. Ambil data lama SELAIN Future model ini
                    df_base_others = df_final[~((df_final['Jenis'] == 'Future') & (df_final['Model'] == sel_mod))].copy()
                    
                    # 2. Ambil data Future model ini yang SUDAH ADA (Data CSV/Colab)
                    df_existing_future = df_final[(df_final['Jenis'] == 'Future') & (df_final['Model'] == sel_mod)].copy()
                    
                    # 3. Filter hasil baru: Hanya ambil tanggal yang BELUM ADA di data lama
                    if not df_existing_future.empty:
                        existing_dates = df_existing_future['Tanggal'].unique()
                        df_new_only = df_calc_result[~df_calc_result['Tanggal'].isin(existing_dates)]
                    else:
                        df_new_only = df_calc_result
                    
                    # 4. Gabungkan
                    df_updated = pd.concat([df_base_others, df_existing_future, df_new_only], ignore_index=True)
                    
                    # Clean up
                    del st.session_state['data_ready']
                    del df_final
                    gc.collect()
                    
                    st.session_state['data_ready'] = df_updated.sort_values('Tanggal')
                    st.rerun()
        except Exception as e:
            st.error(f"Gagal memproses: {str(e)}")
            st.stop()

    # 4. TAMPILAN TABS
    st.write("###")
    list_kategori = sorted(df_final['Kategori'].unique())
    tabs = st.tabs([f"📦 {k}" for k in list_kategori])

    for i, tab in enumerate(tabs):
        with tab:
            sel_cat = list_kategori[i]
            df_cat = df_final[df_final['Kategori'] == sel_cat]
            
            # Filter History
            target_hist_model = 'Aktual (Raw)'
            if sel_mod in ['Model Pintar', 'Model Canggih']:
                target_hist_model = 'Aktual (Smooth)'
                
            df_act = df_cat[(df_cat['Model'] == target_hist_model) & (df_cat['Jenis'] == 'Aktual')].copy()
            if df_act.empty: df_act = df_cat[(df_cat['Model'] == sel_mod) & (df_cat['Jenis'] == 'Aktual')].copy()
            if df_act.empty: df_act = df_cat[df_cat['Jenis'] == 'Aktual'].copy()
            
            df_act = df_act.sort_values('Tanggal').drop_duplicates(subset=['Tanggal'])
            
            # Filter Future (Dari BESOK s/d Tanggal Pilihan User)
            df_fut = df_cat[
                (df_cat['Model'] == sel_mod) & 
                (df_cat['Jenis'] == 'Future') &
                (df_cat['Tanggal'] <= u_end) # Ambil semua dari awal masa depan sampai batas user
            ].sort_values('Tanggal')
            
            df_test = df_cat[(df_cat['Model'] == sel_mod) & (df_cat['Jenis'] == 'Test')].sort_values('Tanggal')

            # --- KPI ---
            acc_val = calculate_metrics(df_final, sel_mod, sel_cat)
            accuracy = max(0, 100 - acc_val)
            
            if accuracy >= 80: acc_color, acc_msg = "#16a34a", "Sangat Akurat"
            elif accuracy >= 60: acc_color, acc_msg = "#16a34a", "Akurat"
            elif accuracy >= 40: acc_color, acc_msg = "#d97706", "Cukup"
            else: acc_color, acc_msg = "#dc2626", "Rendah"

            total_pred = df_fut['Nilai'].sum() if not df_fut.empty else 0
            
            growth_cat = 0
            if not df_act.empty:
                avg_hist = df_act['Nilai'].tail(6).mean()
                num_periods = df_fut['Tanggal'].nunique() or 1
                benchmark = avg_hist * num_periods
                if benchmark > 0:
                    growth_cat = ((total_pred - benchmark) / benchmark) * 100
            
            growth_arrow = "▲" if growth_cat >= 0 else "▼"
            growth_color = "#16a34a" if growth_cat >= 0 else "#dc2626"

            # Layout KPI
            st.write("")
            k1, k2, k3 = st.columns(3)
            with k1: st.markdown(f"""<div class='kpi-container'><div class='kpi-label'>Akurasi Model</div><div class='kpi-value-base' style='font-size:24px; color:{acc_color}'>{accuracy:.1f}%</div><div class='kpi-sub'>{acc_msg}</div></div>""", unsafe_allow_html=True)
            with k2: st.markdown(f"""<div class='kpi-container'><div class='kpi-label'>Total Prediksi</div><div class='kpi-value-base' style='font-size:24px'>{total_pred:,.0f} <span style='font-size:16px; font-weight:normal; color:#64748b'>Pcs</span></div><div class='kpi-sub'>Sampai {u_end.strftime('%d %b %Y')}</div></div>""", unsafe_allow_html=True)
            with k3: st.markdown(f"""<div class='kpi-container'><div class='kpi-label'>Tren Permintaan</div><div class='kpi-value-base' style='font-size:24px; color:{growth_color}'>{growth_arrow} {abs(growth_cat):.1f}%</div><div class='kpi-sub'>vs Rata-rata historis</div></div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader(f"📈 Grafik Tren: {sel_cat}")
            
            # --- PLOTTING ROMANUKE STYLE (SEKARANG LANGSUNG DARI CSV) ---
            LIMIT_PERIODS = 13 
            
            # 1. Tinggal panggil datanya karena sudah ada di CSV!
            df_raw = df_cat[(df_cat['Model'] == 'Aktual (Raw)') & (df_cat['Jenis'] == 'Aktual')].sort_values('Tanggal').drop_duplicates(subset=['Tanggal'])
            df_smooth = df_cat[(df_cat['Model'] == 'Aktual (Smooth)') & (df_cat['Jenis'] == 'Aktual')].sort_values('Tanggal').drop_duplicates(subset=['Tanggal'])
            
            # Fallback jaga-jaga kalau data kosong
            if df_smooth.empty: df_smooth = df_raw
            if df_raw.empty: df_raw = df_smooth

            if len(df_raw) > LIMIT_PERIODS:
                raw_plot = df_raw.iloc[-LIMIT_PERIODS:].copy()
                smooth_plot = df_smooth.iloc[-LIMIT_PERIODS:].copy()
                min_date_plot = smooth_plot['Tanggal'].min()
                df_test_plot = df_test[df_test['Tanggal'] >= min_date_plot].copy()
            else:
                raw_plot = df_raw
                smooth_plot = df_smooth
                df_test_plot = df_test

            fig = go.Figure()
            # Garis Data Mentah (Abu-abu Transparan di Belakang)
            fig.add_trace(go.Scatter(x=raw_plot['Tanggal'], y=raw_plot['Nilai'], mode='lines', name='Aktual (Raw)', line=dict(color='#8C7A6B', width=1.5), opacity=0.4))
            
            # Garis Tren Dasar / Smooth (Hitam Tebal di Depan)
            fig.add_trace(go.Scatter(x=smooth_plot['Tanggal'], y=smooth_plot['Nilai'], mode='lines+markers', name='Aktual (Smooth Tren)', line=dict(color='#2C3E50', width=3), marker=dict(size=6)))
            
            # Garis Validasi (Pengujian Masa Lalu)
            fig.add_trace(go.Scatter(x=df_test_plot['Tanggal'], y=df_test_plot['Nilai'], mode='lines', name='Validasi', line=dict(color='#f97316', width=2, dash='dot')))
            
            if not df_fut.empty and not smooth_plot.empty:
                last_hist = smooth_plot.iloc[-1]
                fut_dates = [last_hist['Tanggal']] + df_fut['Tanggal'].tolist()
                fut_vals = [last_hist['Nilai']] + df_fut['Nilai'].tolist()
                
                # Garis Prediksi Masa Depan
                fig.add_trace(go.Scatter(x=fut_dates, y=fut_vals, mode='lines+markers', name='Prediksi Masa Depan', line=dict(color='#2563eb', width=3), marker=dict(size=6)))
                
                cutoff_date = last_hist['Tanggal']
                fig.add_vline(x=cutoff_date, line_width=1, line_dash="dash", line_color="gray")
                fig.add_annotation(x=cutoff_date, y=raw_plot['Nilai'].max(), text="Sekarang", showarrow=False, yshift=10)

            fig.update_layout(height=450, xaxis_title="Periode Waktu", yaxis_title="Jumlah (Pcs)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified", plot_bgcolor="white")
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader(f"📋 Rincian Stok: {sel_cat}")
            
            if not df_fut.empty:
                tbl_display = df_fut[['Tanggal', 'Nilai']].copy()
                tbl_display['Start Date'] = tbl_display['Tanggal'] - pd.Timedelta(days=13)
                tbl_display['Rentang Periode'] = (tbl_display['Start Date'].dt.strftime('%d %b') + " - " + tbl_display['Tanggal'].dt.strftime('%d %b %Y'))
                tbl_display['Prediksi (Pcs)'] = tbl_display['Nilai'].apply(lambda x: f"{x:,.0f}")
                final_tbl = tbl_display[['Rentang Periode', 'Prediksi (Pcs)']].reset_index(drop=True)
                
                c_tbl, c_dl = st.columns([3, 1])
                with c_tbl: st.table(final_tbl)
                with c_dl:
                    st.write("###")
                    csv = df_fut.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Download Data", data=csv, file_name=f"Detail_{sel_cat}_{sel_mod}.csv", mime="text/csv", type='primary', key=f"dl_btn_{i}")
            else:
                st.info("Belum ada data masa depan pada rentang tanggal yang dipilih.")

    try:
        K.clear_session()
        gc.collect()
    except:
        pass

# =========================================================
# 6.C PAGE 3: PERBANIDNGAN MODEL (ANTI-CRASH & AUTO-RUN)
# =========================================================
elif st.session_state['page'] == "Perbandingan Model":
    st.title("⚖️ Perbandingan Kecerdasan Model")
    st.markdown("Halaman ini menguji model mana yang paling 'pintar' dalam menebak pola penjualan toko Anda.")
    st.divider()

    # --- [INIT] SIAPKAN CATATAN PERCOBAAN ---
    if 'attempted_models' not in st.session_state:
        st.session_state['attempted_models'] = set()

    # 1. Cek model apa yang SUDAH ADA di data
    existing_models = df_final['Model'].unique()
    required_models = ['Model Pintar', 'Model Canggih', 'Model Klasik']
    
    # 2. Cari model yang HILANG (Belum ada di data)
    missing_models = [m for m in required_models if m not in existing_models]
    
    # 3. Filter: Hanya latih model yang BELUM PERNAH DICOBA sesi ini
    # (Ini kunci anti-loop: Kalau sudah pernah dicoba & gagal, jangan dipaksa lagi)
    models_to_train = [m for m in missing_models if m not in st.session_state['attempted_models']]

    # --- LOGIKA AUTO-RUN (HANYA JIKA ADA YG PERLU DILATIH) ---
    if models_to_train:
        status_box = st.empty()
        status_box.info(f"⏳ Sedang melengkapi data perbandingan... Melatih: **{', '.join(models_to_train)}**")
        
        prog_bar = st.progress(0)
        any_success = False
        
        for i, mod_name in enumerate(models_to_train):
            prog_bar.progress(int((i / len(models_to_train)) * 100), text=f"Sedang melatih {mod_name}...")
            
            # Tandai bahwa model ini sedang dicoba (biar gak diulang next refresh)
            st.session_state['attempted_models'].add(mod_name)
            
            # Latih Model
            new_results = train_specific_model(mod_name)
            
            if not new_results.empty:
                # Jika sukses, simpan ke memori
                st.session_state['data_ready'] = pd.concat([st.session_state['data_ready'], new_results], ignore_index=True)
                any_success = True
            
        prog_bar.empty()
        status_box.empty()
        
        # Refresh HANYA jika ada data baru yang masuk (Biar update tampilan)
        # Jika gagal semua, jangan refresh biar gak looping
        if any_success:
            st.rerun()
            
    # --- JIKA GAGAL LATIH (MISSING TAPI SUDAH DICOBA) ---
    failed_models = [m for m in missing_models if m in st.session_state['attempted_models']]
    if failed_models:
        st.warning(f"⚠️ Gagal melatih model berikut (mungkin data kurang): **{', '.join(failed_models)}**. Menampilkan perbandingan yang ada saja.")

    # --- FUNGSI HELPER: RENDER KONTEN ---
    def render_comparison_content(cat_comp):
        comp_data = []
        # Ambil semua model KECUALI data Aktual
        models = [m for m in df_final['Model'].unique() if 'Aktual' not in m]
        
        if not models:
            st.info("Belum ada model yang berhasil dilatih. Cek kembali data Anda.")
            return

        # Hitung Metrik
        for i, m in enumerate(models):
            err = calculate_metrics(df_final, m, cat_comp)
            acc = max(0, 100 - err)
            comp_data.append({'Model': m, 'Akurasi (%)': acc, 'Error (%)': err})
        
        df_comp = pd.DataFrame(comp_data).sort_values('Akurasi (%)', ascending=False).reset_index(drop=True)
        
        if df_comp.empty:
            st.warning("Tidak ada data prediksi untuk dibandingkan.")
            return

        winner = df_comp.iloc[0]

        # Tampilkan Juara
        st.write("###")
        if winner['Akurasi (%)'] >= 80: win_color, border_color, msg = "#dcfce7", "#16a34a", "Sangat direkomendasikan."
        elif winner['Akurasi (%)'] >= 60: win_color, border_color, msg = "#fff7ed", "#f97316", "Cukup stabil."
        else: win_color, border_color, msg = "#fef2f2", "#dc2626", "Perlu hati-hati."

        st.markdown(f"""<div style="background-color: {win_color}; border: 2px solid {border_color}; padding: 20px; border-radius: 10px; text-align: center;"><h3 style="margin:0; color: {border_color};">🏆 JUARA: {winner['Model']}</h3><p style="font-size: 18px; margin-top: 10px;">Akurasi mencapai <b>{winner['Akurasi (%)']:.1f}%</b> (Error hanya {winner['Error (%)']:.1f}%).<br><span style="font-size:14px; color: gray;">{msg}</span></p></div>""", unsafe_allow_html=True)
        st.write("###")

        # Grafik Bar & Tabel Klasemen
        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            st.subheader("📊 Grafik Peringkat Akurasi")
            fig = px.bar(df_comp, x='Akurasi (%)', y='Model', orientation='h', text_auto='.1f', color='Akurasi (%)', color_continuous_scale='Greens')
            fig.update_layout(height=300, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        with col_table:
            st.subheader("📋 Klasemen")
            try:
                st.dataframe(df_comp[['Model', 'Akurasi (%)']].style.format({"Akurasi (%)": "{:.1f}"}).background_gradient(cmap='Greens'), use_container_width=True, hide_index=True)
            except:
                st.dataframe(df_comp[['Model', 'Akurasi (%)']].style.format({"Akurasi (%)": "{:.1f}"}), use_container_width=True, hide_index=True)

        # ==========================================================
        # GRAFIK GARIS (SEKARANG MUNCUL DI GLOBAL JUGA)
        # ==========================================================
        st.markdown("---")
        title_grafik = f"👁️ Bukti Visual (Uji Garis): {cat_comp}" if cat_comp else "👁️ Bukti Visual (Uji Garis): Global (Semua Kategori)"
        st.subheader(title_grafik)
        st.markdown("Menampilkan data historis (garis hitam) berbanding dengan hasil prediksi di masa pengujian (garis putus-putus).")
        
        fig_line = go.Figure()
        
        # --- PERSIAPAN DATA ROMANUKE STYLE (SEKARANG LANGSUNG DARI CSV) ---
        # 1. Tarik data Actual Raw & Smooth langsung dari df_final
        df_raw_all = df_final[(df_final['Jenis'] == 'Aktual') & (df_final['Model'] == 'Aktual (Raw)')].copy()
        df_smooth_all = df_final[(df_final['Jenis'] == 'Aktual') & (df_final['Model'] == 'Aktual (Smooth)')].copy()
        
        # Jaga-jaga kalau CSV-nya belum terupdate (Fallback)
        if df_smooth_all.empty: df_smooth_all = df_raw_all
        if df_raw_all.empty: df_raw_all = df_smooth_all
        
        # Filter spesifik per kategori kalau user milih Tab Kategori
        if cat_comp:
            df_raw_all = df_raw_all[df_raw_all['Kategori'] == cat_comp]
            df_smooth_all = df_smooth_all[df_smooth_all['Kategori'] == cat_comp]
            
        # Agregasi untuk berjaga-jaga, lalu urutkan tanggalnya
        df_raw_agg = df_raw_all.groupby('Tanggal')['Nilai'].sum().reset_index().sort_values('Tanggal')
        df_smooth_agg = df_smooth_all.groupby('Tanggal')['Nilai'].sum().reset_index().sort_values('Tanggal')
        
        # Batasi plot 26 titik (1 Tahun Terakhir)
        if len(df_raw_agg) > 26:
            df_raw_plot = df_raw_agg.iloc[-26:].reset_index(drop=True)
            df_smooth_plot = df_smooth_agg.iloc[-26:].reset_index(drop=True)
        else:
            df_raw_plot = df_raw_agg
            df_smooth_plot = df_smooth_agg
            
        min_date_plot = df_raw_plot['Tanggal'].min()
        max_date_plot = df_raw_plot['Tanggal'].max()

        if not df_raw_plot.empty:
            # 1. Gambar Garis Background (Raw)
            fig_line.add_trace(go.Scatter(x=df_raw_plot['Tanggal'], y=df_raw_plot['Nilai'], mode='lines', name='Aktual (Raw)', line=dict(color='#8C7A6B', width=1.5), opacity=0.4))
            
            # 2. Gambar Garis Foreground (Smooth)
            fig_line.add_trace(go.Scatter(x=df_smooth_plot['Tanggal'], y=df_smooth_plot['Nilai'], mode='lines+markers', name='Aktual (Smooth)', line=dict(color='#2C3E50', width=3), marker=dict(size=6)))
            
            # 3. Warna Spesifik per Model
            model_colors = {'Model Pintar': '#27AE60', 'Model Canggih': '#E74C3C', 'Model Klasik': '#8E44AD'}
            
            for m in models:
                pred_df = df_final[(df_final['Model'] == m) & (df_final['Jenis'] == 'Test')].copy()
                if not pred_df.empty:
                    pred_df = pred_df.drop_duplicates(subset=['Kategori', 'Tanggal'])
                    if cat_comp: pred_df = pred_df[pred_df['Kategori'] == cat_comp]
                    
                    pred_df = pred_df.groupby('Tanggal')['Nilai'].sum().reset_index()
                    pred_df = pred_df.sort_values('Tanggal').reset_index(drop=True)
                    pred_view = pred_df[(pred_df['Tanggal'] >= min_date_plot) & (pred_df['Tanggal'] <= max_date_plot)]
                    
                    if not pred_view.empty:
                        m_color = model_colors.get(m, '#2563eb')
                        fig_line.add_trace(go.Scatter(x=pred_view['Tanggal'], y=pred_view['Nilai'], mode='lines+markers', name=f"{m}", line=dict(width=2.5, dash='dot', color=m_color), marker=dict(size=5)))

        fig_line.update_layout(height=450, xaxis_title="Periode Waktu", yaxis_title="Jumlah Penjualan (Pcs)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_line, use_container_width=True)

    # --- IMPLEMENTASI TABS ---
    tab_global, tab_kategori = st.tabs(["🌐 Global (Semua Barang)", "📦 Spesifik Per Kategori"])

    with tab_global:
        render_comparison_content(None)

    with tab_kategori:
        col_sel, _ = st.columns([1, 2])
        with col_sel:
            cat_selected = st.selectbox("Pilih Kategori untuk Dianalisis:", sorted(df_final['Kategori'].unique()))
        
        render_comparison_content(cat_selected)

# =========================================================
# 6.D PAGE 4: DATA MENTAH
# =========================================================
elif st.session_state['page'] == "Data Mentah":
    st.title("📂 Data & Kategori Produk")
    st.markdown("Halaman ini menampilkan rincian data mentah transaksi, daftar produk per kategori, serta pola penjualan historis toko Anda.")
    st.divider()

    df_raw_display = None
    sumber_data = ""

    # --- LOGIKA PENCARIAN DATA MENTAH ---
    # 1. Cek apakah ada data upload dari pengguna di sesi ini
    if 'raw_df_v20' in st.session_state and st.session_state['raw_df_v20'] is not None:
        df_raw_display = st.session_state['raw_df_v20'].copy()
        sumber_data = "Data Upload Pengguna (Sesi Aktif)"
        
    # 2. Cek apakah ada file mentah asli di folder sistem
    elif os.path.exists("data_mentah_asli.csv"):
        df_raw_display = pd.read_csv("data_mentah_asli.csv", low_memory=False)
        sumber_data = "Database Mentah Internal (data_mentah_asli.csv)"
        
    elif os.path.exists("GABUNGAN_20232025_NOKATEGORI.csv"):
        df_raw_display = pd.read_csv("GABUNGAN_20232025_NOKATEGORI.csv", low_memory=False)
        sumber_data = "Database Mentah Internal (GABUNGAN_...)"

    # --- JIKA DATA MENTAH (TRANSAKSI HARIAN) DITEMUKAN ---
    if df_raw_display is not None:
        st.success(f"✅ Menampilkan sumber data dari: **{sumber_data}**")
        
        # Deteksi otomatis nama kolom dari file mentah
        all_cols = list(df_raw_display.columns)
        guess_date = next((c for c in all_cols if 'tgl' in c.lower() or 'date' in c.lower() or 'time' in c.lower() or 'stamp' in c.lower()), all_cols[0])
        guess_item = next((c for c in all_cols if ('nama' in c.lower() or 'prod' in c.lower() or 'desc' in c.lower() or 'barang' in c.lower()) and not ('kode' in c.lower() or 'id' in c.lower() or 'sku' in c.lower())), all_cols[2] if len(all_cols)>2 else None)
        guess_qty = next((c for c in all_cols if 'jumlah' in c.lower() or 'qty' in c.lower() or 'kuantitas' in c.lower()), all_cols[3] if len(all_cols)>3 else None)
        
        # Pembersihan & Kategorisasi On-The-Fly untuk Tampilan
        df_raw_display['Tanggal'] = pd.to_datetime(df_raw_display[guess_date], errors='coerce')
        df_raw_display = df_raw_display.dropna(subset=['Tanggal'])
        df_raw_display['Kategori'] = df_raw_display[guess_item].apply(categorize_item)
        df_raw_display['Total_Jumlah'] = pd.to_numeric(df_raw_display[guess_qty], errors='coerce').fillna(0)
        
        # Filter hanya untuk kategori yang valid di sistem kita
        df_raw_display = df_raw_display[df_raw_display['Kategori'].isin(VALID_CATEGORIES)]
        
        tab1, tab2, tab3 = st.tabs(["📝 Data Mentah (Harian)", "🏷️ Daftar Produk per Kategori", "📈 Pola 2 Mingguan"])

        # TAB 1: DATA HARIAN
        with tab1:
            st.subheader("Data Transaksi Keseluruhan")
            st.dataframe(df_raw_display, use_container_width=True)

        # TAB 2: NAMA PRODUK
        with tab2:
            st.subheader("Daftar Detail Produk Berdasarkan Kategori")
            kategori_list = sorted(df_raw_display['Kategori'].unique())
            
            for cat in kategori_list:
                produk = sorted(df_raw_display[df_raw_display['Kategori'] == cat][guess_item].dropna().astype(str).unique())
                total_item = len(produk)
                
                with st.expander(f"📦 {cat} ({total_item} Item)"):
                    if total_item > 0:
                        st.markdown(f"<div style='line-height:1.6; color:#334155;'>{', '.join(produk)}</div>", unsafe_allow_html=True)
                    else:
                        st.write("-")

        # TAB 3: GRAFIK AGREGASI 2 MINGGUAN
        with tab3:
            st.subheader("Grafik & Data Agregasi (2 Mingguan)")
            try:
                # Grouping dari data harian menjadi 2 mingguan
                df_agg = df_raw_display.groupby(['Kategori', pd.Grouper(key='Tanggal', freq='2W')])['Total_Jumlah'].sum().reset_index()
                df_agg.rename(columns={'Total_Jumlah': 'Total Penjualan (Pcs)'}, inplace=True)
                
                fig = px.line(df_agg, x='Tanggal', y='Total Penjualan (Pcs)', color='Kategori', markers=True, color_discrete_sequence=px.colors.qualitative.Safe)
                fig.update_layout(height=450, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), plot_bgcolor="white")
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.write("📋 **Tabel Rekapitulasi Historis 2 Mingguan**")
                st.dataframe(df_agg.style.format({"Total Penjualan (Pcs)": "{:,.0f}"}), use_container_width=True)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membuat grafik: {e}")

    # --- JIKA TIDAK ADA DATA MENTAH, PAKAI DATA DEFAULT (AGREGASI) SEBAGAI CADANGAN ---
    else:
        st.warning("⚠️ File data transaksi mentah harian tidak ditemukan di dalam folder sistem. Menampilkan versi agregasi 2 mingguan dari data model.")
        
        if 'data_clean' in st.session_state:
            df_clean = st.session_state['data_clean'].copy()
            df_clean['Tanggal'] = pd.to_datetime(df_clean['Tanggal'], errors='coerce')
            
            tab1, tab2 = st.tabs(["📝 Data Agregasi", "📈 Pola 2 Mingguan"])
            
            with tab1:
                st.subheader("Data Transaksi (Sudah Di-Agregasi)")
                st.dataframe(df_clean, use_container_width=True)
                
            with tab2:
                st.subheader("Grafik & Data Agregasi (2 Mingguan)")
                try:
                    if 'Total_Jumlah' not in df_clean.columns and 'Nilai' in df_clean.columns:
                        col_qty = 'Nilai'
                    else:
                        col_qty = 'Total_Jumlah'
                        
                    df_agg = df_clean.groupby(['Kategori', pd.Grouper(key='Tanggal', freq='2W')])[col_qty].sum().reset_index()
                    df_agg.rename(columns={col_qty: 'Total Penjualan (Pcs)'}, inplace=True)
                    
                    fig = px.line(df_agg, x='Tanggal', y='Total Penjualan (Pcs)', color='Kategori', markers=True, color_discrete_sequence=px.colors.qualitative.Safe)
                    fig.update_layout(height=450, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), plot_bgcolor="white")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Gagal memproses grafik: {e}")
        else:
            st.error("Belum ada data sama sekali.")

# =========================================================
# 6.E PAGE 5: USER GUIDE
# =========================================================
elif st.session_state['page'] == "Panduan Pengguna":
    st.title("📘 Panduan Pengguna")
    st.markdown("Dokumentasi lengkap cara menggunakan aplikasi prediksi penjualan ini.")
    st.divider()

    st.info("💡 **Tips Cepat:** Selalu pastikan format tanggal di file CSV Anda konsisten agar sistem bisa membaca pola waktu dengan akurat.")

    with st.expander("📂 1. Cara Upload Data (Penting!)", expanded=True):
        st.markdown("""
        Agar aplikasi berjalan lancar, pastikan file CSV transaksi Anda memiliki kolom berikut:
        
        1.  **Tanggal Transaksi:** (Contoh: `01/12/2025`, `2025-12-01`, atau kode transaksi yang mengandung tanggal).
        2.  **Nama Barang:** (Contoh: `Gamis Syar'i`, `Kemeja Flanel`, `Ciput Rajut`).
        3.  **Jumlah/Qty:** Angka jumlah barang yang terjual.
        
        **Langkah-langkah:**
        1.  Klik tombol **Browse files** di sidebar sebelah kiri.
        2.  Pilih file `.csv` dari komputer Anda.
        3.  Aplikasi akan otomatis mendeteksi kolom. Jika salah, sesuaikan manual di menu **"Pengaturan Kolom"**.
        4.  Klik tombol **✅ Proses Data**. Tunggu hingga model selesai mempelajari pola data Anda.
        """)

    with st.expander("📊 2. Membaca Ringkasan Prediksi"):
        st.markdown("""
        Halaman ini adalah "Dashboard Utama" untuk melihat masa depan toko Anda.
        
        * **Potensi Penjualan:** Total barang yang diprediksi akan terjual di periode mendatang.
        * **Indikator Tren:** Panah Hijau (▲) artinya naik dibanding rata-rata masa lalu. Panah Merah (▼) artinya turun.
        * **Pilih Model:** Anda bisa mengganti model (Model Klasik, Pintar, atau Canggih) untuk melihat sudut pandang prediksi yang berbeda.
        * **Rentang Waktu:** Ubah dropdown untuk melihat prediksi 2 minggu ke depan, 4 minggu, hingga 8 minggu.
        """)

    with st.expander("📈 3. Analisa Detail Kategori"):
        st.markdown("""
        Gunakan halaman ini jika ingin melihat stok per jenis barang secara spesifik.
        
        * **Tab Kategori:** Klik tab (misal: "Fashion Muslim", "Aksesoris") untuk fokus ke satu jenis barang.
        * **Grafik Tren:** Garis **Hitam** adalah data masa lalu. Garis **Biru** adalah prediksi masa depan.
        * **Tabel Rincian:** Gunakan tabel ini untuk merencanakan stok gudang (Stock Opname). Anda bisa mendownload tabel ini ke Excel/CSV.
        """)

    with st.expander("⚖️ 4. Perbandingan Model Prediksi"):
        st.markdown("""
        Bingung harus percaya model yang mana? Halaman ini jurinya.
        
        * **Juara Akurasi:** Aplikasi akan otomatis menghitung model mana yang tebakannya paling mendekati kenyataan (Error paling kecil).
        * **Uji Garis:** Lihat grafik garis putus-putus. Model terbaik adalah yang garis putus-putusnya paling "nempel" dengan garis hitam (data asli).
        * **Rekomendasi:**
            * Jika data stabil, **Model Klasik (SARIMA)** biasanya cukup.
            * Jika data fluktuatif (naik turun drastis), **Model Pintar (XGBoost)** atau **Canggih (LSTM)** biasanya lebih baik.
        """)

    st.divider()
    st.caption("© 2026 KawanUMKM Forecasting System - Dibuat untuk membantu keputusan bisnis Anda.")

