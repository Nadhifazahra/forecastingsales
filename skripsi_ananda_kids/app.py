import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
import warnings
from hijri_converter import Gregorian

# --- 1. SETUP HALAMAN ---
st.set_page_config(page_title="Dashboard Prediksi", layout="wide", page_icon="🛍️")
warnings.filterwarnings("ignore")

# Session State
if 'page' not in st.session_state:
    st.session_state['page'] = "Ringkasan Prediksi"

# =========================================================
# CSS MINIMAL — DEFAULT STREAMLIT THEME FRIENDLY
# =========================================================
st.markdown("""
<style>

/* KPI CARD */
.kpi-container {
    background-color: var(--secondary-background-color);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(0,0,0,0.06);
    text-align: center;
}

.kpi-label {
    font-size: 13px;
    font-weight: 600;
    opacity: 0.6;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kpi-value {
    font-size: 28px;
    font-weight: 800;
    margin: 6px 0;
}

.kpi-sub {
    font-size: 13px;
    opacity: 0.75;
}

/* SUMMARY BOX */
.summary-box {
    background-color: var(--secondary-background-color);
    border-left: 4px solid var(--primary-color);
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
}

.summary-title {
    font-weight: 700;
    font-size: 16px;
    margin-bottom: 6px;
}

.summary-text {
    font-size: 14px;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. FUNGSI BANTUAN
# =========================================================

def get_hijri_bell_curve(date_val):
    try:
        h = Gregorian(date_val.year, date_val.month, date_val.day).to_hijri()
        if h.month == 8: return 0.2 + (h.day / 30) * 0.3
        elif h.month == 9: return 0.5 + (h.day / 30) * 0.5
        elif h.month == 10:
            if h.day <= 7: return 0.5 - (h.day / 7) * 0.5
            return 0.0
        return 0.0
    except:
        return 0.0

def categorize_item(item_name):
    if pd.isna(item_name): return 'Unknown'
    item_name = str(item_name).lower()
    if any(x in item_name for x in ['gamis','tunik','hijab','koko','mukena']): return 'Fashion Muslim'
    if any(x in item_name for x in ['sepatu','tas','dompet','jam','kacamata']): return 'Aksesoris'
    if any(x in item_name for x in ['anak','bayi','baby']): return 'Fashion Bayi & Anak lainnya'
    if any(x in item_name for x in ['pria','cowok','kemeja']): return 'Pakaian Laki-laki'
    if any(x in item_name for x in ['wanita','dress','blouse','rok']): return 'Pakaian Perempuan'
    return 'Set'

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_pred - y_true) /
                   ((np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-10)) * 100

# =========================================================
# 3. ENGINE TRAINING (MODE UPLOAD)
# =========================================================

def train_predict_xgboost(df, cat, future_periods=4):
    df = df.sort_values('Tanggal').copy()
    df['Total_Jumlah'] = df['Total_Jumlah'].rolling(window=3, min_periods=1).mean()
    df['Target_Log'] = np.log1p(df['Total_Jumlah'])
    df['lag_1'] = df['Target_Log'].shift(1)
    df['lag_2'] = df['Target_Log'].shift(2)
    df['rolling_mean_4'] = df['Target_Log'].shift(1).rolling(4).mean()
    df['velocity'] = df['lag_1'] - df['lag_2']
    df['Bell_Curve'] = df['Tanggal'].apply(get_hijri_bell_curve)
    df['Sin_Week'] = np.sin(2 * np.pi * df["Tanggal"].dt.isocalendar().week / 52)
    df = df.dropna()
    
    if len(df) < 10: return None
    
    FEATS = ["lag_1", "lag_2", "rolling_mean_4", "velocity", "Bell_Curve", "Sin_Week"]
    n_test = max(2, int(len(df) * 0.1))
    
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, objective='reg:tweedie')
    model.fit(df.iloc[:-n_test][FEATS], df.iloc[:-n_test]['Target_Log'])
    
    df_act = pd.DataFrame({'Tanggal': df['Tanggal'], 'Nilai': np.expm1(df['Target_Log']), 'Jenis': 'Aktual'})
    preds_test = np.expm1(model.predict(df.iloc[-n_test:][FEATS]))
    df_test = pd.DataFrame({'Tanggal': df.iloc[-n_test:]['Tanggal'], 'Nilai': preds_test, 'Jenis': 'Test'})
    
    last_date = df['Tanggal'].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=future_periods+1, freq='2W')[1:]
    
    curr_feats = df.iloc[-1:].copy()
    future_vals = []
    
    for date in future_dates:
        curr_feats['Bell_Curve'] = get_hijri_bell_curve(date)
        curr_feats['Sin_Week'] = np.sin(2 * np.pi * date.isocalendar().week / 52)
        p_log = model.predict(curr_feats[FEATS])[0]
        future_vals.append(np.expm1(p_log))
        curr_feats['lag_2'] = curr_feats['lag_1']
        curr_feats['lag_1'] = p_log
        curr_feats['rolling_mean_4'] = (curr_feats['rolling_mean_4']*3 + p_log)/4
        
    df_fut = pd.DataFrame({'Tanggal': future_dates, 'Nilai': future_vals, 'Jenis': 'Future'})
    
    final = pd.concat([df_act, df_test, df_fut])
    final['Kategori'] = cat
    final['Model'] = 'XGBoost (Auto)'
    return final

def process_uploaded_file(uploaded_file):
    df_raw = pd.read_csv(uploaded_file)
    cols = df_raw.columns
    col_date, col_name, col_qty = cols[0], cols[1], cols[2]
    
    df_raw[col_date] = pd.to_datetime(df_raw[col_date], errors='coerce')
    df_raw[col_qty] = pd.to_numeric(df_raw[col_qty], errors='coerce').fillna(0)
    df_raw = df_raw.dropna(subset=[col_date])
    df_raw['Kategori'] = df_raw[col_name].apply(categorize_item)
    
    df_agg = df_raw.groupby(['Kategori', pd.Grouper(key=col_date, freq='2W')])[col_qty].sum().reset_index()
    df_agg.rename(columns={col_date: 'Tanggal', col_qty: 'Total_Jumlah'}, inplace=True)
    
    all_results = []
    cats = df_agg['Kategori'].unique()
    progress = st.progress(0)
    
    for i, cat in enumerate(cats):
        res = train_predict_xgboost(df_agg[df_agg['Kategori'] == cat], cat)
        if res is not None: all_results.append(res)
        progress.progress((i + 1) / len(cats))
        
    if all_results: return pd.concat(all_results, ignore_index=True)
    return None

# =========================================================
# 4. SIDEBAR
# =========================================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921222.png", width=50)
    st.markdown("### Dashboard UMKM")
    st.markdown("---")
    
    # 1. MENU NAVIGASI (Tombol Bersih)
    st.markdown("<small style='color:#64748b; font-weight:600; letter-spacing:1px;'>MENU UTAMA</small>", unsafe_allow_html=True)
    
    if st.button("📊 Ringkasan Prediksi", 
                 key="nav_ringkasan", 
                 type="primary" if st.session_state['page'] == "Ringkasan Prediksi" else "secondary"):
        st.session_state['page'] = "Ringkasan Prediksi"
        st.rerun()

    if st.button("📈 Detail Kategori", 
                 key="nav_detail", 
                 type="primary" if st.session_state['page'] == "Detail Per Kategori" else "secondary"):
        st.session_state['page'] = "Detail Per Kategori"
        st.rerun()
        
    if st.button("⚖️ Perbandingan Model", 
                 key="nav_compare", 
                 type="primary" if st.session_state['page'] == "Perbandingan Model" else "secondary"):
        st.session_state['page'] = "Perbandingan Model"
        st.rerun()
    
    st.markdown("---")
    
    # 2. UPLOAD DATA
    st.markdown("<small style='color:#64748b; font-weight:600; letter-spacing:1px;'>DATA INPUT</small>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")
    if not uploaded_file:
        st.caption("✅ Menggunakan Data Default")

# --- B. Logika Data ---
df_final = None

if uploaded_file:
    try:
        if 'last_uploaded' not in st.session_state or st.session_state['last_uploaded'] != uploaded_file.name:
            with st.spinner("🔄 Memproses data..."):
                df_processed = process_uploaded_file(uploaded_file)
                st.session_state['data_cache'] = df_processed
                st.session_state['last_uploaded'] = uploaded_file.name
        df_final = st.session_state.get('data_cache')
    except Exception as e:
        st.error(f"Gagal memproses file: {e}")
else:
    try:
        df_final = pd.read_csv("database_prediksi_final.csv")
        df_final['Tanggal'] = pd.to_datetime(df_final['Tanggal'])
    except:
        st.error("⚠️ File 'database_prediksi_final.csv' tidak ditemukan.")

# =========================================================
# 5. HALAMAN 1: RINGKASAN PREDIKSI (UI CLEAN & PRO)
# =========================================================
if df_final is not None and st.session_state['page'] == "Ringkasan Prediksi":
    
    st.title("Ringkasan Prediksi Penjualan")
    st.markdown("Analisis performa bisnis untuk periode mendatang.")
    st.markdown("---")
    
    # --- FILTER ---
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        sel_model = st.selectbox("🤖 Model Prediksi", df_final['Model'].unique())
    with col_f2:
        periods = st.selectbox("📅 Rentang Waktu", [2, 4, 6, 8], format_func=lambda x: f"{x} Periode (± {int(x/2)} Bulan)")

    # --- DATA PREP ---
    df_fut = df_final[(df_final['Jenis'] == 'Future') & (df_final['Model'] == sel_model)].sort_values('Tanggal')
    dates = sorted(df_fut['Tanggal'].unique())[:periods]
    df_plot = df_fut[df_fut['Tanggal'].isin(dates)].copy()
    
    df_past = df_final[(df_final['Jenis'] == 'Aktual') & (df_final['Model'] == sel_model)].sort_values('Tanggal')
    df_past_compare = df_past.tail(periods)
    
    # --- KPI CALC ---
    total_forecast = df_plot['Nilai'].sum()
    total_past = df_past_compare['Nilai'].sum()
    growth = ((total_forecast - total_past) / total_past * 100) if total_past > 0 else 0
    
    top_cat = df_plot.groupby('Kategori')['Nilai'].sum().sort_values(ascending=False).head(1)
    best_cat_name = top_cat.index[0] if not top_cat.empty else "-"
    best_cat_val = top_cat.values[0] if not top_cat.empty else 0
    
    # Event Check
    events = []
    for d in dates:
        h = Gregorian(d.year, d.month, d.day).to_hijri()
        if h.month == 9: events.append("Ramadhan")
        elif h.month == 10: events.append("Lebaran")
    event_label = ", ".join(list(set(events))) if events else "Reguler"

    # --- KPI CARDS (CLEAN HTML) ---
    col1, col2, col3 = st.columns(3)
    
    growth_color = "#16a34a" if growth >= 0 else "#dc2626"
    arrow = "▲" if growth >= 0 else "▼"
    
    with col1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Total Prediksi</div>
            <div class="kpi-value">{total_forecast:,.0f}</div>
            <div class="kpi-sub" style="color:{growth_color}">{arrow} {abs(growth):.1f}% <span style="color:#94a3b8">vs periode lalu</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Kategori Unggulan</div>
            <div class="kpi-value" style="font-size:22px">{best_cat_name}</div>
            <div class="kpi-sub" style="color:#64748b">Vol: {best_cat_val:,.0f} Pcs</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Momen Penting</div>
            <div class="kpi-value" style="font-size:22px">{event_label}</div>
            <div class="kpi-sub" style="color:#64748b">Cek strategi marketing</div>
        </div>
        """, unsafe_allow_html=True)

    # --- CHART ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 Grafik Proyeksi")
    
    start = df_plot['Tanggal'] - pd.Timedelta(days=13)
    df_plot['Label'] = start.dt.strftime('%d %b') + " - " + df_plot['Tanggal'].dt.strftime('%d %b')
    df_plot = df_plot.sort_values('Tanggal')

    fig = px.bar(
        df_plot, x="Kategori", y="Nilai", color="Label", barmode="group", text_auto='.0f',
        color_discrete_sequence=px.colors.qualitative.Safe, # Warna Soft
    )
    
    fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis_title=None,
    yaxis_title=None,
    legend_title=None,
    height=450,
    margin=dict(t=20, l=0, r=0, b=0),
    
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
    fig.update_traces(textposition='outside')
    
    st.plotly_chart(fig, use_container_width=True)

    # --- EXECUTIVE SUMMARY (DI BAWAH) ---
    st.markdown(f"""
    <div class="summary-box">
        <div class="summary-title">🤖 Executive Summary</div>
        <p class="summary-text">
            Berdasarkan analisis model <b>{sel_model}</b>, tren penjualan menunjukkan pergerakan 
            <b style="color:{growth_color}">{'POSITIF' if growth > 0 else 'NEGATIF'}</b>. 
            Kategori <b>{best_cat_name}</b> menjadi penopang utama penjualan. 
            {f"Siapkan stok ekstra untuk menyambut <b>{event_label}</b>." if event_label != "Reguler" else "Pastikan ketersediaan stok reguler terjaga."}
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# 6. HALAMAN 2: DETAIL PER KATEGORI
# =========================================================
elif df_final is not None and st.session_state['page'] == "Detail Per Kategori":
    
    st.title("Detail Analisis Kategori")
    st.markdown("Analisis mendalam per kategori barang.")
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        cat = st.selectbox("📂 Kategori", df_final['Kategori'].unique())
    with c2:
        mod = st.selectbox("🤖 Model", df_final[df_final['Kategori'] == cat]['Model'].unique())
        
    df_d = df_final[(df_final['Kategori'] == cat) & (df_final['Model'] == mod)].sort_values('Tanggal')
    
    act = df_d[df_d['Jenis'] == 'Aktual']
    test = df_d[df_d['Jenis'] == 'Test']
    fut = df_d[df_d['Jenis'] == 'Future']
    
    # KPI Mini
    smape_val = 0
    if not test.empty:
        m = pd.merge(test, act, on='Tanggal', suffixes=('_p', '_t'))
        if not m.empty: smape_val = smape(m['Nilai_t'], m['Nilai_p'])
        
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f'<div class="kpi-container"><div class="kpi-label">Akurasi (SMAPE)</div><div class="kpi-value" style="font-size:24px">{smape_val:.1f}%</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-container"><div class="kpi-label">Total Forecast</div><div class="kpi-value" style="font-size:24px">{fut["Nilai"].sum():,.0f}</div></div>', unsafe_allow_html=True)
    with k3:
        avg_sales = act['Nilai'].mean() if not act.empty else 0
        st.markdown(f'<div class="kpi-container"><div class="kpi-label">Rata-rata Historis</div><div class="kpi-value" style="font-size:24px">{avg_sales:,.0f}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=act['Tanggal'], y=act['Nilai'], name='Historis', line=dict(color='#334155', width=2)))
    fig.add_trace(go.Scatter(x=test['Tanggal'], y=test['Nilai'], name='Validasi', line=dict(color='#f59e0b', dash='dot')))
    
    if not fut.empty:
        x_c = [act['Tanggal'].iloc[-1], fut['Tanggal'].iloc[0]]
        y_c = [act['Nilai'].iloc[-1], fut['Nilai'].iloc[0]]
        fig.add_trace(go.Scatter(x=x_c, y=y_c, showlegend=False, line=dict(color='#3b82f6', width=3), hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=fut['Tanggal'], y=fut['Nilai'], name='Prediksi', line=dict(color='#3b82f6', width=3), mode='lines+markers'))
        
    fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis_title=None,
    yaxis_title=None,
    legend_title=None,
    height=450,
    margin=dict(t=20, l=0, r=0, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Tabel
    if not fut.empty:
        st.markdown("#### 📋 Tabel Prediksi")
        df_t = fut[['Tanggal', 'Nilai']].copy()
        df_t['Tanggal'] = df_t['Tanggal'].dt.strftime('%d %B %Y')
        df_t['Nilai'] = df_t['Nilai'].apply(lambda x: f"{x:,.0f}")
        df_t.columns = ['Periode', 'Prediksi (Pcs)']
        st.table(df_t)

# =========================================================
# 7. HALAMAN 3: PERBANDINGAN MODEL
# =========================================================
elif df_final is not None and st.session_state['page'] == "Perbandingan Model":
    
    st.title("Perbandingan Performa")
    st.markdown("Evaluasi akurasi model.")
    st.markdown("---")
    
    unique_models = df_final['Model'].unique()
    
    if len(unique_models) < 2:
        st.warning("⚠️ Hanya terdeteksi 1 Model.")
    else:
        cats = sorted(df_final['Kategori'].unique().tolist())
        opts = ["✨ Semua Kategori (Global)"] + cats
        sel_comp = st.selectbox("Lingkup Analisis", opts)
        
        comp_data = []
        breakdown = []
        
        if sel_comp == "✨ Semua Kategori (Global)":
            for m in unique_models:
                errs = []
                for c in cats:
                    d = df_final[(df_final['Kategori']==c) & (df_final['Model']==m)]
                    act = d[d['Jenis']=='Aktual']
                    test = d[d['Jenis']=='Test']
                    if not test.empty and not act.empty:
                        merged = pd.merge(test, act, on='Tanggal', suffixes=('_p','_t'))
                        if not merged.empty:
                            e = smape(merged['Nilai_t'], merged['Nilai_p'])
                            errs.append(e)
                            breakdown.append({'Kategori': c, 'Model': m, 'SMAPE': e})
                if errs: comp_data.append({'Model': m, 'SMAPE': sum(errs)/len(errs)})
        else:
            for m in unique_models:
                d = df_final[(df_final['Kategori']==sel_comp) & (df_final['Model']==m)]
                act = d[d['Jenis']=='Aktual']
                test = d[d['Jenis']=='Test']
                if not test.empty:
                    merged = pd.merge(test, act, on='Tanggal', suffixes=('_p','_t'))
                    if not merged.empty:
                        comp_data.append({'Model': m, 'SMAPE': smape(merged['Nilai_t'], merged['Nilai_p'])})

        if comp_data:
            df_comp = pd.DataFrame(comp_data).sort_values('SMAPE')
            best = df_comp.iloc[0]
            
            st.markdown(f"""
            <div class="summary-box" style="border-left-color: #10b981;">
                <div class="summary-title" style="color: #059669;">🏆 Rekomendasi: {best['Model']}</div>
                <p class="summary-text">
                    Model ini memiliki tingkat kesalahan rata-rata terendah sebesar <b>{best['SMAPE']:.2f}%</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if sel_comp == "✨ Semua Kategori (Global)" and breakdown:
                df_b = pd.DataFrame(breakdown)
                fig_b = px.bar(df_b, x='Kategori', y='SMAPE', color='Model', barmode='group', 
                               color_discrete_sequence=px.colors.qualitative.Safe)
                fig_b.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', height=400, yaxis_title="Error (%)", xaxis_title=None,
                    margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(orientation="h", y=1.1)
                )
                fig_b.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
                st.plotly_chart(fig_b, use_container_width=True)
                
                # Gallery
                st.markdown("---")
                st.subheader("Detail Grafik Per Kategori")
                for c in cats:
                    st.markdown(f"**{c}**")
                    fig_m = go.Figure()
                    base = df_final[(df_final['Kategori']==c) & (df_final['Model']==unique_models[0])]
                    act_data = base[base['Jenis']=='Aktual']
                    fig_m.add_trace(go.Scatter(x=act_data['Tanggal'], y=act_data['Nilai'], name='Aktual', line=dict(color='#334155', width=2)))
                    
                    for m in unique_models:
                        t_data = df_final[(df_final['Kategori']==c) & (df_final['Model']==m) & (df_final['Jenis']=='Test')]
                        width = 3 if m == best['Model'] else 1.5
                        dash = 'solid' if m == best['Model'] else 'dot'
                        fig_m.add_trace(go.Scatter(x=t_data['Tanggal'], y=t_data['Nilai'], name=m, line=dict(width=width, dash=dash)))
                    
                    fig_m.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=20), showlegend=True, plot_bgcolor='rgba(0,0,0,0)')
                    fig_m.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
                    st.plotly_chart(fig_m, use_container_width=True)
            else:
                fig_b = px.bar(df_comp, x='Model', y='SMAPE', color='Model', text_auto='.2f',
                               color_discrete_sequence=px.colors.qualitative.Safe)
                fig_b.update_layout(plot_bgcolor='rgba(0,0,0,0)', height=350, showlegend=False)
                st.plotly_chart(fig_b, use_container_width=True)

elif df_final is None:
    st.markdown("### 👋 Selamat Datang")
    st.info("Silakan pilih sumber data di menu sebelah kiri.")