import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Eczane Stok Yönetim Sistemi", page_icon="💊", layout="wide")
st.title("💊 Eczane Stok Yönetim Sistemi")
st.markdown("ABC-VED Hibrit Modeli ile Akıllı Stok Optimizasyonu")

st.sidebar.header("🔧 Ayarlar")
st.sidebar.file_uploader("Veri Yükle (CSV)", type="csv", key="file_uploader")
st.sidebar.slider("Güvenlik Stoku Yüzdesi", 10, 50, 20, key="safety_stock")

@st.cache_data
def load_sample_data():
    data = {
        "ilac_adi": ["İnsülin", "Kardiyoloji İlacı", "Ağrı Kesici", "Vitamin Takviyesi", "Antibiyotik", "Alerji İlacı", "Tansiyon İlacı", "Cilt Kremi", "Grip İlacı", "Şeker Ölçüm Cihazı"],
        "aylik_ortalama_tuketim": [15, 30, 100, 50, 40, 60, 80, 20, 70, 10],
        "birim_fiyat": [250, 120, 25, 40, 90, 35, 70, 60, 45, 150],
        "ved_durumu": ["V", "E", "E", "D", "V", "D", "E", "D", "E", "V"],
        "mevcut_stok": [15, 25, 80, 40, 35, 50, 65, 15, 55, 8],
        "gecmis_satislar": [
            [12,14,16,15,13,17,16,15,14,18,16,17],
            [28,30,32,31,29,33,32,30,31,34,32,33],
            [95,100,105,98,102,110,108,105,100,115,112,110],
            [45,48,50,47,49,52,51,50,48,55,53,52],
            [38,40,42,41,39,43,42,40,41,44,42,43],
            [55,58,60,57,59,62,61,60,58,65,63,62],
            [75,78,80,77,79,82,81,80,78,85,83,82],
            [18,20,22,21,19,23,22,20,21,24,22,23],
            [65,68,70,67,69,72,71,70,68,75,73,72],
            [9,10,11,10,9,12,11,10,10,13,12,12]
        ]
    }
    return pd.DataFrame(data)

df = load_sample_data()

def abc_analysis(df):
    df = df.copy()
    df["yillik_tuketim_degeri"] = df["aylik_ortalama_tuketim"] * df["birim_fiyat"] * 12
    df = df.sort_values(by="yillik_tuketim_degeri", ascending=False)
    df["kumulatif_yuzde"] = df["yillik_tuketim_degeri"].cumsum() / df["yillik_tuketim_degeri"].sum() * 100
    
    def abc_classify(kumulatif):
        if kumulatif <= 80: return "A"
        elif kumulatif <= 95: return "B"
        else: return "C"
    
    df["abc_sinifi"] = df["kumulatif_yuzde"].apply(abc_classify)
    return df

def forecast_sales(sales_data, steps=3):
    try:
        model = ARIMA(sales_data, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except:
        return np.mean(sales_data) * np.ones(steps)

def rule_engine(row, forecast_values):
    abc_class = row["abc_sinifi"]
    ved_status = row["ved_durumu"]
    current_stock = row["mevcut_stok"]
    forecast_next_month = forecast_values[0]
    
    if abc_class == "A" and ved_status == "V" and forecast_next_month > current_stock:
        return "🚨 KRİTİK - Acil sipariş ver!"
    elif abc_class == "A" and ved_status == "E" and forecast_next_month > current_stock:
        return "⚠️ ÖNEMLİ - Sipariş ver"
    elif abc_class == "B" and ved_status == "V" and forecast_next_month > current_stock:
        return "🔶 ORTA - Sipariş planla"
    elif forecast_next_month > current_stock:
        return "🔷 DÜŞÜK - Gözlemle"
    else:
        return "✅ YETERLİ - Stok yeterli"

def main():
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Genel Bakış", "🔍 Detaylı Analiz", "📈 Tahminler", "⚡ Öneriler"])
    
    with tab1:
        st.header("Stok Durumu Genel Bakış")
        df_abc = abc_analysis(df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Toplam İlaç Sayısı", len(df))
        with col2: st.metric("A Sınıfı İlaçlar", len(df_abc[df_abc["abc_sinifi"] == "A"]))
        with col3: st.metric("Vital İlaçlar", len(df[df["ved_durumu"] == "V"]))
        with col4: st.metric("Kritik Riskli İlaçlar", len(df_abc[(df_abc["abc_sinifi"] == "A") & (df_abc["ved_durumu"] == "V")]))
        
        st.subheader("ABC-VED Sınıflandırma Matrisi")
        st.dataframe(df_abc[["ilac_adi", "abc_sinifi", "ved_durumu", "yillik_tuketim_degeri", "mevcut_stok"]])
    
    with tab2:
        st.header("Detaylı İlaç Analizi")
        selected_drug = st.selectbox("İlaç Seçin", df["ilac_adi"])
        drug_data = df[df["ilac_adi"] == selected_drug].iloc[0]
        abc_data = abc_analysis(df)
        abc_class = abc_data[abc_data["ilac_adi"] == selected_drug]["abc_sinifi"].values[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**İlaç:** {drug_data['ilac_adi']}")
            st.info(f"**ABC Sınıfı:** {abc_class}")
            st.info(f"**VED Durumu:** {drug_data['ved_durumu']}")
        with col2:
            st.info(f"**Mevcut Stok:** {drug_data['mevcut_stok']}")
            st.info(f"**Aylık Tüketim:** {drug_data['aylik_ortalama_tuketim']}")
            st.info(f"**Birim Fiyat:** {drug_data['birim_fiyat']} ₺")
        
        st.subheader("Son 12 Ay Satış Grafiği")
        fig, ax = plt.subplots()
        ax.plot(drug_data["gecmis_satislar"], marker="o")
        ax.set_title(f"{selected_drug} Satış Geçmişi")
        ax.set_xlabel("Aylar")
        ax.set_ylabel("Satış Miktarı")
        st.pyplot(fig)
    
    with tab3:
        st.header("Talep Tahminleri")
        st.subheader("3 Aylık Talep Tahminleri")
        
        forecast_results = []
        for _, row in df.iterrows():
            forecast = forecast_sales(row["gecmis_satislar"])
            forecast_results.append({
                "İlaç": row["ilac_adi"],
                "Mevcut Stok": row["mevcut_stok"],
                "Tahmin (1. Ay)": round(forecast[0], 1),
                "Tahmin (2. Ay)": round(forecast[1], 1),
                "Tahmin (3. Ay)": round(forecast[2], 1)
            })
        
        forecast_df = pd.DataFrame(forecast_results)
        st.dataframe(forecast_df)
    
    with tab4:
        st.header("Stok Önerileri")
        recommendations = []
        df_abc = abc_analysis(df)
        
        for _, row in df_abc.iterrows():
            sales_data = df[df["ilac_adi"] == row["ilac_adi"]]["gecmis_satislar"].values[0]
            forecast_values = forecast_sales(sales_data)
            recommendation = rule_engine(row, forecast_values)
            
            recommendations.append({
                "İlaç": row["ilac_adi"],
                "ABC": row["abc_sinifi"],
                "VED": row["ved_durumu"],
                "Mevcut Stok": row["mevcut_stok"],
                "Tahmin": round(forecast_values[0], 1),
                "Öneri": recommendation
            })
        
        rec_df = pd.DataFrame(recommendations)
        
        def color_critical(val):
            if "🚨 KRİTİK" in val: return "background-color: #ffcccc; color: #cc0000; font-weight: bold"
            elif "⚠️ ÖNEMLİ" in val: return "background-color: #fff0cc; color: #cc8800; font-weight: bold"
            elif "🔶 ORTA" in val: return "background-color: #ffffcc; color: #888800; font-weight: bold"
            elif "🔷 DÜŞÜK" in val: return "background-color: #ccffcc; color: #008800; font-weight: bold"
            else: return "background-color: #e6ffe6; color: #006600; font-weight: bold"
        
        styled_df = rec_df.style.applymap(color_critical, subset=["Öneri"])
        st.dataframe(styled_df, height=400)
        
        st.subheader("Öneri Özeti")
        critical_count = len(rec_df[rec_df["Öneri"].str.contains("KRİTİK")])
        important_count = len(rec_df[rec_df["Öneri"].str.contains("ÖNEMLİ")])
        
        if critical_count > 0: st.error(f"🚨 {critical_count} KRİTİK uyarı! Acil müdahale gerekiyor.")
        if important_count > 0: st.warning(f"⚠️ {important_count} önemli uyarı! Sipariş planlaması yapılmalı.")

if __name__ == "__main__":
    main()
