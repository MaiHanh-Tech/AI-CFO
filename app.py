import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import numpy as np
from datetime import datetime, timedelta

# --- 1. CẤU HÌNH ---
st.set_page_config(page_title="AI Financial Controller", layout="wide", page_icon="💰")

# Cấu hình Gemini (Lấy key từ Secrets cũ của chị)
try:
    if 'system' in st.secrets and 'gemini_api_key' in st.secrets['system']:
        api_key = st.secrets['system']['gemini_api_key']
    elif 'api_keys' in st.secrets and 'gemini_api_key' in st.secrets['api_keys']:
        api_key = st.secrets['api_keys']['gemini_api_key']
    else:
        st.error("Chưa thấy API Key trong Secrets. Vui lòng kiểm tra lại.")
        st.stop()
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Lỗi cấu hình: {e}")

# --- 2. HÀM TẠO DỮ LIỆU GIẢ LẬP (ĐỂ ĐI PHỎNG VẤN) ---
def tao_du_lieu_mau():
    dates = pd.date_range(start="2024-01-01", periods=12, freq="ME")
    data = {
        "Tháng": dates,
        "Doanh Thu (RMB)": np.random.randint(500000, 1000000, size=12),
        "Chi Phí (RMB)": np.random.randint(300000, 800000, size=12),
    }
    df = pd.DataFrame(data)
    df["Lợi Nhuận"] = df["Doanh Thu (RMB)"] - df["Chi Phí (RMB)"]
    # Tạo một tháng đột biến chi phí (để demo tính năng bắt lỗi)
    df.loc[5, "Chi Phí (RMB)"] = df.loc[5, "Doanh Thu (RMB)"] + 50000 
    return df

# --- 3. GIAO DIỆN CHÍNH ---
st.title("💰 AI Financial Controller (Hệ thống Kiểm soát Tài chính)")
st.caption("Dành cho Kế toán trưởng - Tích hợp Báo cáo Song ngữ Việt/Trung")

# Sidebar
with st.sidebar:
    st.header("🗂️ Nguồn Dữ Liệu")
    data_option = st.radio("Chọn nguồn:", ["📂 Upload Excel Thật", "🎲 Dùng Số Liệu Mẫu (Demo)"])
    
    df = None
    if data_option == "📂 Upload Excel Thật":
        uploaded_file = st.file_uploader("Upload file Excel (Cột: Tháng, Doanh Thu, Chi Phí)", type=['xlsx'])
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
            except: st.error("Lỗi đọc file.")
    else:
        if st.button("Tạo Dữ Liệu Mẫu"):
            df = tao_du_lieu_mau()
            st.session_state.df_finance = df
        
        if 'df_finance' in st.session_state:
            df = st.session_state.df_finance

# --- 4. XỬ LÝ CHÍNH ---
if df is not None:
    # Dashboard
    tong_thu = df.iloc[:, 1].sum()
    tong_chi = df.iloc[:, 2].sum()
    loi_nhuan = tong_thu - tong_chi
    margin = (loi_nhuan / tong_thu) * 100
    
    # 3 Cột chỉ số (Metrics)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng Doanh Thu", f"¥{tong_thu:,.0f}", help="Total Revenue")
    c2.metric("Tổng Chi Phí", f"¥{tong_chi:,.0f}", help="Total Expense")
    c3.metric("Lợi Nhuận Ròng", f"¥{loi_nhuan:,.0f}", f"{margin:.1f}% Margin")
    
    # Nút bấm thần thánh: Dịch sang tiếng Trung
    with c4:
        st.write("")
        if st.button("🇨🇳 Báo Cáo Sếp (Tiếng Trung)", type="primary"):
            with st.spinner("AI đang viết báo cáo..."):
                prompt = f"""
                Bạn là Kế toán trưởng chuyên nghiệp.
                Dựa trên số liệu: Doanh thu {tong_thu}, Chi phí {tong_chi}, Lợi nhuận {loi_nhuan}.
                
                Hãy viết một đoạn báo cáo ngắn gọn (khoảng 50 chữ) bằng **TIẾNG TRUNG QUỐC THƯƠNG MẠI** gửi Tổng Giám đốc.
                Nhận xét tình hình tài chính và đưa ra 1 lời khuyên.
                """
                res = model.generate_content(prompt)
                st.session_state.report_cn = res.text
    
    if 'report_cn' in st.session_state:
        st.success("📩 **Báo cáo Tiếng Trung:**")
        st.info(st.session_state.report_cn)

    # --- TABS ---
    tab1, tab2 = st.tabs(["📊 Biểu Đồ Phân Tích", "🕵️ Soát Xét Rủi Ro"])
    
    with tab1:
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            # Biểu đồ kết hợp
            st.subheader("Xu hướng Dòng tiền")
            fig = px.bar(df, x=df.columns[0], y=[df.columns[1], df.columns[2]], barmode='group', title="Thu vs Chi")
            st.plotly_chart(fig, use_container_width=True)
        
        with c_chart2:
            st.subheader("Cơ cấu Lợi nhuận")
            # Tạo cột màu sắc: Xanh nếu lời, Đỏ nếu lỗ
            df["Color"] = np.where(df.iloc[:, 3] < 0, 'Lỗ', 'Lời')
            fig2 = px.bar(df, x=df.columns[0], y=df.columns[3], color="Color", title="Biến động Lợi nhuận tháng",
                          color_discrete_map={'Lỗ': 'red', 'Lời': 'green'})
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.header("Hệ thống Cảnh báo Sớm (Anomaly Detection)")
        st.caption("Tự động phát hiện các tháng có chi phí bất thường vượt ngưỡng trung bình.")
        
        # Logic phát hiện rủi ro đơn giản
        col_chi_phi = df.columns[2]
        trung_binh_chi = df[col_chi_phi].mean()
        nguong_canh_bao = trung_binh_chi * 1.2 # Cảnh báo nếu vượt 120% trung bình
        
        bat_thuong = df[df[col_chi_phi] > nguong_canh_bao]
        
        c_risk, c_advice = st.columns([2, 1])
        
        with c_risk:
            if not bat_thuong.empty:
                st.error(f"⚠️ CẢNH BÁO: Có {len(bat_thuong)} tháng chi tiêu vượt mức!")
                st.dataframe(bat_thuong.style.highlight_max(axis=0, color='pink'))
            else:
                st.success("✅ Số liệu ổn định. Không có bất thường.")
                
        with c_advice:
            if not bat_thuong.empty:
                if st.button("🤖 AI Phân Tích Nguyên Nhân"):
                    data_str = bat_thuong.to_string()
                    prompt_risk = f"Dữ liệu chi phí bất thường: {data_str}. Hãy đóng vai Kế toán trưởng, đưa ra 3 giả thuyết về nguyên nhân và giải pháp bằng Tiếng Việt."
                    with st.spinner("Đang điều tra..."):
                        res_risk = model.generate_content(prompt_risk)
                        st.warning(res_risk.text)

else:
    st.info("👈 Mời Chị chọn 'Dùng Số Liệu Mẫu' bên trái để xem Demo.")
