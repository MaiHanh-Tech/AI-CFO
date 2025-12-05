import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from pypdf import PdfReader
from docx import Document
import io

# --- 1. CẤU HÌNH & TỪ ĐIỂN NGÔN NGỮ ---
st.set_page_config(page_title="AI Financial Controller", layout="wide", page_icon="💰")

# BỘ TỪ ĐIỂN (VIỆT - ANH - TRUNG)
TRANS = {
    "vi": {
        "title": "💰 AI Financial Controller (Hệ thống Kiểm soát Tài chính)",
        "sidebar_lang": "Ngôn ngữ / Language",
        "sidebar_source": "Nguồn Dữ Liệu",
        "opt_demo": "🎲 Dữ liệu Giả lập (Demo)",
        "opt_upload": "📂 Upload Excel Thực tế",
        "btn_sample": "Tạo dữ liệu mẫu",
        "success_load": "✅ Đã nạp {n} dòng dữ liệu.",
        "tab1": "📊 Dashboard Tổng Quan",
        "tab2": "🕵️ Soi Rủi Ro (ML)",
        "tab3": "🔮 Dự Báo (AI)",
        "tab4": "💬 Chat Tài Chính (RAG)",
        "metric_rev": "Tổng Doanh Thu",
        "metric_exp": "Tổng Chi Phí",
        "metric_net": "Lợi Nhuận Ròng",
        "btn_report_cn": "🇨🇳 Báo Cáo Sếp (Tiếng Trung)",
        "chart_cashflow": "Dòng tiền Doanh nghiệp",
        "chart_trend": "Xu hướng Lợi nhuận",
        "risk_header": "Phát hiện Giao dịch Bất thường (Anomaly Detection)",
        "risk_btn": "🔍 Quét Rủi Ro Ngay",
        "risk_warn": "⚠️ CẢNH BÁO: Hệ thống ML phát hiện {n} tháng bất thường!",
        "risk_ok": "✅ Hệ thống ML xác nhận số liệu ổn định.",
        "forecast_header": "Dự Báo Dòng Tiền (Linear Regression)",
        "forecast_trend": "Xu hướng:",
        "forecast_up": "TĂNG TRƯỞNG 🚀",
        "forecast_down": "SUY GIẢM 📉",
        "chat_header": "Hỏi đáp với Hồ sơ Tài chính (Đa định dạng)",
        "chat_upload": "Upload Báo cáo/Hợp đồng (PDF, Word, Txt)",
        "chat_input": "Hỏi gì đó về tài liệu này...",
    },
    "en": {
        "title": "💰 AI Financial Controller",
        "sidebar_lang": "Language",
        "sidebar_source": "Data Source",
        "opt_demo": "🎲 Demo Data (Simulation)",
        "opt_upload": "📂 Upload Real Excel",
        "btn_sample": "Generate Sample Data",
        "success_load": "✅ Loaded {n} rows.",
        "tab1": "📊 Dashboard",
        "tab2": "🕵️ Risk Detection (ML)",
        "tab3": "🔮 Forecasting (AI)",
        "tab4": "💬 Chat Finance (RAG)",
        "metric_rev": "Total Revenue",
        "metric_exp": "Total Expenses",
        "metric_net": "Net Profit",
        "btn_report_cn": "🇨🇳 Generate Chinese Report",
        "chart_cashflow": "Cash Flow",
        "chart_trend": "Profit Trend",
        "risk_header": "Anomaly Detection System",
        "risk_btn": "🔍 Scan for Risks",
        "risk_warn": "⚠️ WARNING: ML detected {n} abnormal months!",
        "risk_ok": "✅ ML System confirmed stable data.",
        "forecast_header": "Cash Flow Forecasting (Linear Regression)",
        "forecast_trend": "Trend:",
        "forecast_up": "GROWTH 🚀",
        "forecast_down": "DECLINE 📉",
        "chat_header": "Chat with Financial Documents",
        "chat_upload": "Upload Reports/Contracts (PDF, Word, Txt)",
        "chat_input": "Ask something about this document...",
    },
    "zh": {
        "title": "💰 AI 财务控制系统 (CFO Assistant)",
        "sidebar_lang": "语言 / Language",
        "sidebar_source": "数据源",
        "opt_demo": "🎲 模拟数据 (Demo)",
        "opt_upload": "📂 上传 Excel",
        "btn_sample": "生成样本数据",
        "success_load": "✅ 已加载 {n} 行数据。",
        "tab1": "📊 财务概览",
        "tab2": "🕵️ 风险检测 (ML)",
        "tab3": "🔮 预测 (AI)",
        "tab4": "💬 财务对话 (RAG)",
        "metric_rev": "总收入",
        "metric_exp": "总支出",
        "metric_net": "净利润",
        "btn_report_cn": "🇨🇳 生成中文汇报",
        "chart_cashflow": "企业现金流",
        "chart_trend": "利润趋势",
        "risk_header": "异常交易检测 (Anomaly Detection)",
        "risk_btn": "🔍 立即扫描风险",
        "risk_warn": "⚠️ 警告：ML 系统发现 {n} 个异常月份！",
        "risk_ok": "✅ ML 系统确认数据稳定。",
        "forecast_header": "现金流预测 (线性回归)",
        "forecast_trend": "趋势:",
        "forecast_up": "增长 🚀",
        "forecast_down": "下降 📉",
        "chat_header": "财务文档问答",
        "chat_upload": "上传报告/合同 (PDF, Word, Txt)",
        "chat_input": "关于此文档的问题...",
    }
}

# Hàm lấy text đa ngôn ngữ
def T(key):
    lang_code = st.session_state.get('lang_code', 'vi')
    return TRANS[lang_code].get(key, key)

# --- 2. CẤU HÌNH GEMINI ---
try:
    if 'system' in st.secrets: api_key = st.secrets['system']['gemini_api_key']
    elif 'api_keys' in st.secrets: api_key = st.secrets['api_keys']['gemini_api_key']
    else: st.stop()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
except: st.warning("Chưa cấu hình API Key.")

# --- 3. CÁC HÀM XỬ LÝ (CORE) ---

# Hàm đọc đa định dạng (Kế thừa từ App Sách)
def doc_tai_lieu_da_nang(uploaded_file):
    if not uploaded_file: return ""
    # Lấy đuôi file
    ext = uploaded_file.name.split('.')[-1].lower()
    text = ""
    try:
        if ext == 'pdf':
            reader = PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in reader.pages])
        elif ext == 'docx':
            doc = Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext in ['txt', 'md', 'csv']:
            text = str(uploaded_file.read(), "utf-8")
        else:
            return "Định dạng không hỗ trợ."
    except Exception as e: return f"Lỗi đọc file: {e}"
    
    return text

# ML: Phát hiện gian lận
def phat_hien_bat_thuong(df):
    model_iso = IsolationForest(contamination=0.05, random_state=42)
    # Cần đảm bảo tên cột đúng (Giả sử cột 2 là Chi Phí)
    col_chi_phi = df.columns[2] 
    df['Anomaly'] = model_iso.fit_predict(df[[col_chi_phi]])
    return df[df['Anomaly'] == -1]

# ML: Dự báo
def du_bao_tuong_lai(df):
    df['Thang_Num'] = range(len(df))
    X = df[['Thang_Num']]
    y = df.iloc[:, 3] # Cột Lợi nhuận (Giả định cột 3)
    
    reg = LinearRegression().fit(X, y)
    future_months = np.array([[len(df)], [len(df)+1], [len(df)+2]])
    future_pred = reg.predict(future_months)
    return future_pred, reg.coef_[0]

def tao_du_lieu_mau():
    dates = pd.date_range(start="2023-01-01", periods=24, freq="ME")
    data = {
        "Tháng": dates,
        "Doanh Thu": np.random.randint(800, 1500, size=24) * 1000,
        "Chi Phí": np.random.randint(500, 1000, size=24) * 1000,
    }
    df = pd.DataFrame(data)
    df["Lợi Nhuận"] = df["Doanh Thu"] - df["Chi Phí"]
    df.loc[10, "Chi Phí"] = 2000000 
    return df

# --- 4. GIAO DIỆN APP ---

# Sidebar: Chọn Ngôn ngữ & Dữ liệu
with st.sidebar:
    # 1. Chọn Ngôn ngữ
    lang_map = {"Tiếng Việt": "vi", "English": "en", "中文": "zh"}
    sel_lang = st.selectbox("🌐 " + T("sidebar_lang"), list(lang_map.keys()))
    st.session_state.lang_code = lang_map[sel_lang]
    
    st.divider()
    
    # 2. Chọn Nguồn Dữ liệu
    st.header(f"🗂️ {T('sidebar_source')}")
    source = st.radio("", [T("opt_demo"), T("opt_upload")])
    
    df = None
    if source == T("opt_demo"):
        if st.button(T("btn_sample")):
            st.session_state.df_fin = tao_du_lieu_mau()
    else:
        up_file = st.file_uploader("Excel (Month, Rev, Exp)", type=['xlsx'])
        if up_file: st.session_state.df_fin = pd.read_excel(up_file)

    if 'df_fin' in st.session_state:
        df = st.session_state.df_fin
        st.success(T("success_load").format(n=len(df)))

st.title(T("title"))

# Main Content
if df is not None:
    t1, t2, t3, t4 = st.tabs([T("tab1"), T("tab2"), T("tab3"), T("tab4")])

    # TAB 1: DASHBOARD
    with t1:
        tong_thu = df.iloc[:, 1].sum()
        tong_chi = df.iloc[:, 2].sum()
        ln_tong = tong_thu - tong_chi
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(T("metric_rev"), f"{tong_thu:,.0f}")
        c2.metric(T("metric_exp"), f"{tong_chi:,.0f}")
        c3.metric(T("metric_net"), f"{ln_tong:,.0f}")
        
        with c4:
            st.write("")
            if st.button(T("btn_report_cn"), type="primary"):
                with st.spinner("AI writing..."):
                    prompt = f"Role: CFO. Data: Rev {tong_thu}, Exp {tong_chi}, Profit {ln_tong}. Write a short report in Business Chinese."
                    res = model.generate_content(prompt)
                    st.info(res.text)

        col1, col2 = st.columns([3, 1])
        with col1:
            fig = px.bar(df, x=df.columns[0], y=[df.columns[1], df.columns[2]], barmode="group", title=T("chart_cashflow"))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.line(df, x=df.columns[0], y=df.columns[3], title=T("chart_trend"))
            st.plotly_chart(fig2, use_container_width=True)

    # TAB 2: ML ANOMALY
    with t2:
        st.header(T("risk_header"))
        if st.button(T("risk_btn")):
            bat_thuong = phat_hien_bat_thuong(df.copy())
            if not bat_thuong.empty:
                st.error(T("risk_warn").format(n=len(bat_thuong)))
                st.dataframe(bat_thuong.style.highlight_max(axis=0, color='pink'))
                
                # AI Giải thích
                prompt = f"Analyze these anomalies (Finance data): {bat_thuong.to_string()}. Language: {st.session_state.lang_code}. Give possible reasons."
                res = model.generate_content(prompt)
                st.markdown(res.text)
            else:
                st.success(T("risk_ok"))

    # TAB 3: FORECAST
    with t3:
        st.header(T("forecast_header"))
        pred, trend = du_bao_tuong_lai(df)
        xu_huong = T("forecast_up") if trend > 0 else T("forecast_down")
        
        st.metric(T("forecast_trend"), xu_huong)
        st.write("Forecast (Next 3 months):")
        c_f1, c_f2, c_f3 = st.columns(3)
        c_f1.metric("Month +1", f"{pred[0]:,.0f}")
        c_f2.metric("Month +2", f"{pred[1]:,.0f}")
        c_f3.metric("Month +3", f"{pred[2]:,.0f}")

    # TAB 4: CHAT WITH DOCS (RAG LITE)
    with t4:
        st.header(T("chat_header"))
        # Cho phép nhiều định dạng
        uploaded_doc = st.file_uploader(T("chat_upload"), type=["pdf", "docx", "txt"])
        
        if uploaded_doc:
            text_doc = doc_tai_lieu_da_nang(uploaded_doc)
            st.info(f"📄 Loaded: {len(text_doc)} chars")
            
            question = st.chat_input(T("chat_input"))
            if question:
                with st.chat_message("user"): st.write(question)
                with st.chat_message("assistant"):
                    with st.spinner("AI thinking..."):
                        prompt = f"Document Content: {text_doc[:30000]}. User Question: {question}. Language: {st.session_state.lang_code}. Answer as a CFO."
                        res = model.generate_content(prompt)
                        st.markdown(res.text)

else:
    st.info("👈 Please select Data Source / Vui lòng chọn Nguồn dữ liệu.")
