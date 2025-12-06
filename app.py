import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import requests
import io

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="AI Financial Controller Ultimate", layout="wide", page_icon="💰")

# --- TỪ ĐIỂN ĐA NGÔN NGỮ ---
TRANS = {
    "vi": {
        "title": "💰 Hệ thống Giám đốc Tài chính AI (CFO Ultimate)",
        "role_admin": "CFO (Giám đốc Tài chính)",
        "role_chief": "Kế toán trưởng",
        "role_staff": "Kế toán viên",
        "tab1": "📊 Bộ Chỉ Số KPIs",
        "tab2": "📉 Phân Tích Chi Phí",
        "tab3": "🕵️ Soát Xét Rủi Ro (ML)",
        "tab4": "🔮 Chiến Lược & Dự Báo",
        "tab5": "⚖️ Trung Tâm Pháp Chế",
        "kpi_select": "Chọn Nhóm Chỉ Số:",
        "grp_liquid": "1. Khả năng Thanh toán",
        "grp_profit": "2. Khả năng Sinh lời",
        "grp_activity": "3. Hiệu quả Hoạt động",
        "grp_struct": "4. Cấu trúc Vốn",
        "btn_cn": "🇨🇳 Báo Cáo Sếp (Tiếng Trung)",
        "logout": "Đăng xuất"
    },
    "en": {
        "title": "💰 AI Financial Controller Ultimate",
        "role_admin": "CFO",
        "role_chief": "Chief Accountant",
        "role_staff": "Staff",
        "tab1": "📊 Financial KPIs",
        "tab2": "📉 Cost Analysis",
        "tab3": "🕵️ Risk Audit (ML)",
        "tab4": "🔮 Forecast Strategy",
        "tab5": "⚖️ Legal Hub",
        "kpi_select": "Select KPI Group:",
        "grp_liquid": "1. Liquidity",
        "grp_profit": "2. Profitability",
        "grp_activity": "3. Activity",
        "grp_struct": "4. Capital Structure",
        "btn_cn": "🇨🇳 Generate Chinese Report",
        "logout": "Logout"
    },
    "zh": {
        "title": "💰 AI 财务总监控制系统",
        "role_admin": "财务总监 (CFO)",
        "role_chief": "财务经理",
        "role_staff": "会计",
        "tab1": "📊 财务指标 (KPIs)",
        "tab2": "📉 成本分析",
        "tab3": "🕵️ 风险审计 (ML)",
        "tab4": "🔮 战略预测",
        "tab5": "⚖️ 法律中心",
        "kpi_select": "选择指标组:",
        "grp_liquid": "1. 偿债能力",
        "grp_profit": "2. 盈利能力",
        "grp_activity": "3. 营运能力",
        "grp_struct": "4. 资本结构",
        "btn_cn": "🇨🇳 生成深度财务报告",
        "logout": "登出"
    }
}

def T(key):
    lang = st.session_state.get('lang_code', 'vi')
    return TRANS[lang].get(key, key)

# --- 2. AUTH MANAGER ---
class AuthManager:
    def __init__(self):
        self.users = st.secrets.get("users", {})
        self.roles = st.secrets.get("roles", {})
    def verify(self, u, p): return u in self.users and self.users[u] == p
    def get_role(self, u): return self.roles.get(u, "staff")

# --- 3. CẤU HÌNH GEMINI (THÔNG MINH) ---
try:
    # 1. Lấy API Key
    if 'system' in st.secrets: api_key = st.secrets['system']['gemini_api_key']
    elif 'api_keys' in st.secrets: api_key = st.secrets['api_keys']['gemini_api_key']
    else: st.stop()
    
    genai.configure(api_key=api_key)
    
    # 2. Chọn Model (Ưu tiên Pro, trượt về Flash nếu lỗi)
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
    except:
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
        except:
            model = genai.GenerativeModel('gemini-pro') 
except: pass

# --- 4. LOGIC TÀI CHÍNH & DỮ LIỆU ---

def tao_data_full_kpi():
    dates = pd.date_range(start="2024-01-01", periods=12, freq="ME")
    df = pd.DataFrame({"Tháng": dates})
    
    # P&L
    df["Doanh Thu"] = np.random.randint(5000, 8000, 12) * 1000000
    df["Giá Vốn (Trực tiếp)"] = df["Doanh Thu"] * 0.6 
    df["Chi Phí VH (Gián tiếp)"] = np.random.randint(500, 800, 12) * 1000000
    df["Lợi Nhuận Gộp"] = df["Doanh Thu"] - df["Giá Vốn (Trực tiếp)"]
    df["Lợi Nhuận ST"] = df["Lợi Nhuận Gộp"] - df["Chi Phí VH (Gián tiếp)"]
    
    # Balance Sheet
    df["TS Ngắn Hạn"] = np.random.randint(2000, 3000, 12) * 1000000
    df["Nợ Ngắn Hạn"] = np.random.randint(1000, 1500, 12) * 1000000
    df["Hàng Tồn Kho"] = np.random.randint(800, 1200, 12) * 1000000
    df["Phải Thu KH"] = np.random.randint(1000, 1500, 12) * 1000000
    df["Tiền Mặt"] = df["TS Ngắn Hạn"] - df["Hàng Tồn Kho"] - df["Phải Thu KH"]
    df["Tiền Mặt"] = df["Tiền Mặt"].apply(lambda x: max(x, 100000000))
    df["TS Dài Hạn"] = 5000000000 
    df["Tổng Tài Sản"] = df["TS Ngắn Hạn"] + df["TS Dài Hạn"]
    df["Nợ Dài Hạn"] = 1000000000
    df["Tổng Nợ"] = df["Nợ Ngắn Hạn"] + df["Nợ Dài Hạn"]
    df["Vốn Chủ Sở Hữu"] = df["Tổng Tài Sản"] - df["Tổng Nợ"]
    
    # Gài bẫy cho ML
    df.loc[5, "Chi Phí VH (Gián tiếp)"] = 2500000000
    return df

def tinh_chi_so_tai_chinh(df):
    required = ["TS Ngắn Hạn", "Nợ Ngắn Hạn", "Hàng Tồn Kho", "Doanh Thu", "Tổng Tài Sản", "Vốn Chủ Sở Hữu"]
    for c in required:
        if c not in df.columns: df[c] = 1 
    try:
        # 1. Thanh khoản
        df["Current Ratio"] = df["TS Ngắn Hạn"] / df["Nợ Ngắn Hạn"].replace(0, 1)
        df["Quick Ratio"] = (df["TS Ngắn Hạn"] - df.get("Hàng Tồn Kho", 0)) / df["Nợ Ngắn Hạn"].replace(0, 1)
        
        # 2. Hoạt động
        df["Inv Turnover"] = df.get("Giá Vốn (Trực tiếp)", 0) / df.get("Hàng Tồn Kho", 1).replace(0, 1)
        df["Asset Turnover"] = df["Doanh Thu"] / df["Tổng Tài Sản"].replace(0, 1)
        df["AR Turnover"] = df["Doanh Thu"] / df.get("Phải Thu KH", 1).replace(0, 1)
        
        # 3. Sinh lời
        df["Gross Margin"] = (df.get("Lợi Nhuận Gộp", 0) / df["Doanh Thu"].replace(0, 1)) * 100
        df["ROS"] = (df.get("Lợi Nhuận ST", 0) / df["Doanh Thu"].replace(0, 1)) * 100
        df["ROE"] = (df.get("Lợi Nhuận ST", 0) / df["Vốn Chủ Sở Hữu"].replace(0, 1)) * 100
        df["ROA"] = (df.get("Lợi Nhuận ST", 0) / df["Tổng Tài Sản"].replace(0, 1)) * 100
        
        # 4. Cấu trúc vốn
        df["Debt/Asset"] = (df.get("Tổng Nợ", 0) / df["Tổng Tài Sản"].replace(0, 1)) * 100
        df["Debt/Equity"] = (df.get("Tổng Nợ", 0) / df["Vốn Chủ Sở Hữu"].replace(0, 1)) * 100
    except: pass
    return df

# --- 5. CÁC HÀM ĐỌC & ML ---
def doc_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])])
        return text[:20000]
    except Exception as e: return f"Lỗi Web: {e}"

def doc_tai_lieu(uploaded_file):
    try:
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext == 'pdf': return "\n".join([p.extract_text() for p in PdfReader(uploaded_file).pages])
        elif ext == 'docx': return "\n".join([p.text for p in Document(uploaded_file).paragraphs])
        elif ext in ['txt', 'md']: return str(uploaded_file.read(), "utf-8")
    except: return ""
    return ""

def phat_hien_gian_lan_ml(df):
    model_iso = IsolationForest(contamination=0.1, random_state=42)
    col = "Chi Phí VH (Gián tiếp)" if "Chi Phí VH (Gián tiếp)" in df.columns else df.columns[2]
    df['Anomaly'] = model_iso.fit_predict(df[[col]])
    return df[df['Anomaly'] == -1]

# --- 6. GIAO DIỆN CHÍNH ---
def show_dashboard():
    with st.sidebar:
        # Chọn ngôn ngữ
        lang_map = {"Tiếng Việt": "vi", "English": "en", "中文": "zh"}
        sel = st.selectbox("🌐 Language", list(lang_map.keys()))
        st.session_state.lang_code = lang_map[sel]
        
        st.divider()
        role = st.session_state.user_role
        role_name = "role_" + role if role in ["admin", "chief", "staff"] else "role_staff"
        st.success(f"👤 {st.session_state.username} | 🔰 {T(role_name)}")
        
        st.header("🗂️ Data Source")
        if st.button("🔄 Tạo Dữ Liệu Mẫu (Full KPIs)", type="primary"):
            st.session_state.df_fin = tao_data_full_kpi()
            st.rerun()
        
        up = st.file_uploader("Upload Excel", type=['xlsx'])
        if up: st.session_state.df_fin = pd.read_excel(up)

        if st.button(T("logout")):
            st.session_state.is_logged_in = False; st.rerun()

    st.title(T("title"))

    if 'df_fin' not in st.session_state:
        st.info("👈 Mời Giám đốc bấm nút 'Tạo Dữ Liệu Mẫu' để xem demo.")
        return

    # TÍNH TOÁN
    df = tinh_chi_so_tai_chinh(st.session_state.df_fin.copy())
    last = df.iloc[-1]
    is_vip = role in ["admin", "chief"]
    
    t1, t2, t3, t4, t5 = st.tabs([T("tab1"), T("tab2"), T("tab3"), T("tab4"), T("tab5")])

    # === TAB 1: BỘ CHỈ SỐ FULL (ĐẦY ĐỦ NHƯ YÊU CẦU) ===
    with t1:
        st.subheader("Phân tích Hoạt động Kinh tế & Tài chính")
        opts = [T("grp_liquid"), T("grp_profit"), T("grp_activity"), T("grp_struct")]
        sels = st.multiselect(T("kpi_select"), opts, default=opts)
        
        c1, c2, c3 = st.columns(3)
        if T("grp_liquid") in sels:
            c1.markdown(f"#### 💧 {T('grp_liquid')}")
            c1.metric("Thanh toán HH", f"{last.get('Current Ratio', 0):.2f}")
            c1.metric("Thanh toán Nhanh", f"{last.get('Quick Ratio', 0):.2f}")
            
        if T("grp_profit") in sels:
            c2.markdown(f"#### 💰 {T('grp_profit')}")
            c2.metric("Gross Margin", f"{last.get('Gross Margin', 0):.1f}%")
            c2.metric("ROE", f"{last.get('ROE', 0):.1f}%")
            
        if T("grp_activity") in sels:
            c3.markdown(f"#### 🏭 {T('grp_activity')}")
            c3.metric("Vòng quay Tồn kho", f"{last.get('Inv Turnover', 0):.2f}")
            c3.metric("Vòng quay Phải thu", f"{last.get('AR Turnover', 0):.2f}")
        
        # Phần cấu trúc vốn
        if T("grp_struct") in sels:
            st.divider()
            st.markdown(f"#### ⚖️ {T('grp_struct')}")
            k1, k2 = st.columns(2)
            k1.metric("Debt/Equity", f"{last.get('Debt/Equity', 0):.1f}%")
            k2.metric("Debt/Asset", f"{last.get('Debt/Asset', 0):.1f}%")

        if is_vip:
            st.markdown("---")
            if st.button(T("btn_cn"), type="primary"):
                with st.spinner("Writing..."):
                    p = f"Role: CFO. Data: {last.to_dict()}. Write Business Chinese report."
                    res = model.generate_content(p)
                    st.info(res.text)

    # === TAB 2: CHI PHÍ & BIỂU ĐỒ ===
    with t2:
        st.subheader("Phân tích Chi phí")
        c1, c2 = st.columns([2, 1])
        with c1:
            cols = [c for c in ["Giá Vốn (Trực tiếp)", "Chi Phí VH (Gián tiếp)"] if c in df.columns]
            if cols: st.plotly_chart(px.bar(df, x="Tháng", y=cols, title="Cost Structure"), use_container_width=True)
        with c2:
            if "Lợi Nhuận ST" in df.columns:
                vals = [last.get(c, 0) for c in cols] + [last.get("Lợi Nhuận ST", 0)]
                fig2 = px.pie(values=vals, names=cols + ["Lợi Nhuận"], title="Tỷ trọng tháng cuối")
                st.plotly_chart(fig2, use_container_width=True)

    # === TAB 3: RISK (ML) ===
    with t3:
        if is_vip:
            st.header("Hệ thống Phát hiện Gian lận")
            if st.button("🔍 QUÉT RỦI RO"):
                bad = phat_hien_gian_lan_ml(df.copy())
                if not bad.empty:
                    st.error(f"⚠️ Phát hiện {len(bad)} tháng bất thường!")
                    st.dataframe(bad)
                else: st.success("✅ Số liệu ổn định.")
        else: st.warning("Restricted Area")

    # === TAB 4: DỰ BÁO (AI) ===
    with t4:
        if st.session_state.user_role == "admin":
            st.header("Dự báo Chiến lược")
            if "Lợi Nhuận ST" in df.columns:
                df['idx'] = range(len(df))
                reg = LinearRegression().fit(df[['idx']], df['Lợi Nhuận ST'])
                fut = np.array([[len(df)], [len(df)+1], [len(df)+2]])
                pred = reg.predict(fut)
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.write("Dự kiến 3 tháng tới:")
                    for i, v in enumerate(pred): st.metric(f"Tháng +{i+1}", f"{v:,.0f}")
                with c2:
                    fig = px.scatter(df, x="Tháng", y="Lợi Nhuận ST", trendline="ols", title="Xu hướng Lợi nhuận")
                    st.plotly_chart(fig, use_container_width=True)
        else: st.warning("⛔ Chỉ dành cho CFO.")

    # === TAB 5: PHÁP CHẾ (WEB + FILE) ===
    with t5:
        st.header("⚖️ Trung Tâm Pháp Chế")
        
        # 1. Nạp
        with st.expander("📥 Nạp Kiến thức (File/Web)", expanded=True):
            up_laws = st.file_uploader("Upload PDF/Docx", accept_multiple_files=True)
            url_law = st.text_input("Hoặc dán Link Web:")
            
            if st.button("Nạp Dữ liệu"):
                content = ""
                if up_laws:
                    for f in up_laws: content += doc_tai_lieu(f) + "\n"
                if url_law:
                    content += doc_url(url_law) + "\n"
                
                if content:
                    st.session_state.legal_data = content
                    st.success(f"Đã nạp {len(content)} ký tự.")
        
        # 2. Chat
        if 'legal_data' in st.session_state:
            q = st.chat_input("Hỏi về luật...")
            if q:
                st.chat_message("user").write(q)
                with st.chat_message("assistant"):
                    with st.spinner("Tra cứu..."):
                        ctx = st.session_state.legal_data[:30000]
                        res = model.generate_content(f"Context: {ctx}. Q: {q}. Role: Legal Expert.")
                        st.markdown(res.text)

# --- 6. MAIN ---
def main():
    auth = AuthManager()
    if 'is_logged_in' not in st.session_state: st.session_state.is_logged_in = False
    if 'lang_code' not in st.session_state: st.session_state.lang_code = 'vi'

    if not st.session_state.is_logged_in:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.title(T("login_title"))
            user = st.text_input(T("lbl_user"))
            password = st.text_input(T("lbl_pass"), type="password")
            if st.button(T("btn_login"), type="primary", use_container_width=True):
                if auth.verify(user, password):
                    st.session_state.is_logged_in = True
                    st.session_state.username = user
                    st.session_state.user_role = auth.get_role(user)
                    st.rerun()
                else: st.error(T("login_fail"))
    else:
        show_dashboard()

if __name__ == "__main__":
    main()
