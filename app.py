import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from pypdf import PdfReader
from docx import Document
import io

# --- 1. CẤU HÌNH ---
st.set_page_config(page_title="AI Financial Controller", layout="wide", page_icon="💰")

# --- TỪ ĐIỂN ĐA NGÔN NGỮ (ĐÃ SỬA LẠI PHẦN TIẾNG TRUNG ĐẦY ĐỦ) ---
TRANS = {
    "vi": {
        "title": "💰 Hệ thống Giám đốc Tài chính AI",
        "role_admin": "CFO (Giám đốc Tài chính)",
        "role_chief": "Kế toán trưởng",
        "role_staff": "Kế toán viên",
        "tab1": "📊 Bộ Chỉ Số KPIs",
        "tab2": "📉 Phân Tích Chi Phí",
        "tab3": "🕵️ Soát Xét Rủi Ro (ML)",
        "tab4": "🔮 Chiến Lược & Dự Báo",
        "tab5": "📚 Thư Viện Luật & Chat",
        "kpi_select": "Chọn Nhóm Chỉ Số muốn xem:",
        "grp_liquid": "1. Khả năng Thanh toán",
        "grp_profit": "2. Khả năng Sinh lời",
        "grp_activity": "3. Hiệu quả Hoạt động",
        "grp_struct": "4. Cấu trúc Vốn",
        "btn_cn": "🇨🇳 Báo Cáo Sếp (Tiếng Trung)",
        "warn": "⚠️ Cảnh báo: {metric} đang ở mức rủi ro ({val})",
        "logout": "Đăng xuất"
    },
    "en": {
        "title": "💰 AI Financial Controller",
        "role_admin": "CFO",
        "role_chief": "Chief Accountant",
        "role_staff": "Staff",
        "tab1": "📊 Financial KPIs",
        "tab2": "📉 Cost Analysis",
        "tab3": "🕵️ Risk Audit (ML)",
        "tab4": "🔮 Forecast Strategy",
        "tab5": "📚 Law & Chat",
        "kpi_select": "Select KPI Group:",
        "grp_liquid": "1. Liquidity Ratios",
        "grp_profit": "2. Profitability Ratios",
        "grp_activity": "3. Activity Ratios",
        "grp_struct": "4. Capital Structure",
        "btn_cn": "🇨🇳 Generate Chinese Report",
        "warn": "⚠️ Warning: {metric} is risky ({val})",
        "logout": "Logout"
    },
    "zh": {
        "title": "💰 AI 财务总监控制系统 (CFO System)",
        "role_admin": "财务总监 (CFO)",
        "role_chief": "财务经理",
        "role_staff": "会计",
        "tab1": "📊 财务指标 (KPIs)",
        "tab2": "📉 成本分析",
        "tab3": "🕵️ 风险审计 (ML)",
        "tab4": "🔮 战略预测",
        "tab5": "📚 法律与问答",
        "kpi_select": "选择财务指标组:",
        "grp_liquid": "1. 偿债能力 (Liquidity)",
        "grp_profit": "2. 盈利能力 (Profitability)",
        "grp_activity": "3. 营运能力 (Activity)",
        "grp_struct": "4. 资本结构 (Structure)",
        "btn_cn": "🇨🇳 生成深度财务报告",
        "warn": "⚠️ 警告：{metric} 处于风险水平 ({val})",
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

# --- 3. DATA GENERATOR (ĐẦY ĐỦ CỘT) ---
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
    # Đảm bảo tiền mặt không âm
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
    """Tính toán Full Option"""
    # Điền giá trị mặc định nếu thiếu cột (Safety)
    required = ["TS Ngắn Hạn", "Nợ Ngắn Hạn", "Hàng Tồn Kho", "Doanh Thu", "Tổng Tài Sản", "Vốn Chủ Sở Hữu"]
    for c in required:
        if c not in df.columns: df[c] = 1 # Tránh chia cho 0
            
    try:
        # 1. Thanh khoản
        df["Current Ratio"] = df["TS Ngắn Hạn"] / df["Nợ Ngắn Hạn"].replace(0, 1)
        # Giả sử Quick Ratio
        df["Quick Ratio"] = (df["TS Ngắn Hạn"] - df.get("Hàng Tồn Kho", 0)) / df["Nợ Ngắn Hạn"].replace(0, 1)
        
        # 2. Hoạt động
        df["Inv Turnover"] = df.get("Giá Vốn (Trực tiếp)", 0) / df.get("Hàng Tồn Kho", 1).replace(0, 1)
        df["Asset Turnover"] = df["Doanh Thu"] / df["Tổng Tài Sản"].replace(0, 1)
        
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

# Cấu hình Gemini
try:
    if 'system' in st.secrets: api_key = st.secrets['system']['gemini_api_key']
    elif 'api_keys' in st.secrets: api_key = st.secrets['api_keys']['gemini_api_key']
    else: st.stop()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
except: pass

# --- CÁC HÀM PHỤ TRỢ ---
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

# --- 4. GIAO DIỆN CHÍNH ---
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

    df = tinh_chi_so_tai_chinh(st.session_state.df_fin.copy())
    last = df.iloc[-1]
    is_vip = role in ["admin", "chief"]
    
    t1, t2, t3, t4, t5 = st.tabs([T("tab1"), T("tab2"), T("tab3"), T("tab4"), T("tab5")])

    # === TAB 1: DASHBOARD FULL ===
    with t1:
        st.subheader("Phân tích Hoạt động Kinh tế & Tài chính")
        
        # Multiselect để chọn nhóm chỉ số
        opts = [T("grp_liquid"), T("grp_profit"), T("grp_activity"), T("grp_struct")]
        sels = st.multiselect(T("kpi_select"), opts, default=opts)
        
        if T("grp_liquid") in sels:
            st.markdown(f"#### 💧 {T('grp_liquid')}")
            c1, c2 = st.columns(2)
            c1.metric("Current Ratio", f"{last.get('Current Ratio', 0):.2f}")
            c2.metric("Quick Ratio", f"{last.get('Quick Ratio', 0):.2f}")
            st.divider()

        if T("grp_profit") in sels:
            st.markdown(f"#### 💰 {T('grp_profit')}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Gross Margin", f"{last.get('Gross Margin', 0):.1f}%")
            c2.metric("ROS", f"{last.get('ROS', 0):.1f}%")
            c3.metric("ROA", f"{last.get('ROA', 0):.1f}%")
            c4.metric("ROE", f"{last.get('ROE', 0):.1f}%")
            st.divider()

        if T("grp_activity") in sels:
            st.markdown(f"#### 🏭 {T('grp_activity')}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Inv Turnover", f"{last.get('Inv Turnover', 0):.2f}")
            c2.metric("Asset Turnover", f"{last.get('Asset Turnover', 0):.2f}")
            c3.metric("AR Turnover", f"{last.get('AR Turnover', 0):.2f}") # Phải thu
            st.divider()

        if T("grp_struct") in sels:
            st.markdown(f"#### ⚖️ {T('grp_struct')}")
            c1, c2 = st.columns(2)
            c1.metric("Debt/Asset", f"{last.get('Debt/Asset', 0):.1f}%")
            c2.metric("Debt/Equity", f"{last.get('Debt/Equity', 0):.1f}%")

        if is_vip:
            st.markdown("---")
            if st.button(T("btn_cn"), type="primary"):
                with st.spinner("AI writing..."):
                    p = f"""
                    Role: CFO. Data Month: {last['Tháng']}.
                    Liquidity: Current {last.get('Current Ratio',0):.2f}.
                    Profit: ROE {last.get('ROE',0):.1f}%, ROS {last.get('ROS',0):.1f}%.
                    Activity: Inv Turn {last.get('Inv Turnover',0):.2f}.
                    Structure: D/E {last.get('Debt/Equity',0):.1f}%.
                    Task: Write a deep financial analysis report in Business Chinese.
                    """
                    res = model.generate_content(p)
                    st.info(res.text)

    # === TAB 2: CHI PHÍ ===
    with t2:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Cost Structure")
            cols = [c for c in ["Giá Vốn (Trực tiếp)", "Chi Phí VH (Gián tiếp)"] if c in df.columns]
            if cols:
                fig = px.bar(df, x="Tháng", y=cols, title="Cost Breakdown")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Ratio")
            if "Lợi Nhuận ST" in df.columns:
                vals = [last.get(c, 0) for c in cols] + [last.get("Lợi Nhuận ST", 0)]
                names = cols + ["Net Profit"]
                fig2 = px.pie(values=vals, names=names, hole=0.4)
                st.plotly_chart(fig2, use_container_width=True)

    # === TAB 3: RISK ===
    with t3:
        if is_vip:
            st.header("Anomaly Detection")
            if st.button("SCAN RISKS"):
                bad = phat_hien_gian_lan_ml(df.copy())
                if not bad.empty:
                    st.error(f"Found {len(bad)} anomalies!")
                    st.dataframe(bad)
                else: st.success("Data is clean.")
        else: st.warning("Restricted")

    # === TAB 4: FORECAST ===
    with t4:
        if st.session_state.user_role == "admin":
            st.header("Strategic Forecast")
            if "Lợi Nhuận ST" in df.columns:
                df['idx'] = range(len(df))
                reg = LinearRegression().fit(df[['idx']], df['Lợi Nhuận ST'])
                fut = np.array([[len(df)], [len(df)+1], [len(df)+2]])
                pred = reg.predict(fut)
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.write("Next 3 Months:")
                    for i, v in enumerate(pred): st.metric(f"M+{i+1}", f"{v:,.0f}")
                with c2:
                    fig = px.scatter(df, x="Tháng", y="Lợi Nhuận ST", trendline="ols")
                    st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Restricted")

    # === TAB 5: LEGAL ===
    with t5:
        st.header("Legal Assistant")
        up = st.file_uploader("Upload Law Doc", type=["pdf"])
        if up:
            txt = doc_tai_lieu(up)
            q = st.chat_input("Ask...")
            if q:
                res = model.generate_content(f"Context: {txt[:20000]}. Q: {q}")
                st.write(res.text)

# --- 5. MAIN ---
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
