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
st.set_page_config(page_title="AI Financial Controller Pro", layout="wide", page_icon="📈")

# --- TỪ ĐIỂN ĐA NGÔN NGỮ ---
TRANS = {
    "vi": {
        "title": "💰 Hệ thống Phân tích Tài chính & Quản trị (CFO AI)",
        "role_admin": "CFO (Giám đốc Tài chính)",
        "role_chief": "Kế toán trưởng",
        "role_staff": "Kế toán viên",
        "tab1": "📊 Chỉ Số Tài Chính (KPIs)",
        "tab2": "📉 Phân Tích Chi Phí",
        "tab3": "🕵️ Soát Xét Rủi Ro (ML)",
        "tab4": "🔮 Chiến Lược & Dự Báo",
        "tab5": "📚 Thư Viện Luật & Chat",
        "kpi_select": "Chọn Nhóm Chỉ Số muốn xem:",
        "grp_liquid": "Khả năng Thanh toán",
        "grp_profit": "Khả năng Sinh lời",
        "grp_activity": "Hiệu quả Hoạt động",
        "btn_cn": "🇨🇳 Xuất Báo Cáo Sâu (Tiếng Trung)",
        "warn": "⚠️ Cảnh báo: {metric} đang ở mức rủi ro ({val})",
        "logout": "Đăng xuất"
    },
    "en": {
        "title": "💰 AI Financial Controller Pro",
        "role_admin": "CFO",
        "role_chief": "Chief Accountant",
        "role_staff": "Staff",
        "tab1": "📊 Financial KPIs",
        "tab2": "📉 Cost Analysis",
        "tab3": "🕵️ Risk Audit (ML)",
        "tab4": "🔮 Forecast Strategy",
        "tab5": "📚 Law & Chat",
        "kpi_select": "Select KPI Group:",
        "grp_liquid": "Liquidity",
        "grp_profit": "Profitability",
        "grp_activity": "Activity/Turnover",
        "btn_cn": "🇨🇳 Generate Deep Report (Chinese)",
        "warn": "⚠️ Warning: {metric} is risky ({val})",
        "logout": "Logout"
    },
    "zh": {
        "title": "💰 AI 财务分析与管理系统",
        "role_admin": "财务总监 (CFO)",
        "role_chief": "财务经理",
        "role_staff": "会计",
        "tab1": "📊 财务指标 (KPIs)",
        "tab2": "📉 成本分析",
        "tab3": "🕵️ 风险审计 (ML)",
        "tab4": "🔮 战略预测",
        "tab5": "📚 法律与问答",
        "kpi_select": "选择指标组:",
        "grp_liquid": "偿债能力",
        "grp_profit": "盈利能力",
        "grp_activity": "营运能力",
        "btn_cn": "🇨🇳 生成深度报告",
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

# --- 3. LOGIC TÀI CHÍNH (FIX LỖI KEY ERROR) ---
def tao_data_full_kpi():
    # Tạo dữ liệu chuẩn đầy đủ
    dates = pd.date_range(start="2024-01-01", periods=12, freq="ME")
    df = pd.DataFrame({"Tháng": dates})
    
    # Random dữ liệu
    df["Doanh Thu"] = np.random.randint(5000, 8000, 12) * 1000000
    df["Giá Vốn (Trực tiếp)"] = df["Doanh Thu"] * 0.6 
    df["Chi Phí VH (Gián tiếp)"] = np.random.randint(500, 800, 12) * 1000000
    df["Lợi Nhuận ST"] = df["Doanh Thu"] - df["Giá Vốn (Trực tiếp)"] - df["Chi Phí VH (Gián tiếp)"]
    
    df["TS Ngắn Hạn"] = np.random.randint(2000, 3000, 12) * 1000000
    df["Nợ Ngắn Hạn"] = np.random.randint(1000, 1500, 12) * 1000000
    df["Hàng Tồn Kho"] = np.random.randint(800, 1200, 12) * 1000000
    df["Phải Thu KH"] = np.random.randint(1000, 1500, 12) * 1000000
    df["Tổng Tài Sản"] = df["TS Ngắn Hạn"] + 5000000000 
    df["Vốn Chủ Sở Hữu"] = df["Tổng Tài Sản"] * 0.5
    
    # Gài bẫy
    df.loc[5, "Chi Phí VH (Gián tiếp)"] = 2500000000
    df.loc[9, "Chi Phí VH (Gián tiếp)"] = 2200000000
    return df

def tinh_chi_so_tai_chinh(df):
    """Hàm tính toán KPI - ĐÃ GIA CỐ CHỐNG LỖI"""
    
    # 1. Tự động điền các cột thiếu (nếu upload file excel cũ)
    required_cols = [
        "TS Ngắn Hạn", "Nợ Ngắn Hạn", "Hàng Tồn Kho", "Phải Thu KH", 
        "Tổng Tài Sản", "Vốn Chủ Sở Hữu", "Giá Vốn (Trực tiếp)", 
        "Doanh Thu", "Lợi Nhuận ST"
    ]
    
    # Nếu thiếu cột nào, tạo cột đó với giá trị giả định (để không bị sập app)
    for col in required_cols:
        if col not in df.columns:
            # Nếu thiếu, gán bằng 1 (để tránh chia cho 0) hoặc giá trị trung bình giả
            df[col] = 1000000000 
            
    # 2. Tính toán (Có bẫy lỗi chia cho 0)
    try:
        # Thanh toán
        df["Current Ratio"] = df["TS Ngắn Hạn"] / df["Nợ Ngắn Hạn"].replace(0, 1)
        
        # Hoạt động
        df["Inv Turnover"] = df["Giá Vốn (Trực tiếp)"] / df["Hàng Tồn Kho"].replace(0, 1)
        df["AR Turnover"] = df["Doanh Thu"] / df["Phải Thu KH"].replace(0, 1)
        df["Asset Turnover"] = df["Doanh Thu"] / df["Tổng Tài Sản"].replace(0, 1)

        # Sinh lời
        df["ROS"] = (df["Lợi Nhuận ST"] / df["Doanh Thu"].replace(0, 1)) * 100
        df["ROA"] = (df["Lợi Nhuận ST"] / df["Tổng Tài Sản"].replace(0, 1)) * 100
        df["ROE"] = (df["Lợi Nhuận ST"] / df["Vốn Chủ Sở Hữu"].replace(0, 1)) * 100
        
    except Exception as e:
        st.error(f"Lỗi tính toán chỉ số: {e}")
    
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
    # Tìm cột chi phí (ưu tiên Chi Phí VH, nếu không có lấy cột thứ 3)
    target_col = "Chi Phí VH (Gián tiếp)"
    if target_col not in df.columns:
        if len(df.columns) > 2: target_col = df.columns[2]
        else: return pd.DataFrame()
        
    df['Anomaly_Score'] = model_iso.fit_predict(df[[target_col]])
    return df[df['Anomaly_Score'] == -1]

# --- 4. GIAO DIỆN CHÍNH ---
def show_dashboard():
    with st.sidebar:
        lang_map = {"Tiếng Việt": "vi", "English": "en", "中文": "zh"}
        sel = st.selectbox("🌐 Language", list(lang_map.keys()))
        st.session_state.lang_code = lang_map[sel]
        
        st.divider()
        role = st.session_state.user_role
        role_name = "role_" + role if role in ["admin", "chief", "staff"] else "role_staff"
        st.success(f"👤 {st.session_state.username} | 🔰 {T(role_name)}")
        
        st.header("🗂️ Data Source")
        
        # Nút tạo dữ liệu MỚI
        if st.button("Tạo dữ liệu mẫu (Full KPIs)", type="primary"):
            st.session_state.df_fin = tao_data_full_kpi()
            st.rerun()
        
        up = st.file_uploader("Upload Excel", type=['xlsx'])
        if up: st.session_state.df_fin = pd.read_excel(up)

        if st.button(T("logout")):
            st.session_state.is_logged_in = False; st.rerun()

    st.title(T("title"))

    if 'df_fin' not in st.session_state:
        st.info("👈 Mời Giám đốc bấm nút 'Tạo dữ liệu mẫu (Full KPIs)' để khởi động hệ thống.")
        return

    # TÍNH TOÁN AN TOÀN
    df = tinh_chi_so_tai_chinh(st.session_state.df_fin.copy())
    last_month = df.iloc[-1]
    is_vip = role in ["admin", "chief"]
    
    t1, t2, t3, t4, t5 = st.tabs([T("tab1"), T("tab2"), T("tab3"), T("tab4"), T("tab5")])

    # === TAB 1: KPIs ===
    with t1:
        st.subheader("Phân tích Hoạt động Kinh tế & Tài chính")
        options = [T("grp_liquid"), T("grp_profit"), T("grp_activity")]
        selection = st.multiselect(T("kpi_select"), options, default=options)
        
        c1, c2, c3 = st.columns(3)
        if T("grp_liquid") in selection:
            c1.markdown(f"#### 💧 {T('grp_liquid')}")
            c1.metric("Current Ratio", f"{last_month['Current Ratio']:.2f}")
            
        if T("grp_profit") in selection:
            c2.markdown(f"#### 💰 {T('grp_profit')}")
            c2.metric("ROE (Vốn chủ)", f"{last_month['ROE']:.1f}%")
            
        if T("grp_activity") in selection:
            c3.markdown(f"#### 🏭 {T('grp_activity')}")
            c3.metric("Vòng quay Tồn kho", f"{last_month['Inv Turnover']:.2f}")

        if is_vip:
            st.markdown("---")
            if st.button(T("btn_cn"), type="primary"):
                with st.spinner("AI writing..."):
                    p = f"Role: CFO. Month: {last_month['Tháng']}. ROE: {last_month['ROE']:.1f}%. Inv Turnover: {last_month['Inv Turnover']:.2f}. Write a professional report in Business Chinese."
                    res = model.generate_content(p)
                    st.info(res.text)

    # === TAB 2: CHI PHÍ ===
    with t2:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Cơ cấu Chi phí")
            # Tìm cột phù hợp để vẽ
            cols_to_plot = [c for c in ["Giá Vốn (Trực tiếp)", "Chi Phí VH (Gián tiếp)"] if c in df.columns]
            if cols_to_plot:
                fig = px.bar(df, x="Tháng", y=cols_to_plot, title="Biến động Chi phí")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Tỷ trọng")
            if "Lợi Nhuận ST" in df.columns:
                values = [last_month[c] for c in cols_to_plot] + [last_month["Lợi Nhuận ST"]]
                names = cols_to_plot + ["Lợi Nhuận"]
                fig2 = px.pie(values=values, names=names, hole=0.4)
                st.plotly_chart(fig2, use_container_width=True)

    # === TAB 3: RISK ===
    with t3:
        if is_vip:
            st.header("Hệ thống Phát hiện Gian lận")
            if st.button("🔍 QUÉT RỦI RO"):
                bad_data = phat_hien_gian_lan_ml(df.copy())
                if not bad_data.empty:
                    st.error(f"⚠️ CẢNH BÁO: {len(bad_data)} tháng bất thường!")
                    st.dataframe(bad_data)
                else:
                    st.success("✅ Số liệu ổn định.")
        else: st.warning("⛔ Restricted Area")

    # === TAB 4: FORECAST ===
    with t4:
        if st.session_state.user_role == "admin":
            st.header("Dự báo Chiến lược")
            if "Lợi Nhuận ST" in df.columns:
                df['idx'] = range(len(df))
                reg = LinearRegression().fit(df[['idx']], df['Lợi Nhuận ST'])
                future_X = np.array([[len(df)], [len(df)+1], [len(df)+2]])
                pred = reg.predict(future_X)
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.write("Dự kiến 3 tháng tới:")
                    for i, v in enumerate(pred):
                        st.metric(f"Tháng +{i+1}", f"{v:,.0f}")
                with c2:
                    fig = px.scatter(df, x="Tháng", y="Lợi Nhuận ST", trendline="ols", title="Xu hướng")
                    st.plotly_chart(fig, use_container_width=True)
        else: st.warning("⛔ Chỉ dành cho CFO.")

    # === TAB 5: LUẬT ===
    with t5:
        st.header("Trợ lý Pháp chế")
        up_law = st.file_uploader("Upload Luật (PDF)", type=["pdf"])
        if up_law:
            txt = doc_tai_lieu(up_law)
            q = st.chat_input("Hỏi gì đó...")
            if q:
                res = model.generate_content(f"Context: {txt[:30000]}. Q: {q}")
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
