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
import time

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="AI Financial Controller", layout="wide", page_icon="💰")

# --- TỪ ĐIỂN NGÔN NGỮ (GIỮ NGUYÊN) ---
TRANS = {
    "vi": {
        "title": "💰 AI Financial Controller",
        "login_title": "🔐 Cổng Đăng Nhập Nội Bộ",
        "lbl_user": "Tài khoản",
        "lbl_pass": "Mật khẩu",
        "btn_login": "Đăng Nhập",
        "err_login": "Sai tài khoản hoặc mật khẩu!",
        "welcome": "Xin chào",
        "role_admin": "Giám đốc Tài chính (CFO)",
        "role_staff": "Nhân viên Kế toán",
        "sidebar_lang": "Ngôn ngữ / Language",
        "sidebar_source": "Nguồn Dữ Liệu",
        "opt_demo": "🎲 Dữ liệu Demo",
        "opt_upload": "📂 Upload Excel",
        "btn_sample": "Tạo dữ liệu mẫu",
        "tab1": "📊 Dashboard",
        "tab2": "🕵️ Soi Rủi Ro (Admin Only)",
        "tab3": "🔮 Dự Báo (Admin Only)",
        "tab4": "💬 Chat Tài Chính",
        "restricted": "⛔ KHU VỰC HẠN CHẾ: Chỉ dành cho CFO.",
        "logout": "Đăng Xuất"
    },
    "en": {
        "title": "💰 AI Financial Controller",
        "login_title": "🔐 Internal Login Portal",
        "lbl_user": "Username",
        "lbl_pass": "Password",
        "btn_login": "Login",
        "err_login": "Invalid credentials!",
        "welcome": "Welcome",
        "role_admin": "CFO",
        "role_staff": "Accountant",
        "sidebar_lang": "Language",
        "sidebar_source": "Data Source",
        "opt_demo": "🎲 Demo Data",
        "opt_upload": "📂 Upload Excel",
        "btn_sample": "Generate Sample",
        "tab1": "📊 Dashboard",
        "tab2": "🕵️ Risk Audit (Admin)",
        "tab3": "🔮 Forecast (Admin)",
        "tab4": "💬 Chat Finance",
        "restricted": "⛔ RESTRICTED AREA: CFO Access Only.",
        "logout": "Logout"
    },
    "zh": {
        "title": "💰 AI 财务控制系统",
        "login_title": "🔐 内部登录门户",
        "lbl_user": "用户名",
        "lbl_pass": "密码",
        "btn_login": "登录",
        "err_login": "用户名或密码错误！",
        "welcome": "你好",
        "role_admin": "财务总监 (CFO)",
        "role_staff": "会计专员",
        "sidebar_lang": "语言",
        "sidebar_source": "数据源",
        "opt_demo": "🎲 模拟数据",
        "opt_upload": "📂 上传 Excel",
        "btn_sample": "生成样本",
        "tab1": "📊 财务概览",
        "tab2": "🕵️ 风险审计 (仅限管理员)",
        "tab3": "🔮 预测 (仅限管理员)",
        "tab4": "💬 财务问答",
        "restricted": "⛔ 限制区域：仅限财务总监访问。",
        "logout": "登出"
    }
}

def T(key):
    lang = st.session_state.get('lang_code', 'vi')
    return TRANS[lang].get(key, key)

# --- 2. HỆ THỐNG ĐĂNG NHẬP (AUTH MANAGER) ---
class AuthManager:
    def __init__(self):
        # Lấy danh sách user từ secrets
        self.users = st.secrets.get("users", {})
        self.roles = st.secrets.get("roles", {})

    def verify_login(self, username, password):
        if username in self.users and self.users[username] == password:
            return True
        return False

    def get_role(self, username):
        # Mặc định là staff nếu không có trong danh sách roles
        return self.roles.get(username, "staff")

# --- 3. CÁC HÀM XỬ LÝ (CORE) ---
try:
    if 'system' in st.secrets: api_key = st.secrets['system']['gemini_api_key']
    elif 'api_keys' in st.secrets: api_key = st.secrets['api_keys']['gemini_api_key']
    else: st.stop()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
except: pass # Bỏ qua lỗi nếu chưa login

def doc_tai_lieu(uploaded_file):
    try:
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext == 'pdf': return "\n".join([p.extract_text() for p in PdfReader(uploaded_file).pages])
        elif ext == 'docx': return "\n".join([p.text for p in Document(uploaded_file).paragraphs])
        elif ext in ['txt', 'md']: return str(uploaded_file.read(), "utf-8")
    except: return ""
    return ""

def phat_hien_bat_thuong(df):
    model_iso = IsolationForest(contamination=0.05, random_state=42)
    # Tìm cột nào có chữ "Chi" hoặc "Expense" hoặc cột số thứ 3
    col_target = df.columns[2] 
    df['Anomaly'] = model_iso.fit_predict(df[[col_target]])
    return df[df['Anomaly'] == -1]

def du_bao_tuong_lai(df):
    df['X'] = range(len(df))
    reg = LinearRegression().fit(df[['X']], df.iloc[:, 3]) # Cột Lợi nhuận
    future = np.array([[len(df)], [len(df)+1], [len(df)+2]])
    return reg.predict(future), reg.coef_[0]

def tao_data_mau():
    dates = pd.date_range(start="2023-01-01", periods=24, freq="ME")
    df = pd.DataFrame({
        "Tháng": dates,
        "Doanh Thu": np.random.randint(800, 1500, size=24) * 1000,
        "Chi Phí": np.random.randint(500, 1000, size=24) * 1000,
    })
    df["Lợi Nhuận"] = df["Doanh Thu"] - df["Chi Phí"]
    df.loc[10, "Chi Phí"] = 2000000 # Gài bẫy
    return df

# --- 4. GIAO DIỆN CHÍNH (SAU KHI LOGIN) ---
def show_app():
    # Sidebar cấu hình
    with st.sidebar:
        # Chọn Ngôn ngữ
        lang_map = {"Tiếng Việt": "vi", "English": "en", "中文": "zh"}
        sel_lang = st.selectbox("🌐 Language", list(lang_map.keys()))
        st.session_state.lang_code = lang_map[sel_lang]
        
        st.divider()
        
        # Thông tin User
        role_key = "role_admin" if st.session_state.user_role == "admin" else "role_staff"
        st.success(f"👤 {T('welcome')}, {st.session_state.username}")
        st.info(f"🔰 {T(role_key)}")
        
        if st.button(T("logout")):
            st.session_state.is_logged_in = False
            st.rerun()
            
        st.divider()
        
        # Chọn Nguồn Dữ liệu
        st.header(f"🗂️ {T('sidebar_source')}")
        source = st.radio("", [T("opt_demo"), T("opt_upload")])
        
        df = None
        if source == T("opt_demo"):
            if st.button(T("btn_sample")): st.session_state.df_fin = tao_data_mau()
        else:
            up_file = st.file_uploader("Excel", type=['xlsx'])
            if up_file: st.session_state.df_fin = pd.read_excel(up_file)

        if 'df_fin' in st.session_state:
            df = st.session_state.df_fin
            st.success(T("success_load").format(n=len(df)))

    st.title(T("title"))

    if df is not None:
        # PHÂN QUYỀN HIỂN THỊ TAB
        # Nếu là Admin: Thấy hết 4 tab. Nếu là Staff: Chỉ thấy Tab 1 và 4
        is_admin = st.session_state.user_role == "admin"
        
        if is_admin:
            tabs = st.tabs([T("tab1"), T("tab2"), T("tab3"), T("tab4")])
            t1, t2, t3, t4 = tabs[0], tabs[1], tabs[2], tabs[3]
        else:
            tabs = st.tabs([T("tab1"), T("tab4"), "🔒 Admin Zone", "🔒 Admin Zone"])
            t1, t4 = tabs[0], tabs[1]
            t2, t3 = tabs[2], tabs[3] # Tab bị khóa

        # --- NỘI DUNG TABS ---
        
        # TAB 1: DASHBOARD (Ai cũng xem được)
        with t1:
            rev, exp = df.iloc[:, 1].sum(), df.iloc[:, 2].sum()
            net = rev - exp
            c1, c2, c3 = st.columns(3)
            c1.metric("Revenue", f"{rev:,.0f}")
            c2.metric("Expense", f"{exp:,.0f}")
            c3.metric("Profit", f"{net:,.0f}")
            
            # Chỉ Admin mới có nút "Báo cáo tiếng Trung" (Ví dụ phân quyền sâu hơn)
            if is_admin:
                if st.button("🇨🇳 Generate Report (Admin Only)", type="primary"):
                    with st.spinner("AI thinking..."):
                        res = model.generate_content(f"Role: CFO. Data: {rev}, {exp}, {net}. Write report in Business Chinese.")
                        st.info(res.text)
            
            fig = px.bar(df, x=df.columns[0], y=[df.columns[1], df.columns[2]], barmode="group")
            st.plotly_chart(fig, use_container_width=True)

        # TAB 2: ML RISK (Chỉ Admin)
        with t2:
            if is_admin:
                st.header(T("risk_header"))
                if st.button(T("risk_btn")):
                    bad = phat_hien_bat_thuong(df.copy())
                    if not bad.empty:
                        st.error(T("risk_warn").format(n=len(bad)))
                        st.dataframe(bad.style.highlight_max(axis=0, color='pink'))
                        res = model.generate_content(f"Analyze risks: {bad.to_string()}. Lang: {st.session_state.lang_code}")
                        st.markdown(res.text)
                    else: st.success(T("risk_ok"))
            else:
                st.warning(T("restricted"))
                st.image("https://cdn-icons-png.flaticon.com/512/3064/3064197.png", width=100)

        # TAB 3: FORECAST (Chỉ Admin)
        with t3:
            if is_admin:
                st.header(T("forecast_header"))
                pred, trend = du_bao_tuong_lai(df)
                st.write(f"Trend: {'🚀 UP' if trend>0 else '📉 DOWN'}")
                fig2 = px.scatter(df, x=df.columns[0], y=df.columns[3], trendline="ols")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning(T("restricted"))

        # TAB 4: CHAT (Ai cũng dùng được)
        with t4:
            st.header(T("chat_header"))
            up_doc = st.file_uploader(T("chat_upload"), type=["pdf", "docx", "txt"])
            if up_doc:
                txt = doc_tai_lieu(up_doc)
                st.success(f"Loaded {len(txt)} chars.")
                if q := st.chat_input(T("chat_input")):
                    st.chat_message("user").write(q)
                    with st.chat_message("assistant"):
                        res = model.generate_content(f"Context: {txt[:30000]}. Q: {q}. Lang: {st.session_state.lang_code}. Role: CFO.")
                        st.markdown(res.text)
    else:
        st.info("👈 Please select Data Source.")

# --- 5. MÀN HÌNH LOGIN ---
def main():
    auth = AuthManager()
    
    # Khởi tạo session
    if 'is_logged_in' not in st.session_state: st.session_state.is_logged_in = False
    if 'lang_code' not in st.session_state: st.session_state.lang_code = 'vi'

    if not st.session_state.is_logged_in:
        # Giao diện Login đẹp
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.title(T("login_title"))
            st.markdown("---")
            user = st.text_input(T("lbl_user"))
            password = st.text_input(T("lbl_pass"), type="password")
            
            if st.button(T("btn_login"), use_container_width=True, type="primary"):
                if auth.verify_login(user, password):
                    st.session_state.is_logged_in = True
                    st.session_state.username = user
                    st.session_state.user_role = auth.get_role(user)
                    st.toast(f"Welcome {user}!", icon="🎉")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(T("err_login"))
            
            st.caption("Demo Accounts:")
            st.code("CFO: admin_cfo / mai_hanh_vip\nStaff: staff_01 / nv123")
    else:
        show_app()

if __name__ == "__main__":
    main()
