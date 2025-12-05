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
import re

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="AI Financial Controller Pro", layout="wide", page_icon="⚖️")

# --- TỪ ĐIỂN ĐA NGÔN NGỮ (CẬP NHẬT MỚI) ---
TRANS = {
    "vi": {
        "title": "💰 AI Financial Controller (Hệ thống Kiểm soát Tài chính)",
        "login_title": "🔐 Cổng Đăng Nhập Nội Bộ",
        "welcome": "Xin chào",
        "role_admin": "Giám đốc Tài chính (CFO)",
        "role_chief": "Kế toán trưởng (Chief Acc)", # MỚI
        "role_staff": "Nhân viên Kế toán",
        "tab1": "📊 Dashboard",
        "tab2": "🕵️ Soi Rủi Ro (Chief/CFO)",
        "tab3": "🔮 Dự Báo (CFO Only)",
        "tab4": "💬 Chat Dữ Liệu",
        "tab5": "📚 Thư Viện Luật & Thuế", # MỚI
        "legal_warn": "🚨 CẢNH BÁO PHÁP LÝ",
        "legal_status": "Trạng thái văn bản:",
        "legal_expired": "ĐÃ HẾT HIỆU LỰC",
        "legal_valid": "Đang có hiệu lực",
        "btn_check_law": "Kiểm tra hiệu lực & Hỏi AI",
        "restricted": "⛔ KHU VỰC HẠN CHẾ: Bạn không có quyền truy cập.",
        "logout": "Đăng Xuất"
    },
    "en": {
        "title": "💰 AI Financial Controller",
        "login_title": "🔐 Internal Login Portal",
        "welcome": "Welcome",
        "role_admin": "CFO",
        "role_chief": "Chief Accountant",
        "role_staff": "Accountant",
        "tab1": "📊 Dashboard",
        "tab2": "🕵️ Risk Audit (Chief/CFO)",
        "tab3": "🔮 Forecast (CFO Only)",
        "tab4": "💬 Chat Data",
        "tab5": "📚 Legal & Tax Library",
        "legal_warn": "🚨 LEGAL WARNING",
        "legal_status": "Document Status:",
        "legal_expired": "EXPIRED",
        "legal_valid": "Valid",
        "btn_check_law": "Check Validity & Ask AI",
        "restricted": "⛔ RESTRICTED AREA.",
        "logout": "Logout"
    },
    "zh": {
        "title": "💰 AI 财务控制系统",
        "login_title": "🔐 内部登录",
        "welcome": "你好",
        "role_admin": "财务总监 (CFO)",
        "role_chief": "财务经理 (Chief Acc)",
        "role_staff": "会计",
        "tab1": "📊 概览",
        "tab2": "🕵️ 风险审计 (主管)",
        "tab3": "🔮 预测 (CFO)",
        "tab4": "💬 数据问答",
        "tab5": "📚 法律税务库",
        "legal_warn": "🚨 法律警告",
        "legal_status": "文件状态:",
        "legal_expired": "已失效",
        "legal_valid": "有效",
        "btn_check_law": "检查有效性 & 提问",
        "restricted": "⛔ 限制区域",
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

    def verify_login(self, username, password):
        if username in self.users and self.users[username] == password:
            return True
        return False

    def get_role(self, username):
        return self.roles.get(username, "staff")

# --- 3. CORE FUNCTIONS ---
try:
    if 'system' in st.secrets: api_key = st.secrets['system']['gemini_api_key']
    elif 'api_keys' in st.secrets: api_key = st.secrets['api_keys']['gemini_api_key']
    else: st.stop()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
except: pass

def doc_tai_lieu(uploaded_file):
    try:
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext == 'pdf': return "\n".join([p.extract_text() for p in PdfReader(uploaded_file).pages])
        elif ext == 'docx': return "\n".join([p.text for p in Document(uploaded_file).paragraphs])
        elif ext in ['txt', 'md']: return str(uploaded_file.read(), "utf-8")
    except: return ""
    return ""

# [NÂNG CẤP] ML với Data Cleaning
def phat_hien_bat_thuong(df):
    # 1. Clean Data: Xóa dòng trống, ép kiểu số
    df_clean = df.copy()
    col_target = df.columns[2] # Giả định cột 3 là Chi phí
    df_clean[col_target] = pd.to_numeric(df_clean[col_target], errors='coerce')
    df_clean = df_clean.dropna(subset=[col_target])
    
    # 2. Run Isolation Forest
    model_iso = IsolationForest(contamination=0.05, random_state=42)
    df_clean['Anomaly'] = model_iso.fit_predict(df_clean[[col_target]])
    
    return df_clean[df_clean['Anomaly'] == -1]

def du_bao_tuong_lai(df):
    df['X'] = range(len(df))
    reg = LinearRegression().fit(df[['X']], df.iloc[:, 3]) # Cột Lợi nhuận
    future = np.array([[len(df)], [len(df)+1], [len(df)+2]])
    return reg.predict(future), reg.coef_[0]

# [MỚI] TẠO DATABASE LUẬT GIẢ LẬP (Để demo tính năng cảnh báo)
def tao_db_luat_mau():
    return pd.DataFrame({
        "Ten_Van_Ban": ["Luật Kế toán 2003", "Thông tư 200/2014/TT-BTC", "Nghị định 51/2010/NĐ-CP", "Nghị định 123/2020/NĐ-CP"],
        "Trang_Thai": ["Hết hiệu lực", "Hiệu lực", "Hết hiệu lực", "Hiệu lực"],
        "Thay_The_Boi": ["Luật Kế toán 2015", "-", "Nghị định 123/2020/NĐ-CP", "-"]
    })

def kiem_tra_hieu_luc_van_ban(text_ai_tra_loi, df_luat):
    """Quét câu trả lời của AI xem có nhắc đến văn bản hết hiệu lực không"""
    canh_bao = []
    for index, row in df_luat.iterrows():
        # Nếu văn bản hết hiệu lực và tên văn bản xuất hiện trong câu trả lời AI
        if row['Trang_Thai'] == "Hết hiệu lực" and row['Ten_Van_Ban'] in text_ai_tra_loi:
            msg = f"⚠️ {row['Ten_Van_Ban']} đã HẾT HIỆU LỰC! Hãy dùng: {row['Thay_The_Boi']}."
            canh_bao.append(msg)
    return canh_bao

# --- 4. GIAO DIỆN CHÍNH ---
def show_app():
    # Setup Session
    if 'df_luat' not in st.session_state: st.session_state.df_luat = tao_db_luat_mau()

    with st.sidebar:
        # Lang & User Info
        lang_map = {"Tiếng Việt": "vi", "English": "en", "中文": "zh"}
        sel_lang = st.selectbox("🌐 " + T("sidebar_lang"), list(lang_map.keys()))
        st.session_state.lang_code = lang_map[sel_lang]
        
        st.divider()
        role_key = f"role_{st.session_state.user_role}" # role_admin, role_chief, role_staff
        st.success(f"👤 {st.session_state.username}")
        st.info(f"🔰 {T(role_key)}") # Hiện chức danh
        
        if st.button(T("logout")):
            st.session_state.is_logged_in = False
            st.rerun()
            
        st.divider()
        # Data Source (Giản lược để tập trung tính năng)
        st.header(f"🗂️ {T('sidebar_source')}")
        up_file = st.file_uploader("Upload Excel Báo Cáo", type=['xlsx'])
        if up_file: 
            st.session_state.df_fin = pd.read_excel(up_file)
            st.success("Data Loaded!")
        elif st.button(T("btn_sample")):
            # Tạo data mẫu nhanh
            dates = pd.date_range(start="2024-01-01", periods=12, freq="ME")
            st.session_state.df_fin = pd.DataFrame({
                "Month": dates, "Rev": np.random.randint(100,200,12)*10, "Exp": np.random.randint(50,150,12)*10
            })
            st.session_state.df_fin["Profit"] = st.session_state.df_fin["Rev"] - st.session_state.df_fin["Exp"]
            st.session_state.df_fin.iloc[5, 2] = 200000 # Gài lỗi

    st.title(T("title"))

    # PHÂN QUYỀN TABS
    role = st.session_state.user_role
    
    # Logic quyền:
    # Admin (CFO): Full quyền
    # Chief (Kế toán trưởng): Tab 1, 2, 4, 5 (Không xem Dự báo chiến lược Tab 3)
    # Staff: Tab 1, 4 (Chỉ xem và chat)
    
    is_admin = role == "admin"
    is_chief = role == "chief" or is_admin
    
    t1, t2, t3, t4, t5 = st.tabs([T("tab1"), T("tab2"), T("tab3"), T("tab4"), T("tab5")])

    # TAB 1: DASHBOARD (Public)
    with t1:
        if 'df_fin' in st.session_state:
            df = st.session_state.df_fin
            c1, c2, c3 = st.columns(3)
            c1.metric(T("metric_rev"), f"{df.iloc[:,1].sum():,.0f}")
            c2.metric(T("metric_exp"), f"{df.iloc[:,2].sum():,.0f}")
            c3.metric(T("metric_net"), f"{df.iloc[:,3].sum():,.0f}")
            st.plotly_chart(px.bar(df, x=df.columns[0], y=[df.columns[1], df.columns[2]], barmode="group"), use_container_width=True)
        else: st.info("👈 Upload Excel data")

    # TAB 2: ML RISK (Chief + Admin)
    with t2:
        if is_chief:
            st.header(T("risk_header"))
            if 'df_fin' in st.session_state and st.button(T("risk_btn")):
                bad = phat_hien_bat_thuong(st.session_state.df_fin)
                if not bad.empty:
                    st.error(T("risk_warn").format(n=len(bad)))
                    st.dataframe(bad.style.highlight_max(axis=0, color='pink'))
                else: st.success(T("risk_ok"))
        else: st.warning(T("restricted"))

    # TAB 3: FORECAST (Admin Only)
    with t3:
        if is_admin:
            st.header("🔮 Forecasting Strategy")
            if 'df_fin' in st.session_state:
                pred, trend = du_bao_tuong_lai(st.session_state.df_fin)
                st.plotly_chart(px.scatter(st.session_state.df_fin, x=st.session_state.df_fin.columns[0], y=st.session_state.df_fin.columns[3], trendline="ols"), use_container_width=True)
        else: st.warning(T("restricted"))

    # TAB 4: CHAT DATA (Public)
    with t4:
        st.header("💬 Chat")
        q = st.chat_input(T("chat_input"))
        if q:
            st.chat_message("user").write(q)
            with st.spinner("AI thinking..."):
                res = model.generate_content(f"Answer as accountant: {q}")
                st.chat_message("assistant").write(res.text)

    # TAB 5: THƯ VIỆN LUẬT (Chief + Admin) - TÍNH NĂNG MỚI
    with t5:
        if is_chief:
            st.header("📚 Legal & Tax Knowledge Base")
            
            # Phần 1: Quản lý danh sách hiệu lực
            with st.expander("📋 Danh sách Hiệu lực Văn bản (Editable)", expanded=True):
                # Cho phép edit trực tiếp trên bảng (Data Editor)
                edited_df = st.data_editor(st.session_state.df_luat, num_rows="dynamic")
                st.session_state.df_luat = edited_df # Lưu lại thay đổi
            
            # Phần 2: Hỏi đáp Luật & Cảnh báo
            st.divider()
            st.subheader("🤖 Trợ lý Pháp chế (Có cảnh báo hiệu lực)")
            
            # Upload văn bản luật mới
            law_file = st.file_uploader("Upload Văn bản Luật (PDF/Docx) để hỏi", type=["pdf", "docx"])
            law_context = ""
            if law_file: 
                law_context = doc_tai_lieu(law_file)
                st.caption(f"Đã đọc: {law_file.name}")

            q_law = st.text_input("Câu hỏi về Luật/Thuế:", placeholder="Ví dụ: Nghị định 51 còn dùng được không?")
            
            if st.button(T("btn_check_law")):
                with st.spinner("Đang tra cứu và kiểm tra hiệu lực..."):
                    # 1. AI Trả lời
                    prompt = f"""
                    Bạn là Chuyên gia Tư vấn Thuế và Luật Kế toán.
                    Ngữ cảnh văn bản (nếu có): {law_context[:10000]}
                    Câu hỏi: "{q_law}"
                    Trả lời chi tiết, trích dẫn văn bản pháp luật nếu biết.
                    """
                    res = model.generate_content(prompt)
                    
                    # 2. Logic kiểm tra hiệu lực (Cảnh báo đỏ)
                    alerts = kiem_tra_hieu_luc_van_ban(res.text, st.session_state.df_luat)
                    
                    # 3. Hiển thị
                    if alerts:
                        for alert in alerts:
                            st.error(alert) # Hiện cảnh báo đỏ chót
                    else:
                        st.success("✅ Các văn bản được nhắc đến đều đang có hiệu lực (hoặc không nằm trong danh sách theo dõi).")
                        
                    st.markdown("### 💡 Câu trả lời của AI:")
                    st.markdown(res.text)
        else:
            st.warning(T("restricted"))

# --- 5. MAIN LOGIN ---
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
                if auth.verify_login(user, password):
                    st.session_state.is_logged_in = True
                    st.session_state.username = user
                    st.session_state.user_role = auth.get_role(user)
                    st.rerun()
                else: st.error(T("err_login"))
            st.caption("Demo: admin_cfo (CFO) | chief_acc (KTT) | staff_01 (NV)")
    else:
        show_app()

if __name__ == "__main__":
    main()
