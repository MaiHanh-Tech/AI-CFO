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

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="AI Financial Controller Pro", layout="wide", page_icon="📈")

# --- TỪ ĐIỂN ĐA NGÔN NGỮ ---
TRANS = {
    "vi": {
        "title": "💰 Hệ thống Phân tích Tài chính & Quản trị (CFO AI)",
        "role_admin": "CFO (Giám đốc Tài chính)",
        "role_chief": "Kế toán trưởng",
        "role_staff": "Kế toán viên",
        "tab1": "📊 Chỉ Số Tài Chính (KPIs)",
        "tab2": "📉 Phân Tích Hoạt Động",
        "tab3": "🔮 Dự Báo Chiến Lược",
        "tab4": "💬 Trợ Lý Số Liệu",
        "group_liquid": "1. Khả năng Thanh toán",
        "group_profit": "2. Khả năng Sinh lời",
        "group_active": "3. Hiệu quả Hoạt động",
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
        "tab2": "📉 Activity Analysis",
        "tab3": "🔮 Forecast Strategy",
        "tab4": "💬 Data Assistant",
        "group_liquid": "1. Liquidity Ratios",
        "group_profit": "2. Profitability Ratios",
        "group_active": "3. Activity Ratios",
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
        "tab2": "📉 经营分析",
        "tab3": "🔮 战略预测",
        "tab4": "💬 数据助手",
        "group_liquid": "1. 偿债能力",
        "group_profit": "2. 盈利能力",
        "group_active": "3. 营运能力",
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

# --- 3. LOGIC TÀI CHÍNH CHUYÊN SÂU (CORE) ---
def tao_data_chuyen_sau():
    # Giả lập dữ liệu đầy đủ cho Bảng CĐKT và KQKD
    dates = pd.date_range(start="2024-01-01", periods=12, freq="ME")
    df = pd.DataFrame({"Tháng": dates})
    
    # KQKD
    df["Doanh Thu"] = np.random.randint(2000, 3000, 12) * 1000000
    df["Giá Vốn"] = df["Doanh Thu"] * 0.6
    df["Lợi Nhuận Sau Thuế"] = df["Doanh Thu"] * 0.15
    
    # CĐKT (Bình quân)
    df["Tài Sản Ngắn Hạn"] = np.random.randint(1000, 1500, 12) * 1000000
    df["Nợ Ngắn Hạn"] = np.random.randint(500, 800, 12) * 1000000
    df["Tổng Tài Sản"] = np.random.randint(5000, 6000, 12) * 1000000
    df["Vốn Chủ Sở Hữu"] = df["Tổng Tài Sản"] * 0.6
    
    df["Hàng Tồn Kho"] = np.random.randint(300, 500, 12) * 1000000
    df["Phải Thu Khách Hàng"] = np.random.randint(400, 600, 12) * 1000000
    
    return df

def tinh_chi_so_tai_chinh(df):
    """Hàm tính toán bộ chỉ số KPI"""
    # 1. Thanh toán
    df["Current Ratio"] = df["Tài Sản Ngắn Hạn"] / df["Nợ Ngắn Hạn"] # Thanh toán hiện hành
    
    # 2. Hoạt động (Vòng quay - tính theo năm giả định x12 cho tháng)
    # Vòng quay tồn kho = Giá vốn / Tồn kho bq
    df["Inv Turnover"] = df["Giá Vốn"] / df["Hàng Tồn Kho"] 
    # Vòng quay phải thu = Doanh thu / Phải thu bq
    df["AR Turnover"] = df["Doanh Thu"] / df["Phải Thu Khách Hàng"]
    # Vòng quay tài sản = Doanh thu / Tổng tài sản
    df["Asset Turnover"] = df["Doanh Thu"] / df["Tổng Tài Sản"]

    # 3. Sinh lời
    df["ROS"] = (df["Lợi Nhuận Sau Thuế"] / df["Doanh Thu"]) * 100
    df["ROA"] = (df["Lợi Nhuận Sau Thuế"] / df["Tổng Tài Sản"]) * 100
    df["ROE"] = (df["Lợi Nhuận Sau Thuế"] / df["Vốn Chủ Sở Hữu"]) * 100
    
    return df

# Cấu hình Gemini
try:
    if 'system' in st.secrets: api_key = st.secrets['system']['gemini_api_key']
    elif 'api_keys' in st.secrets: api_key = st.secrets['api_keys']['gemini_api_key']
    else: st.stop()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
except: pass

# --- 4. GIAO DIỆN ---
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
        if st.button("Tạo dữ liệu mẫu (Full KPIs)"):
            st.session_state.df_fin = tao_data_chuyen_sau()
            st.rerun()
        
        up = st.file_uploader("Upload Excel (Đủ cột CĐKT & KQKD)", type=['xlsx'])
        if up: st.session_state.df_fin = pd.read_excel(up)

        if st.button(T("logout")):
            st.session_state.is_logged_in = False; st.rerun()

    st.title(T("title"))

    if 'df_fin' not in st.session_state:
        st.info("👈 Vui lòng tạo dữ liệu mẫu để xem các chỉ số chuyên sâu.")
        return

    # Tính toán chỉ số trước khi hiển thị
    df = tinh_chi_so_tai_chinh(st.session_state.df_fin)
    latest = df.iloc[-1] # Lấy tháng gần nhất
    
    is_vip = role in ["admin", "chief"] 
    
    t1, t2, t3, t4 = st.tabs([T("tab1"), T("tab2"), T("tab3"), T("tab4")])

    # === TAB 1: BỘ CHỈ SỐ TÀI CHÍNH (KPIs) ===
    with t1:
        st.subheader(f"Báo cáo Tháng {latest['Tháng'].strftime('%m/%Y')}")
        
        # Nhóm 1: Thanh toán
        st.markdown(f"#### 💧 {T('group_liquid')}")
        k1, k2, k3 = st.columns(3)
        k1.metric("Current Ratio", f"{latest['Current Ratio']:.2f}", help="Tài sản NH / Nợ NH (Tốt: 2-3)")
        # Giả lập Quick Ratio (Tài sản nhanh / Nợ)
        quick_r = (latest['Tài Sản Ngắn Hạn'] - latest['Hàng Tồn Kho']) / latest['Nợ Ngắn Hạn']
        k2.metric("Quick Ratio", f"{quick_r:.2f}", help="Thanh toán nhanh")
        
        # Nhóm 2: Sinh lời
        st.markdown(f"#### 💰 {T('group_profit')}")
        p1, p2, p3 = st.columns(3)
        p1.metric("ROS (Net Margin)", f"{latest['ROS']:.1f}%")
        p2.metric("ROA (Trên Tài sản)", f"{latest['ROA']:.1f}%")
        p3.metric("ROE (Trên Vốn chủ)", f"{latest['ROE']:.1f}%", help="Lợi nhuận / Vốn chủ sở hữu")

        # NÚT BÁO CÁO TIẾNG TRUNG (VIP)
        if is_vip:
            st.markdown("---")
            if st.button(T("btn_cn"), type="primary"):
                with st.spinner("AI đang phân tích các chỉ số..."):
                    p = f"""
                    Role: Chief Accountant. 
                    Data Month: {latest['Tháng']}.
                    Liquidity: Current Ratio {latest['Current Ratio']:.2f}.
                    Profitability: ROE {latest['ROE']:.1f}%, ROS {latest['ROS']:.1f}%.
                    Activity: Inventory Turnover {latest['Inv Turnover']:.2f}.
                    
                    Task: Write a deep financial analysis in Business Chinese. 
                    Focus on: Efficiency and Risk.
                    """
                    res = model.generate_content(p)
                    st.info(res.text)

    # === TAB 2: PHÂN TÍCH HOẠT ĐỘNG (ACTIVITY) ===
    with t2:
        if is_vip:
            st.markdown(f"#### 🏭 {T('group_active')}")
            
            c1, c2 = st.columns(2)
            with c1:
                # Biểu đồ Vòng quay tồn kho
                fig = px.line(df, x="Tháng", y="Inv Turnover", markers=True, title="Vòng quay Hàng Tồn Kho (Lần)")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Cao là tốt: Hàng bán nhanh. Thấp: Ứ đọng vốn.")
                
            with c2:
                # Biểu đồ Vòng quay phải thu
                fig2 = px.line(df, x="Tháng", y="AR Turnover", markers=True, title="Vòng quay Khoản Phải Thu (Lần)")
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("Cao là tốt: Thu hồi nợ nhanh.")
            
            # AI Nhận xét hoạt động
            if st.button("🤖 AI Nhận xét Hiệu quả Hoạt động"):
                data_str = df[['Tháng', 'Inv Turnover', 'AR Turnover']].tail(3).to_string()
                res = model.generate_content(f"Phân tích xu hướng hiệu quả hoạt động dựa trên data này: {data_str}. Ngôn ngữ: {st.session_state.lang_code}")
                st.markdown(res.text)
        else:
            st.warning("⛔ Access Denied")

    # === TAB 3: DỰ BÁO ===
    with t3:
        if st.session_state.user_role == "admin":
            st.header("Dự báo Lợi nhuận (Linear Regression)")
            df['idx'] = range(len(df))
            reg = LinearRegression().fit(df[['idx']], df['Lợi Nhuận Sau Thuế'])
            
            next_months = pd.date_range(start=df["Tháng"].iloc[-1], periods=4, freq="ME")[1:]
            pred = reg.predict(np.array([[len(df)], [len(df)+1], [len(df)+2]]))
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.write("Dự kiến 3 tháng tới:")
                for d, v in zip(next_months, pred):
                    st.metric(d.strftime("%m/%Y"), f"{v:,.0f}")
            with c2:
                fig = px.scatter(df, x="Tháng", y="Lợi Nhuận Sau Thuế", trendline="ols", title="Xu hướng Lợi nhuận")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⛔ Chỉ dành cho CFO.")

    # === TAB 4: CHAT ===
    with t4:
        st.subheader("Hỏi đáp số liệu")
        q = st.chat_input("VD: ROE tháng này có tốt không?")
        if q:
            st.chat_message("user").write(q)
            with st.chat_message("assistant"):
                # Gửi kèm dữ liệu tháng cuối để AI trả lời chính xác
                context = f"Dữ liệu tháng mới nhất: {latest.to_json()}"
                res = model.generate_content(f"Context: {context}. User Q: {q}. Role: Expert Finance. Lang: {st.session_state.lang_code}")
                st.markdown(res.text)

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
            st.caption("Demo: admin_cfo | chief_acc | staff_01")
    else:
        show_dashboard()

if __name__ == "__main__":
    main()
