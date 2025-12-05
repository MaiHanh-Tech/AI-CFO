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
st.set_page_config(page_title="AI Financial Controller Ultimate", layout="wide", page_icon="💰")

# --- TỪ ĐIỂN ĐA NGÔN NGỮ ---
TRANS = {
    "vi": {
        "title": "💰 Hệ thống Giám đốc Tài chính AI (CFO Ultimate)",
        "role_admin": "CFO (Giám đốc Tài chính)",
        "role_chief": "Kế toán trưởng",
        "role_staff": "Nhân viên Kế toán",
        "tab1": "📊 Bộ Chỉ Số KPIs",
        "tab2": "📉 Phân Tích Chi Phí",
        "tab3": "🕵️ Soát Xét Rủi Ro (ML)",
        "tab4": "🔮 Chiến Lược & Dự Báo",
        "tab5": "📚 Thư Viện Luật & Chat",
        "kpi_select": "Chọn Nhóm Chỉ Số muốn xem:",
        "grp_liquid": "Khả năng Thanh toán",
        "grp_profit": "Khả năng Sinh lời",
        "grp_activity": "Hiệu quả Hoạt động",
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
        "tab4": "🔮 Strategy Forecast",
        "tab5": "📚 Law & Chat",
        "kpi_select": "Select KPI Group:",
        "grp_liquid": "Liquidity",
        "grp_profit": "Profitability",
        "grp_activity": "Activity/Turnover",
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
        "tab5": "📚 法律与问答",
        "kpi_select": "选择指标组:",
        "grp_liquid": "偿债能力",
        "grp_profit": "盈利能力",
        "grp_activity": "营运能力",
        "btn_cn": "🇨🇳 生成中文汇报",
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

# --- 3. LOGIC TÀI CHÍNH (DATA GENERATOR SIÊU CẤP) ---
def tao_data_full_kpi():
    # Tạo dữ liệu đủ để tính mọi chỉ số Chị yêu cầu
    dates = pd.date_range(start="2024-01-01", periods=12, freq="ME")
    df = pd.DataFrame({"Tháng": dates})
    
    # P&L (Kết quả kinh doanh)
    df["Doanh Thu"] = np.random.randint(5000, 8000, 12) * 1000000
    df["Giá Vốn (Trực tiếp)"] = df["Doanh Thu"] * 0.6 # 60%
    df["Chi Phí VH (Gián tiếp)"] = np.random.randint(500, 800, 12) * 1000000
    df["Lợi Nhuận ST"] = df["Doanh Thu"] - df["Giá Vốn (Trực tiếp)"] - df["Chi Phí VH (Gián tiếp)"]
    
    # Balance Sheet (Cân đối kế toán - Bình quân)
    df["TS Ngắn Hạn"] = np.random.randint(2000, 3000, 12) * 1000000
    df["Nợ Ngắn Hạn"] = np.random.randint(1000, 1500, 12) * 1000000
    df["Hàng Tồn Kho"] = np.random.randint(800, 1200, 12) * 1000000
    df["Phải Thu KH"] = np.random.randint(1000, 1500, 12) * 1000000
    df["Tổng Tài Sản"] = df["TS Ngắn Hạn"] + 5000000000 # Cộng tài sản dài hạn cố định
    df["Vốn Chủ Sở Hữu"] = df["Tổng Tài Sản"] * 0.5 # Giả định 50% vốn
    
    # Gài bẫy cho ML bắt (Tháng 6 và 10 chi phí cao bất thường)
    df.loc[5, "Chi Phí VH (Gián tiếp)"] = 2500000000
    df.loc[9, "Chi Phí VH (Gián tiếp)"] = 2200000000
    
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
    # Dùng Isolation Forest (Cái cũ chị thích)
    model_iso = IsolationForest(contamination=0.1, random_state=42)
    # Soi trên Chi phí Vận hành
    df['Anomaly_Score'] = model_iso.fit_predict(df[['Chi Phí VH (Gián tiếp)']])
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
        if st.button("Tạo dữ liệu mẫu (Full KPIs)"):
            st.session_state.df_fin = tao_data_full_kpi()
            st.rerun()
        
        up = st.file_uploader("Upload Excel", type=['xlsx'])
        if up: st.session_state.df_fin = pd.read_excel(up)

        if st.button(T("logout")):
            st.session_state.is_logged_in = False; st.rerun()

    st.title(T("title"))

    if 'df_fin' not in st.session_state:
        # Màn hình chờ đẹp
        st.info("👈 Mời Giám đốc tạo dữ liệu mẫu hoặc Upload file.")
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown("### 📊 KPIs & HĐ Kinh Tế\nTính vòng quay, ROE, ROA.")
        with c2: st.markdown("### 🕵️ ML Risk Audit\nPhát hiện gian lận bằng AI.")
        with c3: st.markdown("### 🔮 Chiến Lược\nDự báo dòng tiền tương lai.")
        return

    df = st.session_state.df_fin
    last_month = df.iloc[-1]
    is_vip = role in ["admin", "chief"]
    
    t1, t2, t3, t4, t5 = st.tabs([T("tab1"), T("tab2"), T("tab3"), T("tab4"), T("tab5")])

    # === TAB 1: BỘ CHỈ SỐ TÀI CHÍNH (CHỌN ĐỂ XEM) ===
    with t1:
        st.subheader("Phân tích Hoạt động Kinh tế & Tài chính")
        
        # Multiselect để Chị chọn chỉ số muốn xem
        options = [T("grp_liquid"), T("grp_profit"), T("grp_activity")]
        selection = st.multiselect(T("kpi_select"), options, default=options)
        
        if T("grp_liquid") in selection:
            st.markdown(f"#### 💧 {T('grp_liquid')} (Thanh khoản)")
            k1, k2 = st.columns(2)
            curr_r = last_month["TS Ngắn Hạn"] / last_month["Nợ Ngắn Hạn"]
            quick_r = (last_month["TS Ngắn Hạn"] - last_month["Hàng Tồn Kho"]) / last_month["Nợ Ngắn Hạn"]
            k1.metric("Thanh toán hiện hành", f"{curr_r:.2f}", help="Lý tưởng: 2-3")
            k2.metric("Thanh toán nhanh", f"{quick_r:.2f}", help="Loại bỏ hàng tồn kho")
            st.divider()

        if T("grp_profit") in selection:
            st.markdown(f"#### 💰 {T('grp_profit')} (Sinh lời)")
            p1, p2, p3 = st.columns(3)
            ros = (last_month["Lợi Nhuận ST"] / last_month["Doanh Thu"]) * 100
            roa = (last_month["Lợi Nhuận ST"] / last_month["Tổng Tài Sản"]) * 100
            roe = (last_month["Lợi Nhuận ST"] / last_month["Vốn Chủ Sở Hữu"]) * 100
            p1.metric("ROS (Biên lãi ròng)", f"{ros:.1f}%")
            p2.metric("ROA (Trên tài sản)", f"{roa:.1f}%")
            p3.metric("ROE (Trên vốn chủ)", f"{roe:.1f}%")
            st.divider()

        if T("grp_activity") in selection:
            st.markdown(f"#### 🏭 {T('grp_activity')} (Hiệu quả)")
            a1, a2, a3 = st.columns(3)
            # Tính Vòng quay (giả định số liệu tháng là đại diện)
            inv_turn = last_month["Giá Vốn (Trực tiếp)"] / last_month["Hàng Tồn Kho"]
            ar_turn = last_month["Doanh Thu"] / last_month["Phải Thu KH"]
            asset_turn = last_month["Doanh Thu"] / last_month["Tổng Tài Sản"]
            
            a1.metric("Vòng quay Tồn kho", f"{inv_turn:.2f} vòng", "Tốc độ bán hàng")
            a2.metric("Vòng quay Phải thu", f"{ar_turn:.2f} vòng", "Tốc độ thu tiền")
            a3.metric("Vòng quay Tài sản", f"{asset_turn:.2f} vòng")

        if is_vip:
            st.markdown("---")
            if st.button(T("btn_cn"), type="primary"):
                with st.spinner("AI writing..."):
                    p = f"Role: CFO. Data Month: {last_month['Tháng']}. ROE: {roe}%. Inv Turnover: {inv_turn}. Current Ratio: {curr_r}. Write a professional report in Business Chinese."
                    res = model.generate_content(p)
                    st.info(res.text)

    # === TAB 2: PHÂN TÍCH CHI PHÍ (QUẢN TRỊ) ===
    with t2:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Cơ cấu Chi phí (Trực tiếp vs Gián tiếp)")
            # Stacked Bar Chart
            fig = px.bar(df, x="Tháng", y=["Giá Vốn (Trực tiếp)", "Chi Phí VH (Gián tiếp)"], title="Biến động Chi phí theo Tháng")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Tỷ trọng (Tháng cuối)")
            labels = ["Giá Vốn", "Chi Phí VH", "Lợi Nhuận"]
            values = [last_month["Giá Vốn (Trực tiếp)"], last_month["Chi Phí VH (Gián tiếp)"], last_month["Lợi Nhuận ST"]]
            fig2 = px.pie(values=values, names=labels, hole=0.4)
            st.plotly_chart(fig2, use_container_width=True)

    # === TAB 3: SOI RỦI RO (DÙNG ML CŨ CỦA CHỊ) ===
    with t3:
        if is_vip:
            st.header("Hệ thống Phát hiện Gian lận (Anomaly Detection)")
            st.caption("Sử dụng thuật toán Isolation Forest để tìm các khoản chi bất thường.")
            
            if st.button("🔍 QUÉT RỦI RO (ML SCAN)"):
                bad_data = phat_hien_gian_lan_ml(df.copy())
                if not bad_data.empty:
                    st.error(f"⚠️ CẢNH BÁO: Phát hiện {len(bad_data)} tháng có chi phí bất thường!")
                    st.dataframe(bad_data.style.highlight_max(axis=0, color='pink'))
                    
                    # AI Giải thích
                    with st.spinner("AI đang điều tra nguyên nhân..."):
                        res = model.generate_content(f"Phân tích dữ liệu bất thường này: {bad_data.to_string()}. Đưa ra 3 nguyên nhân (Gian lận? Mùa vụ? Sai sót?). Tiếng Việt.")
                        st.markdown(res.text)
                else:
                    st.success("✅ Hệ thống ML không tìm thấy bất thường.")
        else: st.warning("⛔ Restricted Area")

    # === TAB 4: DỰ BÁO (DÙNG LINEAR REGRESSION CŨ CỦA CHỊ) ===
    with t4:
        if st.session_state.user_role == "admin":
            st.header("Dự báo Chiến lược (Strategic Forecast)")
            
            # Chạy hồi quy
            df['idx'] = range(len(df))
            reg = LinearRegression().fit(df[['idx']], df['Lợi Nhuận ST'])
            future_X = np.array([[len(df)], [len(df)+1], [len(df)+2]])
            pred = reg.predict(future_X)
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Dự báo Tháng tới", f"{pred[0]:,.0f}")
                st.metric("Dự báo 2 tháng tới", f"{pred[1]:,.0f}")
                st.metric("Dự báo 3 tháng tới", f"{pred[2]:,.0f}")
            with c2:
                fig = px.scatter(df, x="Tháng", y="Lợi Nhuận ST", trendline="ols", title="Xu hướng Lợi nhuận")
                st.plotly_chart(fig, use_container_width=True)
        else: st.warning("⛔ Chỉ dành cho CFO.")

    # === TAB 5: THƯ VIỆN LUẬT & CHAT ===
    with t5:
        st.header("Trợ lý Pháp chế & Chat Dữ liệu")
        up_law = st.file_uploader("Upload Văn bản Luật/Báo cáo", type=["pdf", "docx"])
        if up_law:
            txt = doc_tai_lieu(up_law)
            st.success(f"Đã đọc xong {len(txt)} ký tự.")
            q = st.chat_input("Hỏi gì đó về văn bản này...")
            if q:
                st.chat_message("user").write(q)
                with st.chat_message("assistant"):
                    res = model.generate_content(f"Context: {txt[:30000]}. Q: {q}. Role: Legal Expert.")
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
