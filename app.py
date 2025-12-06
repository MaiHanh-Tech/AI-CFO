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
from bs4 import BeautifulSoup
import requests
import io
import time
from google.api_core.exceptions import ResourceExhausted
try:
    from duckduckgo_search import DDGS
    HAS_SEARCH = True
except:
    HAS_SEARCH = False
    
# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="AI CFO Controller", layout="wide", page_icon="💰")

# --- TỪ ĐIỂN ĐA NGÔN NGỮ ---
TRANS = {
    "vi": {
        "title": "💰 Hệ thống Giám đốc Tài chính AI (CFO)",
        "login_title": "🔐 Đăng Nhập Hệ Thống",
        "lbl_user": "Tên đăng nhập",
        "lbl_pass": "Mật khẩu",
        "btn_login": "Đăng Nhập",
        "login_fail": "Sai thông tin đăng nhập!",
        "role_admin": "CFO (Giám đốc Tài chính)",
        "role_chief": "Kế toán trưởng",
        "role_staff": "Kế toán viên",
        "tab1": "📊 Bộ Chỉ Số KPIs",
        "tab2": "📉 Phân Tích Chi Phí",
        "tab3": "🕵️ Rủi Ro & Cross-Check",
        "tab4": "🔮 Dự Báo Đa Chiều", # Đổi tên
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
        "title": "💰 AI CFO Controller",
        "login_title": "🔐 System Login",
        "lbl_user": "Username",
        "lbl_pass": "Password",
        "btn_login": "Login",
        "login_fail": "Wrong credentials!",
        "role_admin": "CFO",
        "role_chief": "Chief Accountant",
        "role_staff": "Staff",
        "tab1": "📊 Financial KPIs",
        "tab2": "📉 Cost Analysis",
        "tab3": "🕵️ Risk & Cross-Check",
        "tab4": "🔮 Multi-dim Forecast",
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
        "login_title": "🔐 系统登录",
        "lbl_user": "用户名",
        "lbl_pass": "密码",
        "btn_login": "登录",
        "login_fail": "登录失败！",
        "role_admin": "财务总监 (CFO)",
        "role_chief": "财务经理",
        "role_staff": "会计",
        "tab1": "📊 财务指标 (KPIs)",
        "tab2": "📉 成本分析",
        "tab3": "🕵️ 风险 & 交叉检查",
        "tab4": "🔮 多维预测",
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
    return TRANS.get(lang, TRANS['vi']).get(key, key)

# --- 2. AUTH MANAGER ---
class AuthManager:
    def __init__(self):
        self.users = st.secrets.get("users", {})
        self.roles = st.secrets.get("roles", {})
    def verify(self, u, p): return u in self.users and self.users[u] == p
    def get_role(self, u): return self.roles.get(u, "staff")

# --- 3. CẤU HÌNH GEMINI (AN TOÀN) ---
try:
    if 'system' in st.secrets: api_key = st.secrets['system']['gemini_api_key']
    elif 'api_keys' in st.secrets: api_key = st.secrets['api_keys']['gemini_api_key']
    else: st.stop()
    
    genai.configure(api_key=api_key)
    try: model = genai.GenerativeModel('gemini-2.5-pro')
    except: 
        try: model = genai.GenerativeModel('gemini-2.5-flash')
        except: model = genai.GenerativeModel('gemini-pro') 
except: pass

def run_gemini_safe(model_func, prompt, retries=3):
    """Hàm gọi AI an toàn, chống lỗi Quota"""
    for i in range(retries):
        try: return model_func(prompt)
        except ResourceExhausted: time.sleep(5)
        except: return None
    return None

# --- IMPORT PROPHET (NẾU CÓ) ---
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

# --- 4. LOGIC TÀI CHÍNH & DỮ LIỆU ---

def tao_data_full_kpi():
    # Tạo 24 tháng để dự báo cho chuẩn
    dates = pd.date_range(start="2023-01-01", periods=24, freq="ME")
    df = pd.DataFrame({"Tháng": dates})
    
    np.random.seed(42) # Cố định random để số liệu ổn định
    
    # P&L Chi tiết
    df["Doanh Thu"] = np.random.randint(5000, 8000, 24) * 1000000
    df["Giá Vốn"] = df["Doanh Thu"] * 0.6 
    df["CP Lương"] = np.random.randint(500, 800, 24) * 1000000
    df["CP Marketing"] = df["Doanh Thu"] * 0.1
    df["CP Khác"] = np.random.randint(100, 200, 24) * 1000000
    
    df["Chi Phí VH"] = df["CP Lương"] + df["CP Marketing"] + df["CP Khác"]
    df["Lợi Nhuận ST"] = df["Doanh Thu"] - df["Giá Vốn"] - df["Chi Phí VH"]
    
    # Dòng tiền & Công nợ
    df["Dòng Tiền Thực"] = df["Lợi Nhuận ST"] * 0.8 # Lãi giả lỗ thật :))
    df["Công Nợ Phải Thu"] = np.random.randint(1000, 2000, 24) * 1000000
    
    # Hàng Tồn Kho Chi Tiết (Theo yêu cầu của chị)
    df["Tồn Kho - NVL Vải"] = np.random.randint(500, 800, 24) * 1000000
    df["Tồn Kho - NVL Chỉ"] = np.random.randint(100, 200, 24) * 1000000
    df["Tồn Kho - Thành Phẩm"] = np.random.randint(1000, 1500, 24) * 1000000
    df["Hàng Tồn Kho Tổng"] = df["Tồn Kho - NVL Vải"] + df["Tồn Kho - NVL Chỉ"] + df["Tồn Kho - Thành Phẩm"]
    
    # Balance Sheet cơ bản
    df["TS Ngắn Hạn"] = df["Tiền Mặt"] = np.random.randint(200, 500, 24) * 1000000
    df["Nợ Ngắn Hạn"] = np.random.randint(1000, 1500, 24) * 1000000
    df["Tổng Tài Sản"] = np.random.randint(10000, 12000, 24) * 1000000
    df["Vốn Chủ Sở Hữu"] = df["Tổng Tài Sản"] * 0.6

    return df

def tinh_chi_so_tai_chinh(df):
    required = ["TS Ngắn Hạn", "Nợ Ngắn Hạn", "Hàng Tồn Kho Tổng", "Doanh Thu", "Tổng Tài Sản", "Vốn Chủ Sở Hữu"]
    for c in required:
        if c not in df.columns: df[c] = 1 
    try:
        df["Current Ratio"] = df["TS Ngắn Hạn"] / df["Nợ Ngắn Hạn"].replace(0, 1)
        df["Quick Ratio"] = (df["TS Ngắn Hạn"] - df.get("Hàng Tồn Kho Tổng", 0)) / df["Nợ Ngắn Hạn"].replace(0, 1)
        df["Inv Turnover"] = df.get("Giá Vốn", 0) / df.get("Hàng Tồn Kho Tổng", 1).replace(0, 1)
        df["Gross Margin"] = (df.get("Doanh Thu", 0) - df.get("Giá Vốn", 0)) / df["Doanh Thu"].replace(0, 1) * 100
        df["ROS"] = (df.get("Lợi Nhuận ST", 0) / df["Doanh Thu"].replace(0, 1)) * 100
        df["ROE"] = (df.get("Lợi Nhuận ST", 0) / df["Vốn Chủ Sở Hữu"].replace(0, 1)) * 100
        df["Debt/Equity"] = (df.get("Nợ Ngắn Hạn", 0) / df["Vốn Chủ Sở Hữu"].replace(0, 1)) * 100
    except: pass
    return df

# --- 5. CÁC HÀM ĐỌC & ML ---
# --- CẬP NHẬT HÀM ĐỌC WEB (Vượt tường lửa) ---
def doc_url(url):
    try:
        # Giả lập trình duyệt Chrome để không bị chặn
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        # Nếu web lỗi (404, 403), trả về rỗng để code chính xử lý
        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Xóa quảng cáo, menu rác
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.extract()
            
        # Lấy toàn bộ chữ trong các thẻ quan trọng
        text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'div', 'article'])])
        
        # Làm sạch khoảng trắng thừa
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text[:30000] # Lấy tối đa 30.000 ký tự
    except Exception:
        return ""

def doc_tai_lieu(uploaded_file):
    try:
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext == 'pdf': 
            return "\n".join([p.extract_text() for p in PdfReader(uploaded_file).pages])
        elif ext == 'docx': 
            return "\n".join([p.text for p in Document(uploaded_file).paragraphs])
        elif ext in ['txt', 'md']: 
            return str(uploaded_file.read(), "utf-8")
        elif ext in ['html', 'htm']: 
            soup = BeautifulSoup(uploaded_file, "html.parser")
            return soup.get_text()
    except Exception as e: return f"Lỗi đọc file: {e}"
    return ""

def phat_hien_gian_lan_ml(df):
    model_iso = IsolationForest(contamination=0.1, random_state=42)
    # Ưu tiên cột chi phí hoặc lợi nhuận
    col = "Lợi Nhuận ST" if "Lợi Nhuận ST" in df.columns else df.select_dtypes(include=np.number).columns[0]
    try:
        df['Anomaly'] = model_iso.fit_predict(df[[col]])
        return df[df['Anomaly'] == -1]
    except: return pd.DataFrame()

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
        if st.button("🔄 Tạo Dữ Liệu Mẫu (Full)", type="primary"):
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

    # === TAB 1: BỘ CHỈ SỐ KPIs ===
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
            
        if T("grp_struct") in sels:
            st.divider()
            st.markdown(f"#### ⚖️ {T('grp_struct')}")
            k1, k2 = st.columns(2)
            k1.metric("Debt/Equity", f"{last.get('Debt/Equity', 0):.1f}%")

        if is_vip:
            st.markdown("---")
            if st.button(T("btn_cn"), type="primary"):
                with st.spinner("Writing..."):
                    p = f"Role: CFO. Data: {last.to_dict()}. Write Business Chinese report."
                    res = run_gemini_safe(model.generate_content, p)
                    if res: st.info(res.text)

    # === TAB 2: CHI PHÍ & BIỂU ĐỒ ===
    with t2:
        st.subheader("Phân tích Chi phí")
        c1, c2 = st.columns([2, 1])
        with c1:
            cols = [c for c in ["Giá Vốn", "Chi Phí VH"] if c in df.columns]
            if cols: st.plotly_chart(px.bar(df, x="Tháng", y=cols, title="Cost Structure"), use_container_width=True)
        with c2:
            if "Lợi Nhuận ST" in df.columns:
                vals = [last.get(c, 0) for c in cols] + [last.get("Lợi Nhuận ST", 0)]
                fig2 = px.pie(values=vals, names=cols + ["Lợi Nhuận"], title="Tỷ trọng tháng cuối")
                st.plotly_chart(fig2, use_container_width=True)

    # === TAB 3: RỦI RO & CROSS-CHECK ===
    with t3:
        if is_vip:
            st.header("🕵️ Soát Xét Rủi Ro & Đối Chiếu (Cross-Check)")
            col_risk, col_check = st.columns([1, 1])
            
            with col_risk:
                st.subheader("A. Quét Bất Thường (Machine Learning)")
                if st.button("🔍 QUÉT GIAN LẬN"):
                    bad = phat_hien_gian_lan_ml(df.copy())
                    if not bad.empty:
                        st.error(f"⚠️ Phát hiện {len(bad)} tháng bất thường (Anomaly)!")
                        st.dataframe(bad)
                    else: st.success("✅ Số liệu ổn định (Isolation Forest).")

            with col_check:
                st.subheader("B. Cross-Check: Soi Mâu Thuẫn")
                check_type = st.selectbox("Chọn loại đối chiếu:", ["Doanh Thu (Thuế vs Sổ Cái)", "Tồn Kho (Thực tế vs Sổ sách)", "Công Nợ (Kế toán vs Kinh doanh)"])
                default_val = float(last.get("Doanh Thu", 1000000000))
                c_k1, c_k2 = st.columns(2)
                with c_k1: val_source1 = st.number_input(f"Số liệu Nguồn A (VD: Tờ khai):", value=default_val)
                with c_k2: val_source2 = st.number_input(f"Số liệu Nguồn B (VD: Sổ sách):", value=default_val * 1.05)
                
                if st.button("⚖️ THỰC HIỆN ĐỐI CHIẾU"):
                    diff = val_source2 - val_source1
                    if abs(diff) > 1000:
                        st.error(f"⚠️ CẢNH BÁO: Lệch {diff:,.0f}")
                        with st.spinner("AI đang suy luận..."):
                            prompt = f"Kế toán trưởng đối chiếu: {check_type}. Lệch: {diff:,.0f}. Tại sao? Rủi ro thuế?"
                            res = run_gemini_safe(model.generate_content, prompt)
                            if res: st.markdown(res.text)
                    else: st.success("✅ Số liệu khớp.")
        else: st.warning("Restricted Area")

    # === TAB 4: DỰ BÁO TOÀN DIỆN & WHAT-IF (ĐÃ NÂNG CẤP DỰ BÁO) ===
    with t4:
        if st.session_state.user_role == "admin":
            st.header("🔮 Cỗ Máy Tiên Tri & Giả Lập Chiến Lược")
            
            # --- PHẦN 1: DỰ BÁO ĐA CHIỀU (ALL METRICS) ---
            with st.expander("📉 Dự báo Xu hướng (Tất cả chỉ tiêu)", expanded=True):
                # 1. Lọc lấy tất cả các cột số (trừ cột Tháng, idx)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                # Loại bỏ các cột không nên dự báo
                exclude = ['idx', 'Anomaly']
                valid_cols = [c for c in numeric_cols if c not in exclude]
                
                # 2. Cho người dùng chọn
                st.info("💡 Chọn bất kỳ chỉ tiêu nào (Doanh thu, Chi phí, Tồn kho từng loại...) để dự báo.")
                target_col = st.selectbox("🎯 Chọn chỉ tiêu muốn tiên tri:", valid_cols, index=valid_cols.index("Lợi Nhuận ST") if "Lợi Nhuận ST" in valid_cols else 0)
                
                if st.button(f"🚀 Dự báo '{target_col}' 12 tháng tới"):
                    with st.spinner(f"Đang chạy thuật toán dự báo cho {target_col}..."):
                        c1, c2 = st.columns([2, 1])
                        
                        # LOGIC DỰ BÁO (Prophet hoặc Linear)
                        forecast_df = None
                        
                        if HAS_PROPHET:
                            try:
                                df_p = df[['Tháng', target_col]].rename(columns={'Tháng': 'ds', target_col: 'y'})
                                m = Prophet()
                                m.fit(df_p)
                                future = m.make_future_dataframe(periods=12, freq='M')
                                forecast = m.predict(future)
                                forecast_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'Tháng', 'yhat': 'Dự Báo'})
                                # Lọc lấy phần tương lai
                                future_only = forecast_df.tail(12)
                            except: pass
                        
                        # Fallback nếu Prophet lỗi hoặc không có
                        if forecast_df is None:
                            df['idx'] = range(len(df))
                            reg = LinearRegression().fit(df[['idx']], df[target_col])
                            fut_idx = np.array(range(len(df), len(df)+12)).reshape(-1, 1)
                            pred = reg.predict(fut_idx)
                            last_date = df['Tháng'].iloc[-1]
                            fut_dates = pd.date_range(start=last_date, periods=13, freq="ME")[1:]
                            future_only = pd.DataFrame({'Tháng': fut_dates, 'Dự Báo': pred})

                        with c1:
                            # Vẽ biểu đồ nối đuôi
                            fig_fc = go.Figure()
                            fig_fc.add_trace(go.Scatter(x=df['Tháng'], y=df[target_col], name='Thực tế (Quá khứ)', line=dict(color='blue')))
                            fig_fc.add_trace(go.Scatter(x=future_only['Tháng'], y=future_only['Dự Báo'], name='Dự báo (Tương lai)', line=dict(color='orange', dash='dash')))
                            fig_fc.update_layout(title=f"Xu hướng: {target_col}", xaxis_title="Thời gian", yaxis_title="Giá trị")
                            st.plotly_chart(fig_fc, use_container_width=True)
                        
                        with c2:
                            st.markdown(f"**Chi tiết dự báo:**")
                            st.dataframe(future_only.style.format({"Dự Báo": "{:,.0f}"}), height=300)

            # --- PHẦN 2: WHAT-IF ANALYSIS ---
            st.divider()
            st.subheader("🎛️ What-If Analysis: Giả lập Kịch bản")
            st.markdown("Giả định thay đổi các yếu tố đầu vào để xem Lợi nhuận thay đổi ra sao.")
            
            # Lấy số liệu gốc
            base_rev = last.get("Doanh Thu", 0)
            base_salary = last.get("CP Lương", base_rev * 0.15) 
            base_cogs = last.get("Giá Vốn", base_rev * 0.6)
            base_other = last.get("Chi Phí VH", 0) - base_salary
            base_profit = last.get("Lợi Nhuận ST", 0)
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1: delta_price = st.slider("🏷️ Giá Bán (%)", -30, 30, 0)
            with col_s2: delta_salary = st.slider("👮 Lương (%)", -30, 30, 0)
            with col_s3: delta_cogs = st.slider("🏭 Giá Vốn (%)", -30, 30, 0)
            
            # Tính toán
            sim_rev = base_rev * (1 + delta_price/100)
            sim_salary = base_salary * (1 + delta_salary/100)
            sim_cogs = base_cogs * (1 + delta_cogs/100)
            sim_profit = sim_rev - sim_cogs - sim_salary - base_other
            
            # Hiển thị
            k1, k2, k3 = st.columns(3)
            k1.metric("Doanh thu Mới", f"{sim_rev:,.0f}", delta=f"{sim_rev-base_rev:,.0f}")
            k2.metric("Tổng Chi phí Mới", f"{(sim_cogs+sim_salary+base_other):,.0f}")
            k3.metric("LỢI NHUẬN MỚI", f"{sim_profit:,.0f}", delta=f"{sim_profit-base_profit:,.0f}", delta_color="normal")
            
            # Biểu đồ Waterfall
            fig_sim = go.Figure(go.Waterfall(
                name = "Kịch bản", orientation = "v",
                measure = ["relative", "relative", "relative", "total"],
                x = ["Lợi Nhuận Gốc", "Tác động Giá", "Tác động Chi Phí", "Lợi Nhuận Mới"],
                text = [f"{base_profit/1e6:.0f}M", f"{(sim_rev-base_rev)/1e6:+.0f}M", f"-{(sim_cogs-base_cogs + sim_salary-base_salary)/1e6:+.0f}M", f"{sim_profit/1e6:.0f}M"],
                y = [base_profit, sim_rev-base_rev, -(sim_cogs-base_cogs + sim_salary-base_salary), sim_profit],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
            ))
            st.plotly_chart(fig_sim, use_container_width=True)

        else: st.warning("⛔ Chỉ dành cho CFO.")

   # === TAB 5: PHÁP CHẾ (BẢN SỬA LỖI & KHÔI PHỤC) ===
    with t5:
        st.header("⚖️ Trung Tâm Pháp Chế & Tra Cứu")
        
        # Chia cột: Bên trái Tìm kiếm - Bên phải Nạp thủ công
        col_auto, col_manual = st.columns([1, 1])
        
        # --- CỘT TRÁI: TÌM KIẾM TỰ ĐỘNG ---
        with col_auto:
            st.subheader("🔍 A. Tìm & Đọc Tự Động")
            if not HAS_SEARCH:
                st.error("⚠️ Chưa cài `duckduckgo-search`")
            else:
                search_kw = st.text_input("Nhập từ khóa luật (VD: Thuế TNCN 2024):", key="search_input")
                
                if st.button("🔎 TÌM KIẾM & ĐỌC WEB", type="primary"):
                    if search_kw:
                        status = st.status(f"🤖 Đang tìm kiếm '{search_kw}'...")
                        try:
                            # 1. Tìm kiếm
                            results = DDGS().text(search_kw, region='vn-vn', max_results=5)
                            
                            found_text = ""
                            count_ok = 0
                            
                            for res in results:
                                title = res['title']
                                link = res['href']
                                snippet = res['body']
                                
                                status.write(f"🔗 Thử đọc: {title}...")
                                
                                # 2. Thử đọc nội dung sâu bên trong
                                deep_content = doc_url(link)
                                
                                if len(deep_content) > 500:
                                    # Nếu đọc được nội dung dài > 500 ký tự -> Tốt
                                    found_text += f"\n\n=== NGUỒN (Full): {title} ===\nLink: {link}\nNội dung: {deep_content}"
                                    count_ok += 1
                                else:
                                    # Nếu bị chặn không đọc được -> Dùng tạm đoạn tóm tắt (Snippet)
                                    found_text += f"\n\n=== NGUỒN (Tóm tắt): {title} ===\nLink: {link}\nNội dung: {snippet}..."
                            
                            # 3. Nạp vào bộ nhớ
                            if found_text:
                                if 'legal_data' not in st.session_state: st.session_state.legal_data = ""
                                st.session_state.legal_data += found_text
                                status.update(label=f"✅ Đã nạp dữ liệu từ {len(results)} trang web (Đọc sâu được {count_ok} trang)!", state="complete")
                            else:
                                status.update(label="❌ Không tìm thấy kết quả nào.", state="error")
                                
                        except Exception as e:
                            status.update(label=f"⚠️ Lỗi kết nối: {str(e)}", state="error")

        # --- CỘT PHẢI: NẠP THỦ CÔNG (ĐÃ KHÔI PHỤC) ---
        with col_manual:
            st.subheader("📂 B. Nạp File & Link Thủ Công")
            
            with st.expander("Mở bảng nạp dữ liệu", expanded=True):
                # 1. Upload File
                up_laws = st.file_uploader("1. Chọn File (PDF/Docx/Txt):", type=["pdf", "docx", "txt"], accept_multiple_files=True)
                
                # 2. Dán Link (ĐÃ KHÔI PHỤC)
                manual_links = st.text_area("2. Hoặc Dán Link Web (Mỗi link 1 dòng):", height=100, placeholder="https://thuvienphapluat.vn/...")
                
                # Nút xử lý chung cho cột phải
                if st.button("📥 NẠP DỮ LIỆU THỦ CÔNG"):
                    manual_text = ""
                    
                    # Xử lý File
                    if up_laws:
                        for f in up_laws:
                            manual_text += f"\n=== FILE: {f.name} ===\n" + doc_tai_lieu(f)
                    
                    # Xử lý Link
                    if manual_links:
                        links = manual_links.split('\n')
                        status_manual = st.status("Đang đọc link thủ công...")
                        for lnk in links:
                            if lnk.strip():
                                status_manual.write(f"Đang đọc: {lnk}...")
                                content = doc_url(lnk.strip())
                                if content:
                                    manual_text += f"\n=== WEB THỦ CÔNG: {lnk} ===\n{content}"
                                else:
                                    status_manual.warning(f"Không đọc được: {lnk}")
                        status_manual.update(label="Xong!", state="complete")

                    if manual_text:
                        if 'legal_data' not in st.session_state: st.session_state.legal_data = ""
                        st.session_state.legal_data += manual_text
                        st.success(f"✅ Đã nạp thêm {len(manual_text):,} ký tự vào bộ nhớ.")
                    else:
                        st.warning("Chưa có dữ liệu để nạp.")

        # --- PHẦN HỎI ĐÁP ---
        st.divider()
        mem_len = len(st.session_state.get('legal_data', ''))
        st.caption(f"🧠 Bộ nhớ hiện tại: **{mem_len:,}** ký tự.")
        
        q = st.chat_input("Hỏi luật sư AI...")
        if q:
            st.chat_message("user").write(q)
            with st.chat_message("assistant"):
                if mem_len == 0:
                    st.warning("⚠️ Bộ nhớ trống. Vui lòng Tìm kiếm hoặc Nạp file trước.")
                else:
                    with st.spinner("Đang tra cứu..."):
                        ctx = st.session_state.legal_data[:500000]
                        prompt = f"Dựa vào dữ liệu: {ctx}\nCâu hỏi: '{q}'\nTrả lời chi tiết + Trích nguồn."
                        res = run_gemini_safe(model.generate_content, prompt)
                        if res: st.markdown(res.text)
            
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
