import streamlit as st
import pandas as pd
import os

# 頁面設定
st.set_page_config(page_title='銷售儀表板', layout='wide')

# 設定檔案路徑
data_dir = 'archive'
calendar_path = os.path.join(data_dir, 'calendar.csv')
customers_path = os.path.join(data_dir, 'customers.csv')
products_path = os.path.join(data_dir, 'products.csv')
stores_path = os.path.join(data_dir, 'stores.csv')
sales_path = os.path.join(data_dir, 'sales.csv')

# 讀取資料
calendar = pd.read_csv(calendar_path)
customers = pd.read_csv(customers_path)
products = pd.read_csv(products_path)
stores = pd.read_csv(stores_path)
# sales.csv 太大，僅讀取部分資料作為示範
sales = pd.read_csv(sales_path, nrows=100000)

# 資料前處理
sales['order_date'] = pd.to_datetime(sales['order_date'], errors='coerce')
sales = sales.merge(stores[['store_id', 'country']], on='store_id', how='left')
sales = sales.merge(products[['product_id', 'brand', 'category']], on='product_id', how='left')
sales = sales.merge(customers[['customer_id', 'loyalty_member']], on='customer_id', how='left')

# ===== 側邊欄篩選器 =====
st.sidebar.header('篩選器')

# 日期篩選
min_date = sales['order_date'].min().date()
max_date = sales['order_date'].max().date()
date_range = st.sidebar.date_input(
    '日期範圍',
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# 時間（月份）篩選
all_months = sorted(sales['order_date'].dt.month.dropna().unique())
month_names = {1: '一月', 2: '二月', 3: '三月', 4: '四月', 5: '五月', 6: '六月',
               7: '七月', 8: '八月', 9: '九月', 10: '十月', 11: '十一月', 12: '十二月'}
selected_months = st.sidebar.multiselect(
    '月份',
    options=all_months,
    default=all_months,
    format_func=lambda x: month_names[x],
)

# 國家篩選
all_countries = sorted(sales['country'].dropna().unique())
selected_countries = st.sidebar.multiselect(
    '國家',
    options=all_countries,
    default=all_countries,
)

# 品牌篩選
all_brands = sorted(sales['brand'].dropna().unique())
selected_brands = st.sidebar.multiselect(
    '品牌',
    options=all_brands,
    default=all_brands,
)

# 品類篩選
all_categories = sorted(sales['category'].dropna().unique())
selected_categories = st.sidebar.multiselect(
    '品類',
    options=all_categories,
    default=all_categories,
)

# 會員狀態篩選
loyalty_options = {'全部': None, '會員': 1, '非會員': 0}
selected_loyalty = st.sidebar.radio(
    '會員狀態',
    options=list(loyalty_options.keys()),
)

# ===== 套用篩選 =====
filtered = sales.copy()

# 日期範圍篩選
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        (filtered['order_date'].dt.date >= start_date) &
        (filtered['order_date'].dt.date <= end_date)
    ]

# 月份篩選
if selected_months:
    filtered = filtered[filtered['order_date'].dt.month.isin(selected_months)]

# 國家篩選
if selected_countries:
    filtered = filtered[filtered['country'].isin(selected_countries)]

# 品牌篩選
if selected_brands:
    filtered = filtered[filtered['brand'].isin(selected_brands)]

# 品類篩選
if selected_categories:
    filtered = filtered[filtered['category'].isin(selected_categories)]

# 會員狀態篩選
if loyalty_options[selected_loyalty] is not None:
    filtered = filtered[filtered['loyalty_member'] == loyalty_options[selected_loyalty]]

# ===== 主頁面 =====
st.title('銷售儀表板 Dashboard')

st.header('關鍵指標')

total_revenue = filtered['revenue'].sum()
total_profit = filtered['profit'].sum()
total_orders = filtered['order_id'].nunique()
active_customers = filtered['customer_id'].nunique()
avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric('總營收', f'${total_revenue:,.0f}')
col2.metric('毛利', f'${total_profit:,.0f}')
col3.metric('訂單數', f'{total_orders:,}')
col4.metric('活躍客戶數', f'{active_customers:,}')
col5.metric('客單價', f'${avg_order_value:,.1f}')

st.header('產品分析')
if 'product_id' in filtered.columns:
    top_products = filtered['product_id'].value_counts().head(10)
    st.bar_chart(top_products)

st.header('門市分析')
if 'store_id' in filtered.columns:
    top_stores = filtered['store_id'].value_counts().head(10)
    st.bar_chart(top_stores)

st.header('顧客分析')
if 'customer_id' in filtered.columns:
    top_customers = filtered['customer_id'].value_counts().head(10)
    st.bar_chart(top_customers)

st.header('時間分析')
sales_by_month = filtered.groupby(filtered['order_date'].dt.to_period('M')).size()
sales_by_month.index = sales_by_month.index.astype(str)
st.line_chart(sales_by_month)

# ===== RFM 分析 =====
st.header('RFM 分析')

analysis_date = filtered['order_date'].max() + pd.Timedelta(days=1)

# 計算每位顧客的 R、F、M
rfm = filtered.groupby('customer_id').agg(
    recency=('order_date', lambda x: (analysis_date - x.max()).days),
    frequency=('order_id', 'nunique'),
    monetary=('revenue', 'sum'),
).reset_index()

# 用四分位數打分（1-4，4 最好）
rfm['R_score'] = pd.qcut(rfm['recency'], 4, labels=[4, 3, 2, 1]).astype(int)
rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

# 客戶分群
def rfm_segment(row):
    if row['RFM_score'] >= 10:
        return '高價值客戶'
    elif row['RFM_score'] >= 7:
        return '中價值客戶'
    elif row['RFM_score'] >= 5:
        return '低價值客戶'
    else:
        return '流失風險客戶'

rfm['客戶分群'] = rfm.apply(rfm_segment, axis=1)

# 顯示分群統計
col_rfm1, col_rfm2 = st.columns(2)

with col_rfm1:
    st.subheader('客戶分群分佈')
    segment_counts = rfm['客戶分群'].value_counts()
    st.bar_chart(segment_counts)

with col_rfm2:
    st.subheader('各群平均指標')
    segment_summary = rfm.groupby('客戶分群').agg(
        平均近期天數=('recency', 'mean'),
        平均消費次數=('frequency', 'mean'),
        平均消費金額=('monetary', 'mean'),
        客戶數=('customer_id', 'count'),
    ).round(1)
    st.dataframe(segment_summary, use_container_width=True)

# RFM 散佈圖
st.subheader('RFM 散佈圖（頻率 vs 消費金額）')
scatter_data = rfm[['frequency', 'monetary', '客戶分群']].copy()
segment_order = ['高價值客戶', '中價值客戶', '低價值客戶', '流失風險客戶']
tabs = st.tabs(segment_order)
for tab, seg in zip(tabs, segment_order):
    with tab:
        seg_data = scatter_data[scatter_data['客戶分群'] == seg]
        st.scatter_chart(seg_data, x='frequency', y='monetary', height=400)

st.info('此為初步儀表板範例，可依需求擴充更多分析與視覺化。')
