import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np
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
sales = pd.read_csv(sales_path, nrows=100000)

# 資料前處理
sales['order_date'] = pd.to_datetime(sales['order_date'], errors='coerce')
sales = sales.merge(stores[['store_id', 'country', 'city']], on='store_id', how='left')
sales = sales.merge(products[['product_id', 'brand', 'category']], on='product_id', how='left')
sales = sales.merge(customers[['customer_id', 'loyalty_member', 'age', 'gender', 'join_date']], on='customer_id', how='left')
sales['join_date'] = pd.to_datetime(sales['join_date'], errors='coerce')

# ===== 側邊欄篩選器 =====
st.sidebar.header('篩選器')

min_date = sales['order_date'].min().date()
max_date = sales['order_date'].max().date()
date_range = st.sidebar.date_input(
    '日期範圍',
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

all_months = sorted(sales['order_date'].dt.month.dropna().unique())
month_names = {1: '一月', 2: '二月', 3: '三月', 4: '四月', 5: '五月', 6: '六月',
               7: '七月', 8: '八月', 9: '九月', 10: '十月', 11: '十一月', 12: '十二月'}
selected_months = st.sidebar.multiselect(
    '月份', options=all_months, default=all_months,
    format_func=lambda x: month_names[x],
)

all_countries = sorted(sales['country'].dropna().unique())
selected_countries = st.sidebar.multiselect('國家', options=all_countries, default=all_countries)

all_brands = sorted(sales['brand'].dropna().unique())
selected_brands = st.sidebar.multiselect('品牌', options=all_brands, default=all_brands)

all_categories = sorted(sales['category'].dropna().unique())
selected_categories = st.sidebar.multiselect('品類', options=all_categories, default=all_categories)

loyalty_options = {'全部': None, '會員': 1, '非會員': 0}
selected_loyalty = st.sidebar.radio('會員狀態', options=list(loyalty_options.keys()))

# ===== 套用篩選 =====
filtered = sales.copy()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        (filtered['order_date'].dt.date >= start_date) &
        (filtered['order_date'].dt.date <= end_date)
    ]
if selected_months:
    filtered = filtered[filtered['order_date'].dt.month.isin(selected_months)]
if selected_countries:
    filtered = filtered[filtered['country'].isin(selected_countries)]
if selected_brands:
    filtered = filtered[filtered['brand'].isin(selected_brands)]
if selected_categories:
    filtered = filtered[filtered['category'].isin(selected_categories)]
if loyalty_options[selected_loyalty] is not None:
    filtered = filtered[filtered['loyalty_member'] == loyalty_options[selected_loyalty]]

# ===== 主頁面（分頁選單） =====
st.title('銷售儀表板 Dashboard')

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    '總覽', 'RFM 分析', 'CAI 顧客活躍度', 'ANOVA 檢定',
    '人口統計分析', '地圖與會員趨勢',
])

# ==================== Tab 1: 總覽 ====================
with tab1:
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
    top_products = filtered['product_id'].value_counts().head(10)
    st.bar_chart(top_products)

    st.header('門市分析')
    top_stores = filtered['store_id'].value_counts().head(10)
    st.bar_chart(top_stores)

    st.header('顧客分析')
    top_customers = filtered['customer_id'].value_counts().head(10)
    st.bar_chart(top_customers)

    st.header('時間分析')
    sales_by_month = filtered.groupby(filtered['order_date'].dt.to_period('M')).size()
    sales_by_month.index = sales_by_month.index.astype(str)
    st.line_chart(sales_by_month)

# ==================== Tab 2: RFM 分析 ====================
with tab2:
    st.header('RFM 分析（80/20 法則）')

    analysis_date = filtered['order_date'].max() + pd.Timedelta(days=1)

    rfm = filtered.groupby('customer_id').agg(
        recency=('order_date', lambda x: (analysis_date - x.max()).days),
        frequency=('order_id', 'nunique'),
        monetary=('revenue', 'sum'),
    ).reset_index()

    def score_80_20(series, reverse=False):
        p20 = series.quantile(0.20)
        p80 = series.quantile(0.80)
        if reverse:
            return series.apply(lambda x: 3 if x <= p20 else (2 if x <= p80 else 1))
        else:
            return series.apply(lambda x: 1 if x <= p20 else (2 if x <= p80 else 3))

    rfm['R'] = score_80_20(rfm['recency'], reverse=True)
    rfm['F'] = score_80_20(rfm['frequency'], reverse=False)
    rfm['M'] = score_80_20(rfm['monetary'], reverse=False)
    rfm['RFM_code'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)

    rfm_names = {
        '333': '鑽石級忠誠客戶', '332': '高頻高近期中消費', '331': '高頻高近期低消費',
        '323': '高近期中頻高消費', '322': '穩定中堅客戶', '321': '高近期中頻低消費',
        '313': '高近期低頻高消費', '312': '高近期低頻中消費', '311': '近期新客低活躍',
        '233': '中近期高頻高消費', '232': '中近期高頻中消費', '231': '中近期高頻低消費',
        '223': '中近期中頻高消費', '222': '一般普通客戶', '221': '中近期中頻低消費',
        '213': '中近期低頻高消費', '212': '中近期低頻中消費', '211': '中近期低頻低消費',
        '133': '流失風險高頻高消費', '132': '流失風險高頻中消費', '131': '流失風險高頻低消費',
        '123': '流失風險中頻高消費', '122': '流失風險中頻中消費', '121': '流失風險中頻低消費',
        '113': '沉睡高消費客戶', '112': '沉睡中消費客戶', '111': '已流失低價值客戶',
    }
    rfm['客戶分群'] = rfm['RFM_code'].map(rfm_names).fillna('其他')

    # 27 群總表
    st.subheader('27 群客戶分群總表')
    group_summary = rfm.groupby(['RFM_code', '客戶分群']).agg(
        客戶數=('customer_id', 'count'),
        平均近期天數=('recency', 'mean'),
        平均消費次數=('frequency', 'mean'),
        平均消費金額=('monetary', 'mean'),
    ).round(1).reset_index().rename(columns={'RFM_code': 'RFM 編碼'})
    group_summary = group_summary.sort_values('RFM 編碼', ascending=False)
    st.dataframe(group_summary, use_container_width=True, height=500)

    # 視覺化
    col_rfm1, col_rfm2 = st.columns(2)

    with col_rfm1:
        st.subheader('R / F / M 各維度分佈')
        rfm_melt = rfm[['R', 'F', 'M']].melt(var_name='維度', value_name='分數')
        dim_counts = rfm_melt.groupby(['維度', '分數']).size().reset_index(name='人數')
        dim_counts['分數標籤'] = dim_counts['分數'].map({1: 'Bottom 20%', 2: 'Middle', 3: 'Top 20%'})
        fig_dim = px.bar(
            dim_counts, x='維度', y='人數', color='分數標籤', barmode='group',
            color_discrete_map={'Top 20%': '#2ecc71', 'Middle': '#3498db', 'Bottom 20%': '#e74c3c'},
        )
        fig_dim.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400, legend_title='分組')
        st.plotly_chart(fig_dim, use_container_width=True)

    with col_rfm2:
        st.subheader('各群客戶人數（Top 15）')
        top15 = rfm['客戶分群'].value_counts().head(15)
        fig_bar = px.bar(
            x=top15.values, y=top15.index, orientation='h',
            labels={'x': '客戶數', 'y': '客戶分群'},
            color=top15.values, color_continuous_scale='YlGnBu',
        )
        fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400,
                              yaxis=dict(autorange='reversed'), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # 3D 散佈圖
    st.subheader('RFM 3D 散佈圖')
    fig_3d = px.scatter_3d(
        rfm, x='recency', y='frequency', z='monetary', color='RFM_code', opacity=0.5,
        labels={'recency': 'Recency（天）', 'frequency': 'Frequency（次）', 'monetary': 'Monetary（$）', 'RFM_code': 'RFM 編碼'},
    )
    fig_3d.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=550)
    st.plotly_chart(fig_3d, use_container_width=True)

    # R-F 熱力圖 + 散佈圖
    col_rfm3, col_rfm4 = st.columns(2)

    with col_rfm3:
        st.subheader('R-F 分數熱力圖（客戶數）')
        heatmap_data = rfm.groupby(['R', 'F']).size().reset_index(name='客戶數')
        heatmap_pivot = heatmap_data.pivot(index='R', columns='F', values='客戶數').fillna(0).astype(int)
        heatmap_pivot = heatmap_pivot.sort_index(ascending=False)
        score_labels = {1: 'Bottom 20%', 2: 'Middle', 3: 'Top 20%'}
        fig_heat = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=[f'F: {score_labels[c]}' for c in heatmap_pivot.columns],
            y=[f'R: {score_labels[r]}' for r in heatmap_pivot.index],
            text=heatmap_pivot.values, texttemplate='%{text}', colorscale='YlGnBu',
            hovertemplate='%{y}, %{x}<br>客戶數: %{z}<extra></extra>',
        ))
        fig_heat.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400,
                               xaxis_title='Frequency', yaxis_title='Recency')
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_rfm4:
        st.subheader('頻率 vs 消費金額')
        fig_scatter = px.scatter(
            rfm, x='frequency', y='monetary', color='RFM_code', opacity=0.4,
            labels={'frequency': '消費次數', 'monetary': '消費金額（$）', 'RFM_code': 'RFM 編碼'},
        )
        fig_scatter.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 雷達圖
    st.subheader('RFM 大類雷達圖')
    rfm['大類'] = rfm['RFM_code'].apply(
        lambda x: '鑽石客戶' if x == '333'
        else '高價值客戶' if int(x[0]) == 3 and (int(x[1]) + int(x[2])) >= 5
        else '成長型客戶' if int(x[0]) >= 2 and int(x[1]) >= 2
        else '流失風險客戶' if int(x[0]) == 1
        else '一般客戶'
    )
    macro_order = ['鑽石客戶', '高價值客戶', '成長型客戶', '一般客戶', '流失風險客戶']
    macro_colors = {'鑽石客戶': '#9b59b6', '高價值客戶': '#2ecc71', '成長型客戶': '#3498db', '一般客戶': '#f39c12', '流失風險客戶': '#e74c3c'}
    radar_data = rfm.groupby('大類').agg(R=('R', 'mean'), F=('F', 'mean'), M=('M', 'mean')).reindex(macro_order).dropna().round(2)
    fig_radar = go.Figure()
    for seg in radar_data.index:
        values = radar_data.loc[seg].tolist() + [radar_data.loc[seg].tolist()[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=values, theta=['Recency', 'Frequency', 'Monetary', 'Recency'],
            fill='toself', name=seg, line=dict(color=macro_colors[seg]), opacity=0.6,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 3.2])),
        margin=dict(t=30, b=30, l=60, r=60), height=450,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ==================== Tab 3: CAI 顧客活躍度 ====================
with tab3:
    st.header('CAI 顧客活躍度分析')
    st.markdown("""
    **CAI（Customer Activity Index）公式：**
    $$CAI = \\frac{\\text{購買間隔平均時長} - \\text{加權購買間隔平均時長}}{\\text{購買間隔平均時長}} \\times 100\\%$$

    - 加權方式：越後面（越近期）的間隔權重越大。有 n 個間隔時，最近一個間隔權重為 n，倒數第二個為 n-1，...，最早的為 1。
    - **CAI > 0**：購買間隔越來越短 → 活躍度上升
    - **CAI < 0**：購買間隔越來越長 → 活躍度下降
    - **CAI ≈ 0**：購買間隔穩定
    - 至少需要 **3 個間隔**（4 次購買）才能計算
    """)

    # 計算每位顧客的購買日期序列
    customer_dates = filtered.groupby('customer_id')['order_date'].apply(
        lambda x: sorted(x.dropna().unique())
    )

    def calc_cai(dates):
        if len(dates) < 4:
            return np.nan
        intervals = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
        n = len(intervals)
        if n < 3:
            return np.nan
        avg_interval = np.mean(intervals)
        if avg_interval == 0:
            return np.nan
        weights = list(range(1, n + 1))
        weighted_avg = np.average(intervals, weights=weights)
        cai = (avg_interval - weighted_avg) / avg_interval * 100
        return round(cai, 2)

    cai_series = customer_dates.apply(calc_cai)
    cai_df = cai_series.reset_index()
    cai_df.columns = ['customer_id', 'CAI']
    cai_df = cai_df.dropna()

    # 合併消費資訊
    cust_monetary = filtered.groupby('customer_id').agg(
        總消費金額=('revenue', 'sum'),
        購買次數=('order_id', 'nunique'),
    ).reset_index()
    cai_df = cai_df.merge(cust_monetary, on='customer_id', how='left')

    st.subheader(f'可計算 CAI 的顧客數：{len(cai_df):,} 人（需至少 4 次購買）')

    # CAI 分群
    def cai_segment(cai):
        if cai > 20:
            return '高度活躍（加速購買）'
        elif cai > 0:
            return '輕度活躍（略有加速）'
        elif cai > -20:
            return '輕度衰退（略有減速）'
        else:
            return '高度衰退（明顯流失）'

    cai_df['活躍分群'] = cai_df['CAI'].apply(cai_segment)

    # 關鍵指標
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    col_c1.metric('平均 CAI', f'{cai_df["CAI"].mean():.2f}%')
    col_c2.metric('CAI 中位數', f'{cai_df["CAI"].median():.2f}%')
    col_c3.metric('活躍客戶佔比', f'{(cai_df["CAI"] > 0).mean():.1%}')
    col_c4.metric('衰退客戶佔比', f'{(cai_df["CAI"] < 0).mean():.1%}')

    # 第一排：CAI 分佈直方圖 + 分群圓餅圖
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.subheader('CAI 分佈')
        fig_hist = px.histogram(
            cai_df, x='CAI', nbins=50,
            labels={'CAI': 'CAI（%）', 'count': '顧客數'},
            color_discrete_sequence=['#3498db'],
        )
        fig_hist.add_vline(x=0, line_dash='dash', line_color='red', annotation_text='CAI=0')
        fig_hist.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_v2:
        st.subheader('活躍分群分佈')
        seg_order = ['高度活躍（加速購買）', '輕度活躍（略有加速）', '輕度衰退（略有減速）', '高度衰退（明顯流失）']
        seg_colors = {
            '高度活躍（加速購買）': '#2ecc71', '輕度活躍（略有加速）': '#3498db',
            '輕度衰退（略有減速）': '#f39c12', '高度衰退（明顯流失）': '#e74c3c',
        }
        seg_counts = cai_df['活躍分群'].value_counts().reindex(seg_order).fillna(0)
        fig_pie_cai = px.pie(
            values=seg_counts.values, names=seg_counts.index,
            color=seg_counts.index, color_discrete_map=seg_colors, hole=0.4,
        )
        fig_pie_cai.update_traces(textinfo='percent+label')
        fig_pie_cai.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=400)
        st.plotly_chart(fig_pie_cai, use_container_width=True)

    # 第二排：CAI vs 消費金額 + 各群統計
    col_v3, col_v4 = st.columns(2)

    with col_v3:
        st.subheader('CAI vs 總消費金額')
        fig_cai_scatter = px.scatter(
            cai_df, x='CAI', y='總消費金額', color='活躍分群',
            color_discrete_map=seg_colors, opacity=0.5,
            labels={'CAI': 'CAI（%）', '總消費金額': '總消費金額（$）'},
            category_orders={'活躍分群': seg_order},
        )
        fig_cai_scatter.add_vline(x=0, line_dash='dash', line_color='gray')
        fig_cai_scatter.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400)
        st.plotly_chart(fig_cai_scatter, use_container_width=True)

    with col_v4:
        st.subheader('各群統計摘要')
        cai_summary = cai_df.groupby('活躍分群').agg(
            客戶數=('customer_id', 'count'),
            平均CAI=('CAI', 'mean'),
            平均消費金額=('總消費金額', 'mean'),
            平均購買次數=('購買次數', 'mean'),
        ).reindex(seg_order).round(2)
        st.dataframe(cai_summary, use_container_width=True)

    # 第三排：CAI vs 購買次數 + 箱型圖
    col_v5, col_v6 = st.columns(2)

    with col_v5:
        st.subheader('CAI vs 購買次數')
        fig_cai_freq = px.scatter(
            cai_df, x='購買次數', y='CAI', color='活躍分群',
            color_discrete_map=seg_colors, opacity=0.5,
            labels={'購買次數': '購買次數', 'CAI': 'CAI（%）'},
            category_orders={'活躍分群': seg_order},
        )
        fig_cai_freq.add_hline(y=0, line_dash='dash', line_color='gray')
        fig_cai_freq.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400)
        st.plotly_chart(fig_cai_freq, use_container_width=True)

    with col_v6:
        st.subheader('各群 CAI 箱型圖')
        fig_box_cai = px.box(
            cai_df, x='活躍分群', y='CAI', color='活躍分群',
            color_discrete_map=seg_colors,
            category_orders={'活躍分群': seg_order},
            labels={'CAI': 'CAI（%）'},
        )
        fig_box_cai.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400, showlegend=False)
        st.plotly_chart(fig_box_cai, use_container_width=True)

    # 顧客明細表
    st.subheader('顧客 CAI 明細（依 CAI 排序）')
    cai_display = cai_df.sort_values('CAI', ascending=False).reset_index(drop=True)
    cai_display.index = cai_display.index + 1
    st.dataframe(cai_display, use_container_width=True, height=400)

# ==================== Tab 4: ANOVA 檢定 ====================
with tab4:
    st.header('ANOVA 檢定：各 RFM 群的消費金額差異')

    # 確保 rfm 已計算（從 tab2 共用）
    analysis_date_a = filtered['order_date'].max() + pd.Timedelta(days=1)
    rfm_a = filtered.groupby('customer_id').agg(
        recency=('order_date', lambda x: (analysis_date_a - x.max()).days),
        frequency=('order_id', 'nunique'),
        monetary=('revenue', 'sum'),
    ).reset_index()

    def score_80_20_a(series, reverse=False):
        p20 = series.quantile(0.20)
        p80 = series.quantile(0.80)
        if reverse:
            return series.apply(lambda x: 3 if x <= p20 else (2 if x <= p80 else 1))
        else:
            return series.apply(lambda x: 1 if x <= p20 else (2 if x <= p80 else 3))

    rfm_a['R'] = score_80_20_a(rfm_a['recency'], reverse=True)
    rfm_a['F'] = score_80_20_a(rfm_a['frequency'], reverse=False)
    rfm_a['M'] = score_80_20_a(rfm_a['monetary'], reverse=False)
    rfm_a['RFM_code'] = rfm_a['R'].astype(str) + rfm_a['F'].astype(str) + rfm_a['M'].astype(str)

    rfm_names_a = {
        '333': '鑽石級忠誠客戶', '332': '高頻高近期中消費', '331': '高頻高近期低消費',
        '323': '高近期中頻高消費', '322': '穩定中堅客戶', '321': '高近期中頻低消費',
        '313': '高近期低頻高消費', '312': '高近期低頻中消費', '311': '近期新客低活躍',
        '233': '中近期高頻高消費', '232': '中近期高頻中消費', '231': '中近期高頻低消費',
        '223': '中近期中頻高消費', '222': '一般普通客戶', '221': '中近期中頻低消費',
        '213': '中近期低頻高消費', '212': '中近期低頻中消費', '211': '中近期低頻低消費',
        '133': '流失風險高頻高消費', '132': '流失風險高頻中消費', '131': '流失風險高頻低消費',
        '123': '流失風險中頻高消費', '122': '流失風險中頻中消費', '121': '流失風險中頻低消費',
        '113': '沉睡高消費客戶', '112': '沉睡中消費客戶', '111': '已流失低價值客戶',
    }
    rfm_a['客戶分群'] = rfm_a['RFM_code'].map(rfm_names_a).fillna('其他')

    groups = [group['monetary'].values for _, group in rfm_a.groupby('RFM_code')]
    groups = [g for g in groups if len(g) >= 2]

    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)

        col_a1, col_a2, col_a3 = st.columns(3)
        col_a1.metric('F 統計量', f'{f_stat:.4f}')
        col_a2.metric('p-value', f'{p_value:.2e}')
        col_a3.metric('結論', '各群有顯著差異' if p_value < 0.05 else '各群無顯著差異')

        if p_value < 0.05:
            st.success(f'p-value = {p_value:.2e} < 0.05，拒絕虛無假設：各 RFM 群的平均消費金額存在顯著差異。')
        else:
            st.warning(f'p-value = {p_value:.2e} >= 0.05，無法拒絕虛無假設：各 RFM 群的平均消費金額無顯著差異。')

        st.subheader('各群消費金額分佈（箱型圖）')
        top_groups = rfm_a['客戶分群'].value_counts().head(15).index.tolist()
        rfm_top = rfm_a[rfm_a['客戶分群'].isin(top_groups)]
        fig_box = px.box(
            rfm_top, x='客戶分群', y='monetary',
            labels={'monetary': '消費金額（$）', '客戶分群': '客戶分群'}, color='客戶分群',
        )
        fig_box.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=450, showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader('各群消費金額統計')
        anova_table = rfm_a.groupby(['RFM_code', '客戶分群']).agg(
            客戶數=('customer_id', 'count'),
            平均消費金額=('monetary', 'mean'),
            中位數消費金額=('monetary', 'median'),
            標準差=('monetary', 'std'),
        ).round(2).reset_index().rename(columns={'RFM_code': 'RFM 編碼'})
        anova_table = anova_table.sort_values('平均消費金額', ascending=False)
        st.dataframe(anova_table, use_container_width=True, height=400)
    else:
        st.warning('篩選後群數不足，無法進行 ANOVA 檢定。')

# ==================== Tab 5: 人口統計分析 ====================
with tab5:
    st.header('人口統計與消費差異分析')

    # 準備顧客層級資料
    cust_spend = filtered.groupby('customer_id').agg(
        總消費金額=('revenue', 'sum'),
        age=('age', 'first'),
        gender=('gender', 'first'),
        loyalty_member=('loyalty_member', 'first'),
    ).reset_index()

    # --- 年齡分析 ---
    st.subheader('年齡 vs 消費金額')
    cust_spend['年齡層'] = pd.cut(
        cust_spend['age'], bins=[0, 25, 35, 45, 55, 65, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'],
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig_age_box = px.box(
            cust_spend.dropna(subset=['年齡層']), x='年齡層', y='總消費金額',
            color='年齡層', labels={'總消費金額': '總消費金額（$）'},
        )
        fig_age_box.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400, showlegend=False)
        st.plotly_chart(fig_age_box, use_container_width=True)

    with col_d2:
        age_summary = cust_spend.dropna(subset=['年齡層']).groupby('年齡層').agg(
            客戶數=('customer_id', 'count'),
            平均消費=('總消費金額', 'mean'),
            中位數消費=('總消費金額', 'median'),
        ).round(1)
        st.dataframe(age_summary, use_container_width=True)

        # 年齡 ANOVA
        age_groups = [g['總消費金額'].values for _, g in cust_spend.dropna(subset=['年齡層']).groupby('年齡層')]
        age_groups = [g for g in age_groups if len(g) >= 2]
        if len(age_groups) >= 2:
            f_age, p_age = stats.f_oneway(*age_groups)
            if p_age < 0.05:
                st.success(f'年齡 ANOVA：F={f_age:.2f}, p={p_age:.2e} → 各年齡層消費有顯著差異')
            else:
                st.info(f'年齡 ANOVA：F={f_age:.2f}, p={p_age:.2e} → 各年齡層消費無顯著差異')

    # --- 性別分析 ---
    st.subheader('性別 vs 消費金額')
    col_d3, col_d4 = st.columns(2)
    with col_d3:
        fig_gender_box = px.box(
            cust_spend.dropna(subset=['gender']), x='gender', y='總消費金額',
            color='gender', labels={'gender': '性別', '總消費金額': '總消費金額（$）'},
        )
        fig_gender_box.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400, showlegend=False)
        st.plotly_chart(fig_gender_box, use_container_width=True)

    with col_d4:
        gender_summary = cust_spend.dropna(subset=['gender']).groupby('gender').agg(
            客戶數=('customer_id', 'count'),
            平均消費=('總消費金額', 'mean'),
            中位數消費=('總消費金額', 'median'),
        ).round(1)
        st.dataframe(gender_summary, use_container_width=True)

        # 性別 t-test
        gender_vals = cust_spend.dropna(subset=['gender'])
        g_male = gender_vals[gender_vals['gender'] == 'Male']['總消費金額'].values
        g_female = gender_vals[gender_vals['gender'] == 'Female']['總消費金額'].values
        if len(g_male) >= 2 and len(g_female) >= 2:
            t_gender, p_gender = stats.ttest_ind(g_male, g_female)
            if p_gender < 0.05:
                st.success(f'性別 t-test：t={t_gender:.2f}, p={p_gender:.2e} → 男女消費有顯著差異')
            else:
                st.info(f'性別 t-test：t={t_gender:.2f}, p={p_gender:.2e} → 男女消費無顯著差異')

    # --- 會員狀態分析 ---
    st.subheader('忠誠會員 vs 消費金額')
    col_d5, col_d6 = st.columns(2)
    cust_spend['會員狀態'] = cust_spend['loyalty_member'].map({1: '會員', 0: '非會員'})

    with col_d5:
        fig_loyalty_box = px.box(
            cust_spend.dropna(subset=['會員狀態']), x='會員狀態', y='總消費金額',
            color='會員狀態', labels={'總消費金額': '總消費金額（$）'},
            color_discrete_map={'會員': '#2ecc71', '非會員': '#e74c3c'},
        )
        fig_loyalty_box.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400, showlegend=False)
        st.plotly_chart(fig_loyalty_box, use_container_width=True)

    with col_d6:
        loyalty_summary = cust_spend.dropna(subset=['會員狀態']).groupby('會員狀態').agg(
            客戶數=('customer_id', 'count'),
            平均消費=('總消費金額', 'mean'),
            中位數消費=('總消費金額', 'median'),
        ).round(1)
        st.dataframe(loyalty_summary, use_container_width=True)

        # 會員 t-test
        g_member = cust_spend[cust_spend['loyalty_member'] == 1]['總消費金額'].values
        g_non = cust_spend[cust_spend['loyalty_member'] == 0]['總消費金額'].values
        if len(g_member) >= 2 and len(g_non) >= 2:
            t_loy, p_loy = stats.ttest_ind(g_member, g_non)
            if p_loy < 0.05:
                st.success(f'會員 t-test：t={t_loy:.2f}, p={p_loy:.2e} → 會員與非會員消費有顯著差異')
            else:
                st.info(f'會員 t-test：t={t_loy:.2f}, p={p_loy:.2e} → 會員與非會員消費無顯著差異')

    # --- 交叉分析：性別 × 會員 ---
    st.subheader('性別 × 會員狀態 交叉分析')
    cust_spend['交叉分群'] = cust_spend['gender'].astype(str) + ' - ' + cust_spend['會員狀態'].astype(str)
    fig_cross = px.box(
        cust_spend.dropna(subset=['gender', '會員狀態']),
        x='交叉分群', y='總消費金額', color='交叉分群',
        labels={'總消費金額': '總消費金額（$）', '交叉分群': ''},
    )
    fig_cross.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400, showlegend=False)
    st.plotly_chart(fig_cross, use_container_width=True)

# ==================== Tab 6: 地圖與會員趨勢 ====================
with tab6:
    # --- 城市品牌銷售地圖 ---
    st.header('各城市品牌銷售地圖')

    # 城市經緯度
    city_coords = {
        'New York': (40.7128, -74.0060),
        'Toronto': (43.6532, -79.3832),
        'London': (51.5074, -0.1278),
        'Paris': (48.8566, 2.3522),
        'Berlin': (52.5200, 13.4050),
        'Sydney': (-33.8688, 151.2093),
        'Melbourne': (-37.8136, 144.9631),
    }

    # 準備資料
    city_brand = filtered.groupby(['city', 'brand']).agg(
        銷售額=('revenue', 'sum'),
        訂單數=('order_id', 'nunique'),
    ).reset_index()
    city_total = filtered.groupby('city')['revenue'].sum().reset_index().rename(columns={'revenue': '城市總銷售額'})
    city_brand = city_brand.merge(city_total, on='city')
    city_brand['佔比'] = (city_brand['銷售額'] / city_brand['城市總銷售額'] * 100).round(1)
    city_brand['lat'] = city_brand['city'].map(lambda c: city_coords.get(c, (0, 0))[0])
    city_brand['lon'] = city_brand['city'].map(lambda c: city_coords.get(c, (0, 0))[1])

    # 品牌選擇器
    brand_list = sorted(filtered['brand'].dropna().unique())
    map_mode = st.radio(
        '地圖模式', ['所有品牌總覽', '選擇特定品牌'],
        horizontal=True, key='map_mode',
    )

    if map_mode == '所有品牌總覽':
        # 各城市總銷售額地圖
        city_agg = filtered.groupby('city').agg(
            銷售額=('revenue', 'sum'),
            訂單數=('order_id', 'nunique'),
            品牌數=('brand', 'nunique'),
        ).reset_index()
        city_agg['lat'] = city_agg['city'].map(lambda c: city_coords.get(c, (0, 0))[0])
        city_agg['lon'] = city_agg['city'].map(lambda c: city_coords.get(c, (0, 0))[1])

        fig_map = px.scatter_geo(
            city_agg, lat='lat', lon='lon',
            size='銷售額', color='銷售額',
            hover_name='city',
            hover_data={'銷售額': ':,.0f', '訂單數': ':,', '品牌數': True, 'lat': False, 'lon': False},
            projection='natural earth',
            color_continuous_scale='YlOrRd',
            size_max=50,
        )
        fig_map.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=550,
                              coloraxis_colorbar=dict(title='銷售額'))
        st.plotly_chart(fig_map, use_container_width=True)

        # 各城市品牌佔比堆疊長條圖
        st.subheader('各城市品牌銷售佔比')
        fig_city_bar = px.bar(
            city_brand, x='city', y='佔比', color='brand',
            labels={'city': '城市', '佔比': '佔比（%）', 'brand': '品牌'},
            barmode='stack',
        )
        fig_city_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400)
        st.plotly_chart(fig_city_bar, use_container_width=True)

    else:
        # 互動：選擇品牌
        selected_brand_map = st.selectbox('選擇品牌', brand_list, key='brand_map_select')

        brand_data = city_brand[city_brand['brand'] == selected_brand_map].copy()

        if brand_data.empty:
            st.warning(f'品牌 "{selected_brand_map}" 在目前篩選條件下無銷售資料。')
        else:
            # 指標
            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric(f'{selected_brand_map} 總銷售額', f'${brand_data["銷售額"].sum():,.0f}')
            col_b2.metric('涵蓋城市數', f'{brand_data["city"].nunique()}')
            col_b3.metric('總訂單數', f'{brand_data["訂單數"].sum():,}')

            # 地圖
            fig_map_brand = px.scatter_geo(
                brand_data, lat='lat', lon='lon',
                size='銷售額', color='city',
                hover_name='city',
                hover_data={'銷售額': ':,.0f', '訂單數': ':,', '佔比': ':.1f', 'lat': False, 'lon': False},
                projection='natural earth',
                size_max=50,
                title=f'{selected_brand_map} — 各城市銷售分佈',
            )
            fig_map_brand.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=550)
            st.plotly_chart(fig_map_brand, use_container_width=True)

            # 該品牌各城市銷售長條圖
            col_bb1, col_bb2 = st.columns(2)
            with col_bb1:
                st.subheader(f'{selected_brand_map} 各城市銷售額')
                brand_sorted = brand_data.sort_values('銷售額', ascending=True)
                fig_brand_bar = px.bar(
                    brand_sorted, x='銷售額', y='city', orientation='h',
                    labels={'銷售額': '銷售額（$）', 'city': '城市'},
                    color='銷售額', color_continuous_scale='Blues',
                )
                fig_brand_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350,
                                            showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_brand_bar, use_container_width=True)

            with col_bb2:
                st.subheader(f'{selected_brand_map} 各城市佔比')
                fig_brand_pie = px.pie(
                    brand_data, values='銷售額', names='city', hole=0.4,
                )
                fig_brand_pie.update_traces(textinfo='percent+label')
                fig_brand_pie.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=350)
                st.plotly_chart(fig_brand_pie, use_container_width=True)

    # --- 會員加入時間趨勢 ---
    st.header('會員加入時間趨勢')

    # 使用完整 customers 資料（不受篩選影響）
    customers_full = customers.copy()
    customers_full['join_date'] = pd.to_datetime(customers_full['join_date'], errors='coerce')
    customers_full = customers_full.dropna(subset=['join_date'])
    customers_full['加入月份'] = customers_full['join_date'].dt.to_period('M')

    # 每月新加入會員數
    monthly_join = customers_full.groupby('加入月份').agg(
        新加入人數=('customer_id', 'count'),
        忠誠會員數=('loyalty_member', 'sum'),
    ).reset_index()
    monthly_join['加入月份'] = monthly_join['加入月份'].astype(str)
    monthly_join['忠誠會員比例'] = (monthly_join['忠誠會員數'] / monthly_join['新加入人數'] * 100).round(1)
    monthly_join['累積會員數'] = monthly_join['新加入人數'].cumsum()

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.subheader('每月新加入會員數')
        fig_join = go.Figure()
        fig_join.add_trace(go.Bar(
            x=monthly_join['加入月份'], y=monthly_join['新加入人數'],
            name='新加入人數', marker_color='#3498db',
        ))
        fig_join.add_trace(go.Bar(
            x=monthly_join['加入月份'], y=monthly_join['忠誠會員數'],
            name='其中忠誠會員', marker_color='#2ecc71',
        ))
        fig_join.update_layout(
            barmode='overlay', margin=dict(t=20, b=20, l=20, r=20), height=400,
            xaxis_tickangle=-45, legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
        )
        st.plotly_chart(fig_join, use_container_width=True)

    with col_m2:
        st.subheader('忠誠會員佔比趨勢')
        fig_ratio = px.line(
            monthly_join, x='加入月份', y='忠誠會員比例',
            labels={'忠誠會員比例': '忠誠會員比例（%）', '加入月份': '月份'},
            markers=True,
        )
        fig_ratio.update_traces(line_color='#e74c3c')
        fig_ratio.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_ratio, use_container_width=True)

    # 累積會員成長
    st.subheader('累積會員成長曲線')
    fig_cum = px.area(
        monthly_join, x='加入月份', y='累積會員數',
        labels={'累積會員數': '累積會員數', '加入月份': '月份'},
    )
    fig_cum.update_traces(fillcolor='rgba(52, 152, 219, 0.3)', line_color='#3498db')
    fig_cum.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350, xaxis_tickangle=-45)
    st.plotly_chart(fig_cum, use_container_width=True)
