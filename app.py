import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 0. 頁面設定與全域變數
# ==========================================
st.set_page_config(
    page_title="Nasdaq 100 動能輪動儀表板",
    page_icon="🚀",
    layout="wide"
)

# 內建 Nasdaq 100 完整清單 (含 QQQ)
FULL_NDX_LIST = [
    'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'ANSS', 'APP',
    'ASML', 'AVGO', 'AXON', 'AZN', 'BIIB', 'BKNG', 'BKR', 'CCEP', 'CDNS', 'CDW', 'CEG', 'CHTR', 'CMCSA',
    'COST', 'CPRT', 'CRWD', 'CSCO', 'CSX', 'CTAS', 'CTSH', 'DDOG', 'DLTR', 'DXCM', 'EA', 'EXC', 'FANG',
    'FAST', 'FTNT', 'GEHC', 'GFS', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'ILMN', 'INTC', 'INTU', 'ISRG',
    'KDP', 'KHC', 'KLAC', 'LIN', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDLZ', 'MELI', 'META', 'MNST', 'MRNA',
    'MRVL', 'MSFT', 'MU', 'NFLX', 'NVDA', 'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD',
    'PEP', 'PYPL', 'QCOM', 'REGN', 'ROP', 'ROST', 'SBUX', 'SIRI', 'SNPS', 'TEAM', 'TMUS', 'TSLA', 'TTD',
    'TXN', 'VRSK', 'VRTX', 'WBA', 'WBD', 'WDAY', 'XEL', 'ZS', 'QQQ'
]

# ==========================================
# 1. 核心函數
# ==========================================
@st.cache_data(ttl=3600) # 快取 1 小時，避免重複下載
def get_data(lookback_years=3):
    """下載過去 N 年的數據"""
    start_date = (datetime.now() - timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
    with st.spinner(f'正在下載 {len(FULL_NDX_LIST)} 支成分股數據...請稍候 ☕'):
        data = yf.download(FULL_NDX_LIST, start=start_date, interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    
    df_close = pd.DataFrame()
    for t in FULL_NDX_LIST:
        try:
            if t in data.columns.levels[0]:
                series = data[t]['Close']
                if len(series.dropna()) > 200: # 過濾新股
                    df_close[t] = series
        except:
            pass
    
    df_close = df_close.fillna(method='ffill').dropna(how='all')
    df_close.index = pd.to_datetime(df_close.index).tz_localize(None)
    return df_close

def calculate_metrics(df, lookback_days=60):
    """計算動能與指標"""
    # 1. 動能 (ROC)
    momentum = df.pct_change(lookback_days)
    
    # 2. QQQ 200MA (大盤濾網)
    qqq_close = df['QQQ']
    qqq_ma200 = qqq_close.rolling(window=200).mean()
    market_trend = qqq_close.iloc[-1] > qqq_ma200.iloc[-1]
    
    return momentum, market_trend, qqq_close, qqq_ma200

# ==========================================
# 2. 側邊欄與參數
# ==========================================
st.sidebar.header("⚙️ 策略參數設定")
LOOKBACK = st.sidebar.slider("動能週期 (天)", 20, 120, 60, step=20, help="計算過去多少天的報酬率來排名")
TOP_N = st.sidebar.slider("持有檔數 (Top N)", 3, 10, 5)
INITIAL_CASH = st.sidebar.number_input("初始資金 ($)", 10000, 1000000, 200000)

st.sidebar.markdown("---")
st.sidebar.info("💡 **策略邏輯：**\n1. 每月底檢查\n2. 篩選 Nasdaq 100 成分股\n3. 買入過去一季漲幅最強的 Top 5\n4. 若 QQQ 跌破 200MA 則示警")

# ==========================================
# 3. 主畫面邏輯
# ==========================================
st.title("🚀 Nasdaq 100 動能輪動戰情室")
st.markdown(f"**當前追蹤池：** {len(FULL_NDX_LIST)} 支成分股 | **策略：** 去弱留強 (Momentum Rotation)")

# 獲取數據
try:
    df = get_data()
    momentum, is_bull_market, qqq, ma200 = calculate_metrics(df, LOOKBACK)
    
    # --- A. 市場紅綠燈 (Risk On/Off) ---
    col1, col2, col3 = st.columns(3)
    
    current_qqq = qqq.iloc[-1]
    current_ma = ma200.iloc[-1]
    
    with col1:
        st.metric("QQQ 現價", f"${current_qqq:.2f}", f"{(current_qqq/qqq.iloc[-2]-1)*100:.2f}%")
    
    with col2:
        ma_delta = current_qqq - current_ma
        color = "normal" if ma_delta > 0 else "inverse"
        label = "🐂 牛市 (Price > 200MA)" if is_bull_market else "🐻 熊市 (Price < 200MA)"
        st.metric("市場狀態 (200MA)", label, f"{ma_delta:.2f}", delta_color=color)
        
    with col3:
        last_rebalance = df.resample('ME').last().index[-1]
        next_rebalance = (last_rebalance + timedelta(days=20)).replace(day=1) + timedelta(days=32)
        next_rebalance = next_rebalance.replace(day=1) - timedelta(days=1)
        st.metric("下一次換股日 (月底)", last_rebalance.strftime('%Y-%m-%d'))

    st.divider()

    # --- B. 核心訊號：現在買什麼？ ---
    st.subheader("🏆 本月最強 Top Picks (即時運算)")
    
    if not is_bull_market:
        st.error("⚠️ **警告：大盤 (QQQ) 位於 200MA 之下！建議空手或持有債券 (TLT/BIL)，暫停買入股票。**")
    
    # 取得最新一天的動能排名
    latest_mom = momentum.iloc[-1].drop('QQQ', errors='ignore')
    latest_mom = latest_mom[latest_mom > 0] # 濾除下跌股
    top_picks = latest_mom.sort_values(ascending=False).head(TOP_N)
    
    # 展示 Top N 卡片
    cols = st.columns(TOP_N)
    for i, (ticker, mom_val) in enumerate(top_picks.items()):
        current_price = df[ticker].iloc[-1]
        # 嘗試取得公司名稱 (這裡簡化，實戰可用字典映射)
        with cols[i]:
            st.success(f"#{i+1} {ticker}")
            st.metric("現價", f"${current_price:.2f}")
            st.metric(f"{LOOKBACK}天漲幅", f"{mom_val*100:.1f}%")
            
    # 詳細表格
    with st.expander("查看完整排名列表 (Top 20)"):
        top_20 = latest_mom.sort_values(ascending=False).head(20).to_frame(name='Momentum')
        top_20['Price'] = df[top_20.index].iloc[-1]
        top_20['Momentum %'] = (top_20['Momentum'] * 100).map('{:.2f}%'.format)
        st.dataframe(top_20[['Price', 'Momentum %']], use_container_width=True)

    # --- C. 回測圖表 ---
    st.divider()
    st.subheader("📈 策略歷史績效 (Live Backtest)")
    
    if st.button("▶️ 執行即時回測 (需約 10 秒)"):
        
        # 簡易回測引擎 (與 Colab 邏輯相同)
        rebalance_dates = df.resample('ME').last().index
        equity = [INITIAL_CASH]; cash = INITIAL_CASH; holdings = {}
        
        # 為了速度，簡化繪圖點數
        bt_df = df.copy()
        
        start_idx = bt_df.index.searchsorted(rebalance_dates[0])
        if start_idx < LOOKBACK: start_idx = LOOKBACK
        
        for i in range(start_idx, len(bt_df)):
            curr_date = bt_df.index[i]
            
            # 更新淨值
            val = cash
            for t, s in holdings.items():
                if t in bt_df.columns:
                    price = bt_df[t].iloc[i]
                    if not pd.isna(price): val += s * price
            
            # 換股
            if curr_date in rebalance_dates:
                try:
                    scores = momentum.iloc[i-1].drop('QQQ', errors='ignore')
                    scores = scores[scores > 0] # 動能濾網
                    picks = scores.sort_values(ascending=False).head(TOP_N).index.tolist()
                    
                    # 全賣
                    pool = cash
                    for t, s in holdings.items():
                        pool += s * bt_df[t].iloc[i] * 0.999 # 簡易手續費
                    
                    # 全買
                    cash = 0; holdings = {}
                    if len(picks) > 0:
                        size = pool / len(picks)
                        for t in picks:
                            holdings[t] = size / bt_df[t].iloc[i]
                        cash = 0
                    else:
                        cash = pool # 空手
                except: pass
            
            equity.append(val)
            
        # 繪圖
        bt_dates = bt_df.index[start_idx-1:]
        perf_series = pd.Series(equity, index=bt_dates)
        
        # 基準
        bench = bt_df['QQQ'][start_idx-1:]
        bench = bench / bench.iloc[0] * INITIAL_CASH
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=perf_series.index, y=perf_series, mode='lines', name='Momentum Strategy', line=dict(color='#00E676', width=2)))
        fig.add_trace(go.Scatter(x=bench.index, y=bench, mode='lines', name='QQQ Benchmark', line=dict(color='gray', dash='dash')))
        fig.update_layout(title='策略淨值走勢', template='plotly_dark', height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        total_ret = (equity[-1]/INITIAL_CASH - 1)*100
        st.info(f"回測結果：策略總報酬 **{total_ret:.2f}%** (參數: {LOOKBACK}天動能, 持有 Top {TOP_N})")

except Exception as e:
    st.error(f"數據載入失敗，請稍後重試: {e}")