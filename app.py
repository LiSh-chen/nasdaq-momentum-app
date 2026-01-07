import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime, timedelta

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(
    page_title="Nasdaq 100 動能輪動戰情室",
    page_icon="🚀",
    layout="wide"
)

# 內建備用清單
STATIC_BACKUP = [
    'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'ANSS', 'APP',
    'ASML', 'AVGO', 'AXON', 'AZN', 'BIIB', 'BKNG', 'BKR', 'CCEP', 'CDNS', 'CDW', 'CEG', 'CHTR', 'CMCSA',
    'COST', 'CPRT', 'CRWD', 'CSCO', 'CSX', 'CTAS', 'CTSH', 'DDOG', 'DLTR', 'DXCM', 'EA', 'EXC', 'FANG',
    'FAST', 'FTNT', 'GEHC', 'GFS', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'ILMN', 'INTC', 'INTU', 'ISRG',
    'KDP', 'KHC', 'KLAC', 'LIN', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDLZ', 'MELI', 'META', 'MNST', 'MRNA',
    'MRVL', 'MSFT', 'MU', 'NFLX', 'NVDA', 'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD',
    'PEP', 'PYPL', 'QCOM', 'REGN', 'ROP', 'ROST', 'SBUX', 'SIRI', 'SNPS', 'TEAM', 'TMUS', 'TSLA', 'TTD',
    'TXN', 'VRSK', 'VRTX', 'WBA', 'WBD', 'WDAY', 'XEL', 'ZS', 'QQQ', 
    'WDC', 'STX', 'ARM', 'SMCI'
]

# ==========================================
# 1. 智能清單獲取函數 (自動更新)
# ==========================================
@st.cache_data(ttl=86400)
def get_latest_components():
    """自動抓取 Nasdaq 100 最新成分股"""
    tickers = []
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        r = requests.get(url, headers=headers, timeout=5)
        r.raise_for_status()
        
        tables = pd.read_html(r.text)
        target_table = None
        for t in tables:
            if 'Ticker' in t.columns:
                target_table = t; break
            elif 'Symbol' in t.columns:
                target_table = t; break
        
        if target_table is not None:
            col = 'Ticker' if 'Ticker' in target_table.columns else 'Symbol'
            tickers = target_table[col].tolist()
            tickers = [t.replace('.', '-') for t in tickers]
            if 'QQQ' not in tickers: tickers.append('QQQ')
            return tickers
        else:
            return STATIC_BACKUP

    except Exception as e:
        print(f"⚠️ 自動更新失敗: {e}，切換至備用清單。")
        return STATIC_BACKUP

# ==========================================
# 2. 獲取數據主函數 (修正快取衝突)
# ==========================================
@st.cache_data(ttl=3600)
def download_market_data(tickers, lookback_years=3):
    """
    純粹的數據下載與清洗邏輯
    注意：這裡不能放 st.spinner 或 st.toast
    """
    start_date = (datetime.now() - timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
    
    # 下載數據
    data = yf.download(tickers, start=start_date, interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    
    # --- 數據清洗核心邏輯 ---
    df_close = pd.DataFrame()
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
            df_close = data.xs('Close', level=1, axis=1)
        except KeyError:
            try:
                df_close = data.xs('Close', level=0, axis=1)
            except KeyError:
                for t in tickers:
                    if t in data.columns:
                        df_close[t] = data[t]['Close']
    else:
        if 'Close' in data.columns:
            df_close = data['Close']
        else:
            for t in tickers:
                 if t in data.columns:
                     df_close[t] = data[t]

    df_close = df_close.fillna(method='ffill').dropna(how='all')
    df_close.index = pd.to_datetime(df_close.index).tz_localize(None)
    
    return df_close

def calculate_metrics(df, lookback_days):
    """計算動能與指標"""
    momentum = df.pct_change(lookback_days)
    
    qqq_close = df['QQQ']
    qqq_ma200 = qqq_close.rolling(window=200).mean()
    market_trend = qqq_close.iloc[-1] > qqq_ma200.iloc[-1]
    
    return momentum, market_trend, qqq_close, qqq_ma200

# ==========================================
# 3. 側邊欄與參數 (更新使用說明)
# ==========================================
st.sidebar.header("⚙️ 策略參數設定")

# 修正：預設值改為 60
LOOKBACK = st.sidebar.slider("動能週期 (天)", 20, 120, 60, step=1, help="60交易日約等於一季")
TOP_N = st.sidebar.slider("持有檔數 (Top N)", 3, 10, 5)
INITIAL_CASH = st.sidebar.number_input("初始資金 ($)", 10000, 1000000, 200000)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 APP 使用指南")
st.sidebar.info(
    """
    **1. 🚦 檢查市場狀態 (最上方)**
    * **牛市 (Bull)**：QQQ 在 200日均線之上，**可積極進場**。
    * **熊市 (Bear)**：QQQ 跌破 200日均線，建議**清空持股**，轉持有現金或美債 (如 BIL/SHV)。
    
    **2. 🏆 每月換股 (Top Picks)**
    * 本策略每月調整一次持倉。
    * 請參考 **「本月最強 Top 5」** 卡片。
    * 買入這 5 支股票，並持有到下個月底。
    
    **3. 🔄 汰弱留強**
    * 下個月底打開此 APP，若名單變動，則賣出舊的、買入新的。
    * 若市場轉為「熊市」，則無條件賣出所有股票。
    """
)
st.sidebar.caption(f"系統每日自動從 Wiki 更新成分股清單")

# ==========================================
# 4. 主畫面邏輯 (UI 邏輯移到這裡)
# ==========================================
st.title("🚀 Nasdaq 100 動能輪動戰情室")

try:
    # 1. 先獲取清單
    current_tickers = get_latest_components()
    
    # 2. 顯示載入動畫 (移到 cache 函數外面)
    with st.spinner(f'正在下載 {len(current_tickers)} 支成分股數據...'):
        df = download_market_data(current_tickers)
        
    # 3. 顯示成功訊息 (移到 cache 函數外面)
    st.toast(f'已載入 {len(current_tickers)} 支最新成分股', icon="✅")

    momentum, is_bull_market, qqq, ma200 = calculate_metrics(df, LOOKBACK)
    
    # --- A. 市場紅綠燈 ---
    col1, col2, col3 = st.columns(3)
    current_qqq = qqq.iloc[-1]
    current_ma = ma200.iloc[-1]
    
    with col1:
        st.metric("QQQ 現價", f"${current_qqq:.2f}", f"{(current_qqq/qqq.iloc[-2]-1)*100:.2f}%")
    
    with col2:
        ma_delta = current_qqq - current_ma
        status_text = "🐂 牛市" if is_bull_market else "🐻 熊市"
        delta_color = "normal" if is_bull_market else "inverse"
        st.metric("市場狀態 (vs 200MA)", status_text, f"{ma_delta:.2f} 點", delta_color=delta_color)
        
    with col3:
        last_rebalance = df.resample('ME').last().index[-1]
        st.metric("最近一次換股日", last_rebalance.strftime('%Y-%m-%d'))

    st.divider()

    # --- B. 核心訊號 ---
    st.subheader(f"🏆 本月最強 Top {TOP_N} (即時運算)")
    
    if not is_bull_market:
        st.error("🛑 **目前處於熊市保護模式 (QQQ < 200MA)！**\n\n策略建議：**100% 持有現金** 或 **短債ETF (BIL)**，暫停買入任何股票。")
    
    # 確保只取最新的數據，且去除 QQQ
    latest_mom = momentum.iloc[-1].drop('QQQ', errors='ignore')
    latest_mom = latest_mom.sort_values(ascending=False)
    
    # 簡單濾網：只顯示正報酬
    latest_mom = latest_mom[latest_mom > -100] 
    
    top_picks = latest_mom.head(TOP_N)
    
    cols = st.columns(TOP_N)
    for i, (ticker, mom_val) in enumerate(top_picks.items()):
        if ticker in df.columns:
            current_price = df[ticker].iloc[-1]
            with cols[i]:
                st.success(f"#{i+1} {ticker}")
                st.metric("現價", f"${current_price:.2f}")
                st.metric(f"{LOOKBACK}天漲幅", f"{mom_val*100:.1f}%")
            
    with st.expander("查看完整排名列表 (Top 20)"):
        top_20_tickers = latest_mom.head(20).index
        top_20_df = pd.DataFrame({
            'Price': df[top_20_tickers].iloc[-1],
            'Momentum': latest_mom.head(20)
        })
        top_20_df['Momentum %'] = (top_20_df['Momentum'] * 100).map('{:.2f}%'.format)
        top_20_df['Price'] = top_20_df['Price'].map('${:.2f}'.format)
        st.dataframe(top_20_df[['Price', 'Momentum %']], use_container_width=True)

    # --- C. 回測與驗證圖表 ---
    st.divider()
    st.subheader("📈 策略驗證與回測 (Live Backtest)")
    
    if st.button("▶️ 執行回測與驗證"):
        # 回測引擎
        rebalance_dates = df.resample('ME').last().index
        equity = [INITIAL_CASH]; cash = INITIAL_CASH; holdings = {}
        history_records = []
        
        bt_df = df.copy()
        start_idx = bt_df.index.searchsorted(rebalance_dates[0])
        if start_idx < LOOKBACK: start_idx = LOOKBACK
        
        progress_bar = st.progress(0)
        total_steps = len(bt_df) - start_idx
        
        for i in range(start_idx, len(bt_df)):
            curr_date = bt_df.index[i]
            
            val = cash
            for t, s in holdings.items():
                if t in bt_df.columns:
                    price = bt_df[t].iloc[i]
                    if not pd.isna(price): val += s * price
            
            if curr_date in rebalance_dates:
                try:
                    scores = momentum.iloc[i-1].drop('QQQ', errors='ignore')
                    scores = scores[scores > 0] 
                    picks = scores.sort_values(ascending=False).head(TOP_N).index.tolist()
                    
                    history_records.append({'Date': curr_date.strftime('%Y-%m-%d'), 'Stocks': picks})
                    
                    pool = cash
                    for t, s in holdings.items():
                        pool += s * bt_df[t].iloc[i] * 0.999 
                    
                    cash = 0; holdings = {}
                    if len(picks) > 0:
                        size = pool / len(picks)
                        for t in picks:
                            holdings[t] = size / bt_df[t].iloc[i]
                        cash = 0
                    else:
                        cash = pool 
                except: pass
            
            equity.append(val)
            if i % 50 == 0: progress_bar.progress((i - start_idx) / total_steps)
            
        progress_bar.empty()
        
        bt_dates = bt_df.index[start_idx-1:]
        perf_series = pd.Series(equity, index=bt_dates)
        bench = bt_df['QQQ'][start_idx-1:]
        bench = bench / bench.iloc[0] * INITIAL_CASH
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=perf_series.index, y=perf_series, mode='lines', name='Momentum Strategy', line=dict(color='#00E676', width=2)))
        fig.add_trace(go.Scatter(x=bench.index, y=bench, mode='lines', name='QQQ Benchmark', line=dict(color='gray', dash='dash')))
        fig.update_layout(title='資金淨值曲線', template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"策略總報酬: {(equity[-1]/INITIAL_CASH - 1)*100:.2f}%")
        
        hist_df = pd.DataFrame(history_records)
        if not hist_df.empty:
            hist_df['Top Picks'] = hist_df['Stocks'].apply(lambda x: ", ".join(x))
            st.dataframe(hist_df[['Date', 'Top Picks']].sort_values('Date', ascending=False), use_container_width=True)
            
            heatmap_data = []
            for rec in history_records:
                for stock in rec['Stocks']:
                    heatmap_data.append({'Date': rec['Date'], 'Stock': stock, 'Held': 1})
            hm_df = pd.DataFrame(heatmap_data)
            fig_hm = px.scatter(hm_df, x="Date", y="Stock", color="Stock", title="動能輪動軌跡", height=600)
            fig_hm.update_traces(marker=dict(size=10, symbol='square'))
            fig_hm.update_layout(showlegend=False, template='plotly_dark')
            st.plotly_chart(fig_hm, use_container_width=True)

except Exception as e:
    st.error(f"系統發生錯誤: {e}")
