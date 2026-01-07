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
# 1. 智能清單獲取函數
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
            return tickers, "✅ 資料來源：Wikipedia (即時更新)", True
        else:
            return STATIC_BACKUP, "⚠️ 資料來源：系統內建 (備用清單)", False

    except Exception as e:
        print(f"⚠️ 自動更新失敗: {e}，切換至備用清單。")
        return STATIC_BACKUP, f"⚠️ 資料來源：系統內建 (連線錯誤: {str(e)[:20]}...)", False

# ==========================================
# 2. 數據獲取函數
# ==========================================
@st.cache_data(ttl=3600)
def get_qqq_ohlc(lookback_years=5):
    """下載 QQQ OHLC 並計算 MA200"""
    start_date = (datetime.now() - timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
    df = yf.download("QQQ", start=start_date, progress=False, auto_adjust=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs('QQQ', level=1, axis=1)
        except:
            df.columns = df.columns.droplevel(1)
            
    # 計算 200MA
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

@st.cache_data(ttl=3600)
def download_market_data(tickers, lookback_years=5):
    """下載成分股數據"""
    start_date = (datetime.now() - timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start_date, interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    
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
    """計算動能"""
    momentum = df.pct_change(lookback_days)
    return momentum

# ==========================================
# 3. 側邊欄與參數
# ==========================================
st.sidebar.header("⚙️ 策略參數設定")
LOOKBACK = st.sidebar.slider("動能週期 (天)", 20, 120, 60, step=1, help="60交易日約等於一季")
TOP_N = st.sidebar.slider("持有檔數 (Top N)", 3, 10, 5)
INITIAL_CASH = st.sidebar.number_input("初始資金 ($)", 10000, 1000000, 200000)

# 【新增功能】讓使用者決定是否要開啟濾網
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ 風險控管")
USE_MARKET_FILTER = st.sidebar.checkbox(
    "啟用 QQQ 200MA 濾網", 
    value=False, 
    help="勾選後：當 QQQ 跌破 200MA 時強制空手 (持有現金)。\n取消勾選：無論牛熊市，永遠持有最強的 5 支股票 (獲利可能較高，但回撤也較大)。"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 APP 使用指南")
st.sidebar.info(
    """
    **1. 🚦 市場狀態**
    * 參考最上方 K 線圖。
    * 橘線為 200日均線。
    
    **2. 🏆 策略執行**
    * **濾網開啟**：熊市空手，牛市選股。
    * **濾網關閉**：永遠持有 Top 5 強勢股。
    """
)

# ==========================================
# 4. 主畫面邏輯
# ==========================================
st.title("🚀 Nasdaq 100 動能輪動戰情室")

try:
    # 1. 獲取清單與來源
    current_tickers, source_msg, is_live = get_latest_components()
    
    # 2. 數據下載
    with st.spinner(f'正在下載數據 (近5年)...'):
        df_qqq = get_qqq_ohlc() 
        df_close = download_market_data(current_tickers)
        
    st.toast(f'已載入 {len(current_tickers)} 支最新成分股', icon="✅")

    # 3. 計算動能
    momentum = calculate_metrics(df_close, LOOKBACK)
    
    # 4. 判斷牛熊 (目前狀態)
    curr_qqq_price = float(df_qqq['Close'].iloc[-1])
    curr_ma200 = float(df_qqq['MA200'].iloc[-1])
    is_bull_market = curr_qqq_price > curr_ma200

    # --- A. QQQ K線圖 ---
    st.subheader("🚦 市場趨勢判讀 (QQQ vs 200MA)")
    if is_live:
        st.caption(source_msg)
    else:
        st.warning(source_msg)

    fig_qqq = go.Figure()
    fig_qqq.add_trace(go.Candlestick(
        x=df_qqq.index,
        open=df_qqq['Open'],
        high=df_qqq['High'],
        low=df_qqq['Low'],
        close=df_qqq['Close'],
        name='QQQ Price'
    ))
    fig_qqq.add_trace(go.Scatter(
        x=df_qqq.index, 
        y=df_qqq['MA200'], 
        mode='lines', 
        name='200 MA',
        line=dict(color='orange', width=2)
    ))
    fig_qqq.update_layout(
        title=f'QQQ 趨勢圖 (目前狀態: {"🐂 牛市" if is_bull_market else "🐻 熊市"})',
        yaxis_title='Price', template='plotly_dark', xaxis_rangeslider_visible=False, height=500
    )
    st.plotly_chart(fig_qqq, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        prev_price = float(df_qqq['Close'].iloc[-2])
        st.metric("QQQ 現價", f"${curr_qqq_price:.2f}", f"{(curr_qqq_price/prev_price-1)*100:.2f}%")
    with col2:
        dist_ma = (curr_qqq_price - curr_ma200) / curr_ma200 * 100
        st.metric("乖離率 (距200MA)", f"{dist_ma:.2f}%", delta_color="normal" if is_bull_market else "inverse")
    with col3:
        last_rebalance = df_close.resample('ME').last().index[-1]
        st.metric("最近一次換股日", last_rebalance.strftime('%Y-%m-%d'))

    st.divider()

    # --- B. 核心訊號 ---
    st.subheader(f"🏆 本月最強 Top {TOP_N} (即時運算)")
    
    # 【新增邏輯】根據使用者的開關來決定顯示什麼警語
    if USE_MARKET_FILTER and (not is_bull_market):
        st.error("🛑 **熊市保護啟動中 (QQQ < 200MA)**\n\n您已啟用風控濾網，系統建議：**100% 持有現金**。")
        show_picks = False
    else:
        if not is_bull_market:
            st.warning("⚠️ **注意：目前 QQQ < 200MA，但您選擇「關閉濾網」**。請小心操作，下方顯示強勢股名單：")
        show_picks = True
    
    # 始終計算名單，但根據 show_picks 決定是否建議買入
    latest_mom = momentum.iloc[-1].drop('QQQ', errors='ignore')
    latest_mom = latest_mom.sort_values(ascending=False)
    latest_mom = latest_mom[latest_mom > -100] 
    top_picks = latest_mom.head(TOP_N)
    
    if show_picks:
        cols = st.columns(TOP_N)
        for i, (ticker, mom_val) in enumerate(top_picks.items()):
            if ticker in df_close.columns:
                current_price = df_close[ticker].iloc[-1]
                with cols[i]:
                    st.success(f"#{i+1} {ticker}")
                    st.metric("現價", f"${current_price:.2f}")
                    st.metric(f"{LOOKBACK}天漲幅", f"{mom_val*100:.1f}%")
            
    with st.expander("查看完整排名列表 (Top 20)"):
        top_20_tickers = latest_mom.head(20).index
        top_20_df = pd.DataFrame({
            'Price': df_close[top_20_tickers].iloc[-1],
            'Momentum': latest_mom.head(20)
        })
        top_20_df['Momentum %'] = (top_20_df['Momentum'] * 100).map('{:.2f}%'.format)
        top_20_df['Price'] = top_20_df['Price'].map('${:.2f}'.format)
        st.dataframe(top_20_df[['Price', 'Momentum %']], use_container_width=True)

    # --- C. 回測與驗證 ---
    st.divider()
    st.subheader("📈 策略驗證與回測 (Live Backtest)")
    
    if st.button(f"▶️ 執行回測 (濾網狀態: {'開啟' if USE_MARKET_FILTER else '關閉'})"):
        rebalance_dates = df_close.resample('ME').last().index
        equity = [INITIAL_CASH]; cash = INITIAL_CASH; holdings = {}
        history_records = []
        
        bt_df = df_close.copy()
        start_idx = bt_df.index.searchsorted(rebalance_dates[0])
        if start_idx < LOOKBACK: start_idx = LOOKBACK
        
        progress_bar = st.progress(0)
        total_steps = len(bt_df) - start_idx
        
        for i in range(start_idx, len(bt_df)):
            curr_date = bt_df.index[i]
            
            # 1. 更新淨值
            val = cash
            for t, s in holdings.items():
                if t in bt_df.columns:
                    price = bt_df[t].iloc[i]
                    if not pd.isna(price): val += s * price
            
            # 2. 換股日
            if curr_date in rebalance_dates:
                try:
                    # 取得回測當下時間點的 QQQ 狀態
                    hist_qqq_price = df_qqq['Close'].asof(curr_date)
                    hist_qqq_ma = df_qqq['MA200'].asof(curr_date)
                    is_bull = hist_qqq_price > hist_qqq_ma
                    
                    picks = []
                    
                    # 【核心修改】使用者開關邏輯
                    # 如果 (不使用濾網) 或 (是牛市) -> 選股
                    if (not USE_MARKET_FILTER) or is_bull:
                        scores = momentum.iloc[i-1].drop('QQQ', errors='ignore')
                        scores = scores[scores > 0] 
                        picks = scores.sort_values(ascending=False).head(TOP_N).index.tolist()
                    else:
                        # 使用濾網 且 熊市 -> 空手
                        picks = []
                    
                    history_records.append({
                        'Date': curr_date.strftime('%Y-%m-%d'), 
                        'Stocks': picks if picks else ['CASH (Bear Market)']
                    })
                    
                    # 執行換倉
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
                        
                except Exception as e:
                    pass
            
            equity.append(val)
            if i % 50 == 0: progress_bar.progress((i - start_idx) / total_steps)
            
        progress_bar.empty()
        
        # 繪圖
        bt_dates = bt_df.index[start_idx-1:]
        perf_series = pd.Series(equity, index=bt_dates)
        bench = bt_df['QQQ'][start_idx-1:]
        bench = bench / bench.iloc[0] * INITIAL_CASH
        
        pct_return = (perf_series - INITIAL_CASH) / INITIAL_CASH * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=perf_series.index, 
            y=perf_series, 
            mode='lines', 
            name=f'Strategy (Filter: {USE_MARKET_FILTER})', 
            line=dict(color='#00E676', width=2),
            customdata=pct_return,
            hovertemplate='<b>Date</b>: %{x}<br><b>Equity</b>: $%{y:,.0f}<br><b>Return</b>: %{customdata:.2f}%<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=bench.index, 
            y=bench, 
            mode='lines', 
            name='QQQ Benchmark', 
            line=dict(color='gray', dash='dash'),
            hovertemplate='<b>Date</b>: %{x}<br><b>Equity</b>: $%{y:,.0f}<extra></extra>'
        ))
        fig.update_layout(title='資金淨值曲線', template='plotly_dark', height=400, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"策略總報酬: {(equity[-1]/INITIAL_CASH - 1)*100:.2f}%")
        
        hist_df = pd.DataFrame(history_records)
        if not hist_df.empty:
            hist_df['Top Picks'] = hist_df['Stocks'].apply(lambda x: ", ".join(x))
            st.dataframe(hist_df[['Date', 'Top Picks']].sort_values('Date', ascending=False), use_container_width=True)
            
            heatmap_data = []
            for rec in history_records:
                for stock in rec['Stocks']:
                    if stock != 'CASH (Bear Market)':
                        heatmap_data.append({'Date': rec['Date'], 'Stock': stock, 'Held': 1})
            
            if heatmap_data:
                hm_df = pd.DataFrame(heatmap_data)
                fig_hm = px.scatter(hm_df, x="Date", y="Stock", color="Stock", title="動能輪動軌跡 (空白處即為持有現金)", height=600)
                fig_hm.update_traces(marker=dict(size=10, symbol='square'))
                fig_hm.update_layout(showlegend=False, template='plotly_dark')
                st.plotly_chart(fig_hm, use_container_width=True)

except Exception as e:
    st.error(f"系統發生錯誤: {e}")
