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
# 2. 數據獲取函數 (支援動態年份)
# ==========================================
@st.cache_data(ttl=3600)
def get_qqq_ohlc(years):
    """下載 QQQ OHLC 並計算 MA200"""
    start_date = (datetime.now() - timedelta(days=years*365 + 200)).strftime('%Y-%m-%d') # 多抓200天算均線
    df = yf.download("QQQ", start=start_date, progress=False, auto_adjust=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs('QQQ', level=1, axis=1)
        except:
            df.columns = df.columns.droplevel(1)
            
    # 計算 200MA
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    # 裁切回使用者要的年份 (去掉暖身期)
    cut_date = (datetime.now() - timedelta(days=years*365))
    df = df[df.index >= cut_date]
    return df

@st.cache_data(ttl=3600)
def download_market_data(tickers, years):
    """下載成分股數據"""
    start_date = (datetime.now() - timedelta(days=years*365 + 60)).strftime('%Y-%m-%d') # 多抓60天算動能
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
    
    # 裁切回使用者要的年份
    cut_date = (datetime.now() - timedelta(days=years*365))
    df_close = df_close[df_close.index >= cut_date]
    
    return df_close

def calculate_metrics(df, lookback_days):
    """計算動能"""
    momentum = df.pct_change(lookback_days)
    return momentum

# ==========================================
# 3. 側邊欄與參數
# ==========================================
st.sidebar.header("⚙️ 策略參數設定")

# 【新增】回測年份選擇
BACKTEST_YEARS = st.sidebar.number_input("回測歷史長度 (年)", min_value=1, max_value=15, value=10, step=1, help="設定數據回溯的年份，預設為10年以涵蓋多空循環。")

LOOKBACK = st.sidebar.slider("動能週期 (天)", 20, 120, 60, step=1, help="60交易日約等於一季")
TOP_N = st.sidebar.slider("持有檔數 (Top N)", 3, 10, 5)
INITIAL_CASH = st.sidebar.number_input("初始資金 ($)", 10000, 1000000, 200000)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ 風險控管")
USE_MARKET_FILTER = st.sidebar.checkbox(
    "啟用 QQQ 200MA 濾網", 
    value=False, 
    help="勾選後：當 QQQ 跌破 200MA 時強制空手 (持有現金)。\n取消勾選：無論牛熊市，永遠持有最強的 5 支股票。"
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"""
    **APP 使用指南**
    1. **數據範圍**：目前載入近 **{BACKTEST_YEARS}** 年數據。
    2. **市場狀態**：參考上方 K 線圖與 200MA。
    3. **策略回測**：點擊下方按鈕，比較「動能策略」vs「持有大盤」vs「持有現金」的績效。
    """
)

# ==========================================
# 4. 主畫面邏輯
# ==========================================
st.title("🚀 Nasdaq 100 動能輪動戰情室")

try:
    # 1. 獲取清單
    current_tickers, source_msg, is_live = get_latest_components()
    
    # 2. 數據下載 (傳入年份參數)
    with st.spinner(f'正在下載數據 (近 {BACKTEST_YEARS} 年)...'):
        df_qqq = get_qqq_ohlc(BACKTEST_YEARS) 
        df_close = download_market_data(current_tickers, BACKTEST_YEARS)
        
    st.toast(f'已載入 {len(current_tickers)} 支成分股數據', icon="✅")

    # 3. 計算動能
    momentum = calculate_metrics(df_close, LOOKBACK)
    
    # 4. 判斷目前市場狀態
    curr_qqq_price = float(df_qqq['Close'].iloc[-1])
    curr_ma200 = float(df_qqq['MA200'].iloc[-1]) if not pd.isna(df_qqq['MA200'].iloc[-1]) else curr_qqq_price
    is_bull_market = curr_qqq_price > curr_ma200

    # --- A. QQQ K線圖 ---
    st.subheader(f"🚦 市場趨勢判讀 (QQQ vs 200MA)")
    if is_live:
        st.caption(source_msg)
    else:
        st.warning(source_msg)

    fig_qqq = go.Figure()
    fig_qqq.add_trace(go.Candlestick(
        x=df_qqq.index,
        open=df_qqq['Open'], high=df_qqq['High'], low=df_qqq['Low'], close=df_qqq['Close'],
        name='QQQ Price'
    ))
    fig_qqq.add_trace(go.Scatter(
        x=df_qqq.index, y=df_qqq['MA200'], mode='lines', name='200 MA',
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
    
    if USE_MARKET_FILTER and (not is_bull_market):
        st.error("🛑 **熊市保護啟動中 (QQQ < 200MA)**\n\n風控濾網已啟用，系統建議：**100% 持有現金**。")
        show_picks = False
    else:
        if not is_bull_market:
            st.warning("⚠️ **注意：目前 QQQ < 200MA**，但濾網未開啟。")
        show_picks = True
    
    latest_mom = momentum.iloc[-1].drop('QQQ', errors='ignore').sort_values(ascending=False)
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
    st.subheader(f"📈 策略驗證與三方對決 (近 {BACKTEST_YEARS} 年)")
    
    if st.button(f"▶️ 執行回測 (濾網: {'開啟' if USE_MARKET_FILTER else '關閉'})"):
        rebalance_dates = df_close.resample('ME').last().index
        
        # 初始化三組資金
        cash = INITIAL_CASH; equity = [INITIAL_CASH]; holdings = {} # 策略
        
        history_records = []
        
        bt_df = df_close.copy()
        start_idx = bt_df.index.searchsorted(rebalance_dates[0])
        if start_idx < LOOKBACK: start_idx = LOOKBACK
        
        progress_bar = st.progress(0)
        total_steps = len(bt_df) - start_idx
        
        for i in range(start_idx, len(bt_df)):
            curr_date = bt_df.index[i]
            
            # 1. 策略更新淨值
            val = cash
            for t, s in holdings.items():
                if t in bt_df.columns:
                    price = bt_df[t].iloc[i]
                    if not pd.isna(price): val += s * price
            
            # 2. 換股日
            if curr_date in rebalance_dates:
                try:
                    # QQQ 狀態檢查
                    # 使用 asof 避免日期對不齊 (QQQ 數據與個股數據)
                    # 這裡需注意：get_qqq_ohlc 回傳的 df_qqq 索引已是 datetime
                    hist_qqq_price = df_qqq['Close'].asof(curr_date)
                    hist_qqq_ma = df_qqq['MA200'].asof(curr_date)
                    
                    is_bull = False
                    if not pd.isna(hist_qqq_price) and not pd.isna(hist_qqq_ma):
                         is_bull = hist_qqq_price > hist_qqq_ma
                    else:
                        is_bull = True # 缺資料時預設通過
                    
                    picks = []
                    
                    # 決策邏輯
                    if (not USE_MARKET_FILTER) or is_bull:
                        scores = momentum.iloc[i-1].drop('QQQ', errors='ignore')
                        scores = scores[scores > 0] 
                        picks = scores.sort_values(ascending=False).head(TOP_N).index.tolist()
                    else:
                        picks = [] # 熊市空手
                    
                    history_records.append({
                        'Date': curr_date.strftime('%Y-%m-%d'), 
                        'Stocks': picks if picks else ['CASH (Bear Market)']
                    })
                    
                    # 賣出變現
                    pool = cash
                    for t, s in holdings.items():
                        pool += s * bt_df[t].iloc[i] * 0.999 # 簡易交易成本
                    
                    # 重新買入
                    # 【關鍵修正】現金先繼承 pool，再扣成本
                    cash = pool; holdings = {}
                    
                    if len(picks) > 0:
                        size = pool / len(picks)
                        for t in picks:
                            price_buy = bt_df[t].iloc[i]
                            if not pd.isna(price_buy) and price_buy > 0:
                                shares = size / price_buy
                                holdings[t] = shares
                                # 這裡假設 size 已經包含成本，或者在淨值計算時扣除
                                # 為了與 Colab 邏輯一致，這裡簡化處理：
                                # 實際上 cash 已經是 0 (因為全投入)
                        cash = 0
                    else:
                        cash = pool # 空手，全現金
                        
                except Exception as e:
                    pass
            
            equity.append(val)
            if i % 50 == 0: progress_bar.progress((i - start_idx) / total_steps)
            
        progress_bar.empty()
        
        # 準備數據繪圖
        bt_dates = bt_df.index[start_idx-1:]
        
        # 1. 策略曲線
        perf_series = pd.Series(equity, index=bt_dates)
        
        # 2. QQQ 基準曲線 (Buy & Hold)
        bench = bt_df['QQQ'][start_idx-1:]
        bench = bench / bench.iloc[0] * INITIAL_CASH
        
        # 3. 現金基準曲線 (Cash)
        cash_curve = pd.Series([INITIAL_CASH] * len(bt_dates), index=bt_dates)
        
        # 計算回報率 (Hover用)
        pct_return_strat = (perf_series - INITIAL_CASH) / INITIAL_CASH * 100
        pct_return_bench = (bench - INITIAL_CASH) / INITIAL_CASH * 100
        
        # 績效摘要
        ret_s = (perf_series.iloc[-1]/INITIAL_CASH-1)*100
        ret_q = (bench.iloc[-1]/INITIAL_CASH-1)*100
        
        st.success(f"回測結果：動能策略 {ret_s:.2f}% vs QQQ {ret_q:.2f}% vs 現金 0.00%")
        
        fig = go.Figure()
        
        # 策略
        fig.add_trace(go.Scatter(
            x=perf_series.index, y=perf_series, mode='lines', name=f'Strategy',
            line=dict(color='#00E676', width=2),
            customdata=pct_return_strat,
            hovertemplate='<b>Strategy</b>: $%{y:,.0f} (+%{customdata:.1f}%)<extra></extra>'
        ))
        
        # QQQ
        fig.add_trace(go.Scatter(
            x=bench.index, y=bench, mode='lines', name='QQQ (Buy & Hold)',
            line=dict(color='#2962FF', width=2),
            customdata=pct_return_bench,
            hovertemplate='<b>QQQ</b>: $%{y:,.0f} (+%{customdata:.1f}%)<extra></extra>'
        ))
        
        # Cash
        fig.add_trace(go.Scatter(
            x=cash_curve.index, y=cash_curve, mode='lines', name='Cash (Risk Free)',
            line=dict(color='gray', dash='dash'),
            hovertemplate='<b>Cash</b>: $%{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(title=f'{BACKTEST_YEARS}年 資產增長競賽', template='plotly_dark', height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # 熱力圖
        hist_df = pd.DataFrame(history_records)
        if not hist_df.empty:
            heatmap_data = []
            for rec in history_records:
                for stock in rec['Stocks']:
                    if stock != 'CASH (Bear Market)':
                        heatmap_data.append({'Date': rec['Date'], 'Stock': stock, 'Held': 1})
            
            if heatmap_data:
                hm_df = pd.DataFrame(heatmap_data)
                fig_hm = px.scatter(hm_df, x="Date", y="Stock", color="Stock", title="動能輪動軌跡", height=500)
                fig_hm.update_traces(marker=dict(size=10, symbol='square'))
                fig_hm.update_layout(showlegend=False, template='plotly_dark')
                st.plotly_chart(fig_hm, use_container_width=True)

except Exception as e:
    st.error(f"系統發生錯誤: {e}")
