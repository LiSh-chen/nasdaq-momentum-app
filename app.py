import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests # 新增這行
from datetime import datetime, timedelta

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(
    page_title="Nasdaq 100 動能輪動戰情室",
    page_icon="🚀",
    layout="wide"
)

# 內建備用清單 (萬一爬蟲掛掉時的保險)
STATIC_BACKUP = [
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
# 1. 智能清單獲取函數 (自動更新)
# ==========================================
@st.cache_data(ttl=86400) # 設定快取：每 24 小時 (86400秒) 才重新爬一次，其他時間直接用
def get_latest_components():
    """
    自動抓取 Nasdaq 100 最新成分股
    """
    tickers = []
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        # 偽裝成 Chrome 瀏覽器，繞過 403 Forbidden
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # 發送請求
        r = requests.get(url, headers=headers)
        r.raise_for_status() # 檢查是否連線成功
        
        # 讀取表格
        tables = pd.read_html(r.text)
        
        # 尋找包含 Ticker 的表格
        target_table = None
        for t in tables:
            if 'Ticker' in t.columns:
                target_table = t
                break
            elif 'Symbol' in t.columns:
                target_table = t
                break
        
        if target_table is not None:
            col = 'Ticker' if 'Ticker' in target_table.columns else 'Symbol'
            tickers = target_table[col].tolist()
            # 處理特殊代碼 (如 BRK.B -> BRK-B)
            tickers = [t.replace('.', '-') for t in tickers]
            # 確保 QQQ 在裡面
            if 'QQQ' not in tickers: tickers.append('QQQ')
            return tickers
        else:
            raise ValueError("找不到表格")

    except Exception as e:
        # 如果爬蟲失敗，靜默切換到備用清單，但可以在 Log 看到
        print(f"⚠️ 自動更新失敗: {e}，切換至靜態備用清單。")
        return STATIC_BACKUP

# ==========================================
# 2. 獲取數據主函數
# ==========================================
@st.cache_data(ttl=3600)
def get_data(lookback_years=3):
    """下載過去 N 年的數據"""
    
    # 【關鍵】這裡不再用靜態變數，而是呼叫上面的自動更新函數
    current_tickers = get_latest_components()
    
    start_date = (datetime.now() - timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
    
    # 顯示目前抓到了幾支股票，讓使用者知道
    st.toast(f'已載入 {len(current_tickers)} 支最新成分股', icon="✅")
    
    with st.spinner(f'正在下載 {len(current_tickers)} 支成分股數據...請稍候 ☕'):
        data = yf.download(current_tickers, start=start_date, interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    
    df_close = pd.DataFrame()
    for t in current_tickers:
        try:
            if t in data.columns.levels[0]:
                series = data[t]['Close']
                if len(series.dropna()) > 200:
                    df_close[t] = series
        except:
            pass
    
    df_close = df_close.fillna(method='ffill').dropna(how='all')
    df_close.index = pd.to_datetime(df_close.index).tz_localize(None)
    return df_close

# ... (以下程式碼保持不變，直接從 def calculate_metrics 開始接續) ...
