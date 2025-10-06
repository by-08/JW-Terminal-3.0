import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
from calendar import monthrange
import json
import os
import io
import subprocess
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests  # Added for GitHub download
import hashlib  # For file hash check

# Page config
st.set_page_config(page_title="John Wick Terminal", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for cyberpunk theme
st.markdown("""
<style>
    .main {
        background-color: black;
        color: lime;
        font-family: 'Courier New', monospace;
    }
    .stApp {
        background-color: black;
    }
    .css-1d391kg {
        background-color: black;
    }
    .stSidebar {
        background-color: black;
    }
    [data-testid="stSidebar"] {
        background-color: black;
    }
    .stTextInput > div > div > input {
        background-color: black;
        color: lime;
        border: 1px solid lime;
        border-radius: 0;
    }
    .stNumberInput > div > div > input {
        background-color: black;
        color: lime;
        border: 1px solid lime;
        border-radius: 0;
    }
    .stSelectbox > div > div > select {
        background-color: black;
        color: lime;
        border: 1px solid lime;
        border-radius: 0;
    }
    .stDateInput > div > div > input {
        background-color: black;
        color: lime;
        border: 1px solid lime;
        border-radius: 0;
    }
    .stCheckbox > div {
        color: lime;
    }
    .stButton > button {
        background-color: black;
        color: lime;
        border: 2px solid lime;
        font-family: 'Courier New', monospace;
        font-size: 10px;
        border-radius: 0;
        display: block;
        margin: 0 auto;
    }
    .stButton {
        text-align: center;
    }
    .stButton > button:hover {
        background-color: #00FF00;
        color: black;
        border: 2px solid #00FF00;
    }
    .stDownloadButton > button {
        background-color: black;
        color: lime;
        border: 2px solid lime;
        font-family: 'Courier New', monospace;
        font-size: 10px;
        border-radius: 0;
    }
    .stDownloadButton > button:hover {
        background-color: #00FF00;
        color: black;
        border: 2px solid #00FF00;
    }
    .stDataFrame {
        background-color: black;
        color: lime;
        font-family: 'Courier New', monospace;
        font-size: 9px;
        border-radius: 0;
    }
    .dataframe {
        background-color: black;
        color: lime;
        font-family: 'Courier New', monospace;
        font-size: 9px;
        border-radius: 0;
    }
    .dataframe thead th {
        background-color: darkgreen;
        color: lime;
        font-weight: bold;
        font-size: 10px;
        font-family: 'Courier New', monospace;
        border-radius: 0;
        text-align: center !important;
    }
    .dataframe tbody td {
        background-color: black;
        color: lime;
        border: none;
        text-align: center !important;
    }
    .stTabs {
        background-color: black;
        border-radius: 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: black;
        border-bottom: 1px solid darkgreen;
        border-radius: 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: lime;
        font-family: 'Courier New', monospace;
        font-size: 10px;
        padding: 8px 12px;
        border-radius: 0;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00FF00;
        background-color: black;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: black;
        background-color: darkgreen;
        border: 1px solid darkgreen;
        border-radius: 0;
    }
    .stAlert {
        background-color: black;
        color: lime;
        border: 1px solid lime;
        border-radius: 0;
    }
    .stInfo {
        background-color: black;
        color: lime;
        border: 1px solid lime;
        border-radius: 0;
        font-family: 'Courier New', monospace;
        font-size: 12px;
    }
    .stWarning {
        background-color: black;
        color: #FF6B6B;
        border: 1px solid #FF6B6B;
        border-radius: 0;
        font-family: 'Courier New', monospace;
        font-size: 12px;
    }
    .stSuccess {
        background-color: black;
        color: #51CF66;
        border: 1px solid #51CF66;
        border-radius: 0;
    }
    .stSelectbox > label {
        color: lime;
    }
    .stTextInputs > label {
        color: lime;
    }
    .stNumberInput > label {
        color: lime;
    }
    .stDateInput > label {
        color: lime;
    }
    .css-1aumxhk {
        text-align: center;
    }
    .metric-card {
        background-color: black;
        color: lime;
        border: 1px solid lime;
        padding: 10px;
        text-align: center;
        font-family: 'Courier New', monospace;
    }
    .stNumberInput input[type=number]::-webkit-outer-spin-button,
    .stNumberInput input[type=number]::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    .stNumberInput input[type=number] {
        -moz-appearance: textfield;
    }
</style>
""", unsafe_allow_html=True)

# Title and subheader
st.markdown("<h1 style='text-align: center; color: lime; font-family: \"Courier New\", monospace; font-size: 16px; font-weight: bold; margin-bottom: 0;'>JOHN WICK TERMINAL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: lime; font-family: \"Courier New\", monospace; font-size: 12px; font-style: italic; margin-top: 0;'>Si vis pacem, para bellum.</p>", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_df' not in st.session_state:
    st.session_state.last_df = pd.DataFrame()
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'custom_tickers' not in st.session_state:
    st.session_state.custom_tickers = {}
if 'ticker_lists' not in st.session_state:
    # Load from file if exists
    ticker_lists_file = 'ticker_lists.json'
    if os.path.exists(ticker_lists_file):
        with open(ticker_lists_file, 'r') as f:
            st.session_state.ticker_lists = json.load(f)
    else:
        st.session_state.ticker_lists = {}
if 'selected_ticker_list' not in st.session_state:
    st.session_state.selected_ticker_list = 'Default'
if 'show_settings' not in st.session_state:
    st.session_state.show_settings = False
if 'default_tickers' not in st.session_state:
    st.session_state.default_tickers = []
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'bt_start_date' not in st.session_state:
    st.session_state.bt_start_date = datetime.now().date() - timedelta(days=365)
if 'bt_end_date' not in st.session_state:
    st.session_state.bt_end_date = datetime.now().date()

# Cache files for tickers
@st.cache_data
def load_cached_tickers():
    cache_file = 'tickers_cache.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return []

def save_tickers(tickers):
    cache_file = 'tickers_cache.json'
    with open(cache_file, 'w') as f:
        json.dump(tickers, f)

def push_to_github():
    try:
        subprocess.run(['git', 'add', 'tickers_cache.json'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Update tickers cache'], check=True)
        subprocess.run(['git', 'push'], check=True)
        st.success("Pushed updated tickers_cache to GitHub.")
    except Exception as e:
        st.warning(f"Could not push to GitHub: {e}. Ensure git is configured with credentials.")

def push_to_github_historical():
    parquet_file = 'historical_ohlcv.parquet'
    try:
        # Check if there are changes to commit
        result = subprocess.run(['git', 'diff', '--quiet', parquet_file], capture_output=True)
        if result.returncode == 0:
            st.info("No changes to historical_ohlcv.parquet, skipping push.")
            return
        subprocess.run(['git', 'add', parquet_file], check=True)
        subprocess.run(['git', 'commit', '-m', 'Update historical cache'], check=True)
        subprocess.run(['git', 'push'], check=True)
        st.success("Pushed updated historical_ohlcv.parquet to GitHub.")
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            st.info("No changes to commit for historical_ohlcv.parquet.")
        else:
            st.warning(f"Could not push historical to GitHub: {e}. Ensure git is configured with credentials.")
    except Exception as e:
        st.warning(f"Could not push historical to GitHub: {e}. Ensure git is configured with credentials.")

def save_ticker_lists():
    with open('ticker_lists.json', 'w') as f:
        json.dump(st.session_state.ticker_lists, f)

# Added function to ensure parquet is downloaded from GitHub
def ensure_parquet_from_github():
    local_file = 'historical_ohlcv.parquet'
    if os.path.exists(local_file):
        return  # Already local
    # Replace with your GitHub raw URL
    github_url = "https://raw.githubusercontent.com/yourusername/yourrepo/main/historical_ohlcv.parquet"  # Update this URL
    try:
        r = requests.get(github_url)
        r.raise_for_status()
        with open(local_file, 'wb') as f:
            f.write(r.content)
        st.success("Downloaded historical_ohlcv.parquet from GitHub.")
    except Exception as e:
        st.warning(f"Could not download historical_ohlcv.parquet from GitHub: {e}. Proceeding with local file if available or full yfinance download.")

def safe_float(val):
    if isinstance(val, pd.Series):
        val = val.iloc[0] if not val.empty else np.nan
    if pd.isna(val):
        return float('nan')
    return float(val)

def safe_int(val):
    if isinstance(val, pd.Series):
        val = val.iloc[0] if not val.empty else 0
    if pd.isna(val):
        return 0
    return int(val)

def flatten_new_data(new_data):
    if not new_data:
        return pd.DataFrame()
    rows = []
    for ticker, date_dict in new_data.items():
        for date_str, vals in date_dict.items():
            dt = pd.to_datetime(date_str)
            if dt.date() < datetime.now().date():
                rows.append({
                    'ticker': ticker,
                    'date': dt,
                    'open': vals['O'],
                    'high': vals['H'],
                    'low': vals['L'],
                    'close': vals['C'],
                    'volume': vals['V']
                })
    return pd.DataFrame(rows)

def update_parquet(df):
    if df.empty:
        return
    df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
    parquet_file = 'historical_ohlcv.parquet'
    if os.path.exists(parquet_file):
        existing = pd.read_parquet(parquet_file)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['ticker', 'date'])
        combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)
        if len(combined) == len(existing):
            st.info("No new data added to historical_ohlcv.parquet.")
            return
    else:
        combined = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    combined.to_parquet(parquet_file, index=False, engine='pyarrow')
    push_to_github_historical()

def get_tickers():
    all_tickers = []
    
    # S&P 500 from Wikipedia
    try:
        sp_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        all_tickers.extend(sp_table['Symbol'].tolist())
    except:
        pass
    
    # S&P 500 from stockanalysis.com
    try:
        sp_table = pd.read_html('https://stockanalysis.com/list/sp-500-stocks/')[0]
        all_tickers.extend(sp_table['Symbol'].tolist())
    except:
        pass
    
    # Nasdaq-100 from Wikipedia
    try:
        nasdaq_tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        for table in nasdaq_tables:
            if 'Ticker' in table.columns:
                all_tickers.extend(table['Ticker'].tolist())
                break
    except:
        pass
    
    # Nasdaq-100 from slickcharts
    try:
        tables = pd.read_html('https://www.slickcharts.com/nasdaq100')
        for table in tables:
            if 'Symbol' in table.columns:
                all_tickers.extend(table['Symbol'].tolist())
                break
    except:
        pass
    
    # High volume ETFs from etfdb.com
    try:
        tables = pd.read_html('https://etfdb.com/compare/volume/')
        for table in tables:
            if 'Symbol' in table.columns:
                # Take top 50 high volume ETFs
                top_etfs = table['Symbol'].head(50).tolist()
                all_tickers.extend(top_etfs)
                break
    except:
        # Fallback hardcoded high volume ETFs
        hardcoded_etfs = ['SPY', 'QQQ', 'IWM', 'EEM', 'TLT', 'GLD', 'USO', 'XLF', 'XLE', 'XLI', 'XLB', 'XLY', 'XLP', 'XLU', 'XLK', 'XLV', 'XLI', 'XLRE', 'XLC', 'VXX', 'UVXY', 'TQQQ', 'SQQQ']
        all_tickers.extend(hardcoded_etfs)
    
    # Sector SPDRs hardcoded
    spdr_sectors = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLK']
    all_tickers.extend(spdr_sectors)
    
    all_tickers = list(set(all_tickers))
    all_tickers = [t for t in all_tickers if t not in ['BF.B', 'BRK.B']]
    return all_tickers

def custom_progress(container, value, text):
    text_color = "lime" if value < 0.5 else "black"
    html = f"""
    <div style="
        width: 100%;
        height: 25px;
        background-color: black;
        border: 2px solid lime;
        position: relative;
        border-radius: 0;
        font-family: 'Courier New', monospace;
        font-size: 11px;
    ">
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            background-color: lime;
            width: {int(value * 100)}%;
            transition: width 0.2s ease;
        "></div>
        <span style="
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: {text_color};
            z-index: 1;
            white-space: nowrap;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        ">{text}</span>
    </div>
    """
    container.markdown(html, unsafe_allow_html=True)

def process_mode(mode, mode_progress_start, mode_progress_end, progress_container, tickers, date_str, percentage, start_30d, end_date, min_avg_vol, min_rel_vol, min_rvolat, full_data=None, current_date=None):
    # Adapted for backtest: if full_data provided, use slice instead of download
    if full_data is not None and current_date is not None:
        # For backtest, slice to current_date bar
        all_hist_single = {}
        for ticker in tickers:
            if ticker in full_data.columns.get_level_values(0):
                ticker_data = full_data[ticker]
                # Ensure exact date match, normalize to date
                mask = (ticker_data.index.date == current_date.date())
                single_data = ticker_data[mask]
                if not single_data.empty:
                    all_hist_single[ticker] = single_data.iloc[0:1]  # Take first if multiple, but should be one
        total = len([t for t in tickers if t in all_hist_single])
    else:
        # Original single-day logic
        total = len(tickers)
        batch_size = 100
        batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
        all_hist_single = {}
        download_processed = 0
        total_batches = len(batches)
        batch_num = 0
        for chunk in batches:
            batch_num += 1
            hist_chunk = yf.download(chunk, start=date_str, end=end_date, group_by='ticker', threads=False, progress=False, auto_adjust=False)
            for ticker in chunk:
                if ticker in hist_chunk.columns.get_level_values(0) and not hist_chunk[ticker].empty:
                    all_hist_single[ticker] = hist_chunk[ticker]
            download_processed += len(chunk)

    all_data = []
    calc_processed = 0
    for ticker in tickers:
        try:
            if ticker in all_hist_single:
                single_data = all_hist_single[ticker]
                if not single_data.empty:
                    o = safe_float(single_data['Open'])
                    h = safe_float(single_data['High'])
                    l = safe_float(single_data['Low'])
                    c = safe_float(single_data['Close'])
                    v = safe_int(single_data['Volume'])
                    
                    range_pct = ((h - l) / c * 100) if c != 0 else 0
                    
                    range_val = (h - l)
                    signal = 'No'
                    jw_pct_for_table = 0
                    if range_val == 0:
                        close_pct_raw = 0
                        open_pct_raw = 0
                    else:
                        close_pct_raw = ((h - c) / range_val * 100)
                        open_pct_raw = ((h - o) / range_val * 100)
                    
                    bear_threshold = 100 - percentage
                    
                    if mode == 'Bullish':
                        if open_pct_raw < percentage and close_pct_raw < percentage:
                            signal = 'Yes'
                            jw_pct_for_table = close_pct_raw
                    else:  # Bearish
                        if open_pct_raw > bear_threshold and close_pct_raw > bear_threshold:
                            signal = 'Yes'
                            jw_pct_for_table = 100 - close_pct_raw
                    
                    all_data.append({
                        'Ticker': ticker,
                        'Open': round(o, 2),
                        'High': round(h, 2),
                        'Low': round(l, 2),
                        'Close': round(c, 2),
                        'Volume': int(v),
                        'Range %': round(range_pct, 2),
                        'JW %': round(jw_pct_for_table, 2),
                        'Signal': signal,
                        'JW Mode': mode
                    })
                    
                    # Print for backtest only if signal Yes - print raw values
                    if signal == 'Yes' and full_data is not None and current_date is not None:
                        print(f"{ticker}, {current_date.strftime('%Y-%m-%d')}, {close_pct_raw:.2f}, {open_pct_raw:.2f}")
        except Exception as e:
            print(f"Error for {ticker}: {e}")
            continue
        
        calc_processed += 1

    if not all_data:
        return pd.DataFrame()

    df_mode = pd.DataFrame(all_data)
    df_mode = df_mode[df_mode['Signal'] == 'Yes']
    
    if df_mode.empty:
        return pd.DataFrame()

    yes_tickers_mode = df_mode['Ticker'].tolist()
    if full_data is not None:
        # For backtest, use full_data for 30d
        all_hist_30d = {}
        for ticker in yes_tickers_mode:
            if ticker in full_data.columns.get_level_values(0):
                ticker_data = full_data[ticker]
                start_mask = (ticker_data.index.date >= pd.to_datetime(start_30d).date())
                hist_30d_mask = start_mask & (ticker_data.index.date < current_date.date())
                hist_30d = ticker_data[hist_30d_mask]
                if not hist_30d.empty:
                    all_hist_30d[ticker] = hist_30d
    else:
        # Original 30d fetch
        yes_batch_size = 50
        yes_batches = [yes_tickers_mode[i:i+yes_batch_size] for i in range(0, len(yes_tickers_mode), yes_batch_size)]
        all_hist_30d = {}
        fetch_processed = 0
        yes_total_batches = len(yes_batches)
        yes_batch_num = 0
        for yes_chunk in yes_batches:
            yes_batch_num += 1
            hist_30d_chunk = yf.download(yes_chunk, start=start_30d, end=end_date, group_by='ticker', threads=False, progress=False, auto_adjust=False)
            for ticker in yes_chunk:
                if ticker in hist_30d_chunk.columns.get_level_values(0) and not hist_30d_chunk[ticker].empty:
                    all_hist_30d[ticker] = hist_30d_chunk[ticker]
            fetch_processed += len(yes_chunk)

    # Initialize columns
    df_mode['30D Avg Vol'] = 0
    df_mode['rVolume'] = 0.0
    df_mode['rVolatility'] = 0.0

    comp_total = len(df_mode)
    comp_processed = 0
    single_idx = current_date if current_date is not None else pd.to_datetime(date_str)
    for idx, row in df_mode.iterrows():
        ticker = row['Ticker']
        try:
            if ticker in all_hist_30d:
                full_data_t = all_hist_30d[ticker]
                if not full_data_t.empty:
                    v = row['Volume']
                    
                    volumes = full_data_t['Volume'].dropna()
                    volumes_before = volumes[volumes.index.date < single_idx.date()]
                    volumes_for_avg = volumes_before.tail(30) if len(volumes_before) >= 30 else volumes_before
                    if not volumes_for_avg.empty:
                        if len(volumes_for_avg) == 1:
                            avg_vol = float(volumes_for_avg.iloc[0])
                        else:
                            avg_vol = volumes_for_avg.mean()
                    else:
                        avg_vol = 0
                    
                    rel_vol = v / avg_vol if avg_vol > 0 else 0
                    
                    df_mode.at[idx, '30D Avg Vol'] = int(round(avg_vol, 0)) if avg_vol > 0 else 0
                    df_mode.at[idx, 'rVolume'] = round(rel_vol, 2)

                    # Compute rVolatility
                    before_data = full_data_t[full_data_t.index.date < single_idx.date()]
                    if len(before_data) >= 30:
                        recent_data = before_data.tail(30)
                    elif len(before_data) > 0:
                        recent_data = before_data
                    else:
                        recent_data = pd.DataFrame()

                    if not recent_data.empty:
                        highs_recent = recent_data['High']
                        lows_recent = recent_data['Low']
                        closes_recent = recent_data['Close']
                        daily_range_pct = ((highs_recent - lows_recent) / closes_recent) * 100
                        if len(daily_range_pct) == 1:
                            avg_range_pct = float(daily_range_pct.iloc[0])
                        else:
                            avg_range_pct = daily_range_pct.mean()
                    else:
                        avg_range_pct = 0

                    current_range_pct = row['Range %']

                    r_volatility = current_range_pct / avg_range_pct if avg_range_pct > 0 else 0

                    df_mode.at[idx, 'rVolatility'] = round(r_volatility, 2)
        except Exception as e:
            print(f"Error for {ticker} volume: {e}")
            continue
        
        comp_processed += 1

    # Log exclusions in backtest mode
    if full_data is not None and current_date is not None:
        for idx, row in df_mode.iterrows():
            ticker = row['Ticker']
            excluded = []
            if row['30D Avg Vol'] <= min_avg_vol * 1000000:
                avg_m = row['30D Avg Vol'] / 1000000
                excluded.append(f"{avg_m:.1f} aVolume")
            if row['rVolume'] <= min_rel_vol:
                excluded.append(f"{row['rVolume']:.2f} rVolume")
            if row['rVolatility'] <= min_rvolat:
                excluded.append(f"{row['rVolatility']:.2f} rVolatility")
            if excluded:
                print(f"{current_date.strftime('%m/%d/%Y')} {ticker}, {', '.join(excluded)}")

    df_mode = df_mode[df_mode['30D Avg Vol'] > min_avg_vol * 1000000]
    df_mode = df_mode[df_mode['rVolume'] > min_rel_vol]
    df_mode = df_mode[df_mode['rVolatility'] > min_rvolat]
    
    if df_mode.empty:
        return pd.DataFrame()
    
    return df_mode

def process_data(date_str, percentage, filter_mode, min_avg_vol, min_rel_vol, min_rvolat, tickers=None, progress_container=None, full_data=None, current_date=None):
    if tickers is None or len(tickers) == 0:
        if progress_container is not None:
            custom_progress(progress_container, 1.0, "100%")
        return pd.DataFrame()
    
    if progress_container is None and full_data is None:
        progress_container = st.empty()
        custom_progress(progress_container, 0, "0%")
    elif progress_container is None:
        # backtest, no progress
        pass
    
    date = datetime.strptime(date_str, '%Y-%m-%d').date() if current_date is None else pd.to_datetime(current_date).date()
    end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
    start_30d = (date - timedelta(days=45)).strftime('%Y-%m-%d')

    if filter_mode == 'All':
        df_bull = process_mode('Bullish', 0.0, 0.5, progress_container, tickers, date_str, percentage, start_30d, end_date, min_avg_vol, min_rel_vol, min_rvolat, full_data, current_date)
        df_bear = process_mode('Bearish', 0.5, 1.0, progress_container, tickers, date_str, percentage, start_30d, end_date, min_avg_vol, min_rel_vol, min_rvolat, full_data, current_date)
        df = pd.concat([df_bull, df_bear]) if not df_bear.empty else df_bull
    else:
        df = process_mode(filter_mode, 0.0, 1.0, progress_container, tickers, date_str, percentage, start_30d, end_date, min_avg_vol, min_rel_vol, min_rvolat, full_data, current_date)

    if df.empty:
        if progress_container is not None:
            custom_progress(progress_container, 1.0, "100%")
        return pd.DataFrame()

    if progress_container is not None:
        custom_progress(progress_container, 1.0, "100%")

    def rel_vol_score(x):
        if x <= 0.5:
            return 0
        elif x <= 0.8:
            return 2 * (x - 0.5) / 0.3
        elif x <= 1.0:
            return 2 + 2 * (x - 0.8) / 0.2
        elif x <= 1.25:
            return 4 + 2 * (x - 1.0) / 0.25
        elif x <= 1.5:
            return 6 + 1.5 * (x - 1.25) / 0.25
        elif x <= 2.5:
            return 7.5 + 2.5 * (x - 1.5) / 1.0
        else:
            return 10
    df['Rel Vol Score'] = df['rVolume'].apply(lambda x: min(10, max(0, rel_vol_score(x))))
    
    def rvol_score(x):
        if x < 0.7:
            return 0
        elif x <= 1.2:
            return 5 * (x - 0.7) / 0.5
        elif x <= 2.0:
            return 5 + 5 * (x - 1.2) / 0.8
        else:
            return 10
    df['rVol Score'] = df['rVolatility'].apply(lambda x: min(10, max(0, rvol_score(x))))
    
    # Adjusted Close Score based on mode
    df['Close Score'] = 0.0
    bull_mask = df['JW Mode'] == 'Bullish'
    bear_mask = df['JW Mode'] == 'Bearish'
    df.loc[bull_mask, 'Close Score'] = 10 * (1 - (df.loc[bull_mask, 'JW %'] / percentage))
    df.loc[bear_mask, 'Close Score'] = 10 * ((df.loc[bear_mask, 'JW %'] - (100 - percentage)) / percentage)
    df['Close Score'] = df['Close Score'].clip(lower=0, upper=10)
    
    df['Strength'] = round((df['rVol Score'] + df['Close Score'] + df['Rel Vol Score']) / 3, 2)

    df = df.sort_values('Strength', ascending=False)
    
    # Save to session state history (only for live scan)
    if full_data is None and not df.empty:
        new_records = []
        for _, row in df.iterrows():
            rec = row.to_dict()
            rec['Query_Date'] = date_str
            new_records.append(rec)
        st.session_state.history.extend(new_records)
        if len(st.session_state.history) > 1000:
            st.session_state.history = st.session_state.history[-1000:]
    
    return df

def get_color(strength):
    # Interpolate from red to green
    factor = strength / 10.0
    r = int(255 * (1 - factor))
    g = int(255 * factor)
    b = 0
    return f'rgb({r},{g},{b})'

def style_df(df, minimalist):
    def highlight_strength(val):
        if isinstance(val, (int, float)):
            color = get_color(val)
            return f'color: {color}'
        return ''

    def highlight_pnl(val):
        if isinstance(val, (int, float)):
            color = 'lime' if val > 0 else '#FF6B6B'
            return f'color: {color}'
        return ''

    def highlight_jw_signal(val):
        if pd.isna(val):
            return ''
        color = 'lime' if val == 'Bullish' else '#FF6B6B' if val == 'Bearish' else ''
        return f'color: {color}'

    def highlight_outcome(val):
        if pd.isna(val):
            return ''
        color = 'lime' if val == 'Win' else '#FF6B6B' if val == 'Loss' else ''
        return f'color: {color}'

    subset = df.copy()

    if 'Volume' in subset.columns:
        subset['Volume'] = subset['Volume'].apply(lambda v: f"{v/1000000:.2f} M" if isinstance(v, (int, float)) and v > 0 else "0.00 M")
    if '30D Avg Vol' in subset.columns:
        subset['30D Avg Vol'] = subset['30D Avg Vol'].apply(lambda v: f"{v/1000000:.2f} M" if isinstance(v, (int, float)) and v > 0 else "0.00 M")

    if 'Date' in df.columns:  # backtest mode
        display_columns = ['Date', 'Ticker', 'Close Price', 'rVolume', 'rVolatility', 'JW %', 'JW Signal', 'Strength', 'Num Shares', 'S Balance', 'E Balance', 'Days Held', 'Win/Loss', 'PnL $', 'PnL %', 'Take Profit Target Price', 'Stop Loss Price']
        subset = subset[display_columns]
        # Handle TP and SL columns for display
        if 'Take Profit Target Price' in subset.columns:
            subset['Take Profit Target Price'] = subset['Take Profit Target Price'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
        if 'Stop Loss Price' in subset.columns:
            subset['Stop Loss Price'] = subset['Stop Loss Price'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
        float_cols = ['Open', 'High', 'Low', 'Close', 'Close Price', 'rVolume', 'rVolatility', 'JW %', 'Strength', 'PnL $', 'S Balance', 'E Balance']
    else:  # live scan
        if minimalist:
            jw_col = 'JW Mode'
            close_col = 'Close'
            display_columns = ['Ticker', close_col, 'Volume', 'rVolume', 'rVolatility', 'JW %', jw_col, 'Strength']
            subset = subset[display_columns]
        else:
            jw_col = 'JW Mode'
            close_col = 'Close'
            display_columns = ['Ticker', 'Open', 'High', 'Low', close_col, 'Volume', '30D Avg Vol', 'rVolume', 'rVolatility', 'JW %', jw_col, 'Strength']
            subset = subset.reindex(columns=[c for c in display_columns if c in subset.columns])
        float_cols = ['Open', 'High', 'Low', 'Close', 'Close Price', 'rVolume', 'rVolatility', 'JW %', 'Strength', 'PnL $']

    format_dict = {col: '{:.2f}' for col in float_cols if col in subset.columns}

    # Format PnL columns
    if 'PnL %' in subset.columns:
        format_dict['PnL %'] = '{:.1f}%'
    if 'PnL $' in subset.columns:
        format_dict['PnL $'] = '{:.2f}'
    if 'Date' in df.columns:
        format_dict['Num Shares'] = '{:.0f}'
        if 'S Balance' in subset.columns:
            format_dict['S Balance'] = '{:.2f}'
        if 'E Balance' in subset.columns:
            format_dict['E Balance'] = '{:.2f}'

    subset = subset.style.format(format_dict)

    # Apply styling to Strength column
    subset = subset.map(highlight_strength, subset=pd.IndexSlice[:, ['Strength']])
    # For backtest, highlight PnL if present
    pnl_cols = ['PnL $', 'PnL %']
    for col in pnl_cols:
        if col in df.columns:
            subset = subset.map(highlight_pnl, subset=pd.IndexSlice[:, [col]])
    
    # Highlight JW Signal
    if 'JW Signal' in subset.columns:
        subset = subset.map(highlight_jw_signal, subset=pd.IndexSlice[:, ['JW Signal']])
    
    # Highlight Outcome
    if 'Win/Loss' in subset.columns:
        subset = subset.map(highlight_outcome, subset=pd.IndexSlice[:, ['Win/Loss']])

    return subset

# Improved backtest helpers with progress and console logs
def fetch_backtest_data(start_date, end_date, tickers_list, progress_container=None):
    ensure_parquet_from_github()  # Ensure parquet is available from GitHub
    
    if isinstance(tickers_list, str) and tickers_list == 'Full Cache':
        tickers = st.session_state.default_tickers
    else:
        tickers = [tickers_list.upper()] if isinstance(tickers_list, str) else tickers_list
    
    data_dict = {}
    new_data = {}
    current_date = datetime.now().date()
    parquet_file = 'historical_ohlcv.parquet'
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    if progress_container:
        custom_progress(progress_container, 0.0, "Initializing data fetch...")
    
    print(f"Fetching backtest data from {start_date} to {end_date} for {len(tickers)} tickers.")
    
    parquet_existed = os.path.exists(parquet_file)
    grouped = {}
    if parquet_existed:
        print(f"Loading historical_ohlcv.parquet from local/GitHub...")
        if progress_container:
            custom_progress(progress_container, 0.05, "Loading parquet file...")
        # Load entire parquet once for efficiency
        df_all = pd.read_parquet(parquet_file)
        df_all['date'] = pd.to_datetime(df_all['date'])
        df_range = df_all[(df_all['date'] >= start_dt) & (df_all['date'] <= end_dt)]
        print(f"Loaded {len(df_range)} total rows from parquet for the date range.")
        
        # Group by ticker
        grouped = df_range.groupby('ticker')
        print(f"Found data in parquet for {len(grouped)} unique tickers.")
        # Convert GroupBy to dict for consistent access
        grouped = {name: group for name, group in grouped}
        
        if progress_container:
            custom_progress(progress_container, 0.1, f"Processing {len(tickers)} tickers from parquet...")
    else:
        # Fallback: download all data from yfinance if no parquet
        print("No historical_ohlcv.parquet found. Downloading full data from yfinance (this may take time).")
        grouped = {}
        df_range = pd.DataFrame()
        if progress_container:
            custom_progress(progress_container, 0.0, "Downloading full data from yfinance...")
        batch_size = 50  # Smaller batches for full download
        batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
        full_hist = pd.DataFrame()
        batch_idx = 0
        for batch in batches:
            batch_idx += 1
            print(f"Downloading batch {batch_idx}/{len(batches)}: {batch[:3]}...")
            if progress_container:
                custom_progress(progress_container, (batch_idx / len(batches)) * 0.4, f"Downloading batch {batch_idx}/{len(batches)}...")
            batch_data = yf.download(batch, start=start_date, end=(pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d'), group_by='ticker', progress=False, auto_adjust=False)
            full_hist = pd.concat([full_hist, batch_data], axis=1)
        
        # Build grouped from full_hist for consistency
        for ticker in tickers:
            if ticker in full_hist.columns.get_level_values(0):
                ticker_df = full_hist[ticker].sort_index()
                if not ticker_df.empty:
                    # Convert to long format for grouped simulation
                    long_df = ticker_df.reset_index().rename(columns={'Date': 'date'})
                    long_df['ticker'] = ticker
                    long_df['open'] = long_df['Open']
                    long_df['high'] = long_df['High']
                    long_df['low'] = long_df['Low']
                    long_df['close'] = long_df['Close']
                    long_df['volume'] = long_df['Volume']
                    long_df = long_df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
                    long_df['date'] = pd.to_datetime(long_df['date'])
                    grouped[ticker] = long_df
                    print(f"Downloaded {len(long_df)} days for {ticker} from yfinance.")
        
        print("Full download complete.")
        if progress_container:
            custom_progress(progress_container, 0.5, "Full yfinance download complete.")
        
        # Save the full download to parquet since no existing file
        all_long_dfs = [grouped[ticker] for ticker in tickers if ticker in grouped]
        if all_long_dfs:
            full_new_df = pd.concat(all_long_dfs, ignore_index=True)
            update_parquet(full_new_df)
            print("Full download saved to parquet.")
    
    # Compute union of actual trading dates across all loaded tickers
    union_dates = set()
    for t, g in grouped.items():
        if not g.empty and 'date' in g.columns:
            union_dates.update(g['date'].dt.date)
    print(f"Union of {len(union_dates)} actual trading dates across tickers.")
    
    # Process tickers
    tickers_set = set(tickers)
    processed_tickers = 0
    for ticker in tickers:
        processed_tickers += 1
        print(f"Processing {ticker} ({processed_tickers}/{len(tickers)})...")
        
        # Update progress less frequently to avoid multiple bars
        if progress_container and processed_tickers % 10 == 0:
            prog = 0.1 + (processed_tickers / len(tickers)) * 0.3
            custom_progress(progress_container, prog, f"Processing ticker {processed_tickers}/{len(tickers)}: {ticker}")
        
        group_long = grouped.get(ticker, pd.DataFrame())
        existing_dates = set(group_long['date'].dt.date) if not group_long.empty else set()
        
        missing_dates_set = union_dates - existing_dates
        if missing_dates_set:
            print(f"Found {len(missing_dates_set)} missing dates for {ticker}, downloading...")
            missing_dates = sorted([pd.Timestamp(d) for d in missing_dates_set])
            miss_start = missing_dates[0].strftime('%Y-%m-%d')
            miss_end = (missing_dates[-1] + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"Downloading missing data for {ticker} from {miss_start} to {miss_end} via yfinance...")
            # Removed progress update for missing downloads to avoid frequent updates
            miss_data = yf.download(ticker, start=miss_start, end=miss_end, progress=False, auto_adjust=False)
            
            if not miss_data.empty:
                downloaded_count = 0
                for date_idx, row in miss_data.iterrows():
                    date_d = date_idx.date()
                    if date_d not in missing_dates_set:
                        continue
                    # Skip if NaN
                    if pd.isna(row[['Open', 'High', 'Low', 'Close', 'Volume']]).any():
                        continue
                    
                    # Collect for new_data update
                    date_str = date_idx.strftime('%Y-%m-%d')
                    if ticker not in new_data:
                        new_data[ticker] = {}
                    new_data[ticker][date_str] = {
                        'O': safe_float(row['Open']),
                        'H': safe_float(row['High']),
                        'L': safe_float(row['Low']),
                        'C': safe_float(row['Close']),
                        'V': safe_int(row['Volume'])
                    }
                    
                    # Add to group_long
                    added_row = pd.Series({
                        'ticker': ticker,
                        'date': date_idx,
                        'open': row['Open'],
                        'high': row['High'],
                        'low': row['Low'],
                        'close': row['Close'],
                        'volume': row['Volume']
                    })
                    group_long = pd.concat([group_long, added_row.to_frame().T], ignore_index=True)
                    
                    downloaded_count += 1
                print(f"Downloaded and added {downloaded_count} missing days for {ticker} from yfinance.")
            else:
                print(f"No data downloaded for missing dates of {ticker}.")
        
        # Now set index and rename for the updated group
        group = group_long
        if not group.empty:
            group = group.set_index('date')
            group.index = pd.to_datetime(group.index)
            # Rename columns to match yfinance
            group = group[['open', 'high', 'low', 'close', 'volume']].rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
            })
            print(f"Loaded {len(group)} days from parquet for {ticker}.")
        else:
            group = pd.DataFrame()
            print(f"No data available for {ticker} after download.")
        
        if not group.empty:
            group = group.sort_index()
            data_dict[ticker] = group
    
    print("All tickers processed. Building full dataset...")
    if progress_container:
        custom_progress(progress_container, 0.5, "Building full dataset...")
    
    # Build full_data as MultiIndex
    if data_dict:
        full_data_list = []
        for ticker, tdf in data_dict.items():
            tdf_copy = tdf.copy()
            tdf_copy.columns = pd.MultiIndex.from_product([[ticker], tdf.columns])
            full_data_list.append(tdf_copy)
        full_data = pd.concat(full_data_list, axis=1)
        print(f"Built full_data with {len(full_data)} rows across {len(data_dict)} tickers.")
    else:
        full_data = pd.DataFrame()
        print("No data built.")
    
    # Update Parquet at the end with new data
    if new_data:
        print("Updating parquet with new data...")
        new_df = flatten_new_data(new_data)
        update_parquet(new_df)
        print("Parquet updated.")
    
    if progress_container:
        custom_progress(progress_container, 0.5, "Historical data fetch complete.")
    
    return full_data

def get_forward_data(full_data, ticker, current_date):
    if ticker not in full_data.columns.get_level_values(0):
        return pd.DataFrame()
    ticker_data = full_data[ticker]
    mask = ticker_data.index.date > current_date.date()
    after = ticker_data[mask]
    return after

def backtest_engine(start_date, end_date, tickers_list, jw_mode, sl_strategy, sl_percent=0.0, jw_percent=20.0, min_avg_vol=0.1, min_rel_vol=0.9, min_rvolat=1.0, rr=1.5, portfolio_size=100000.0, position_size=10000.0, progress_container=None):
    # Handle tickers: full list or single
    if isinstance(tickers_list, str) and tickers_list == 'Full Cache':
        tickers = st.session_state.default_tickers
    else:
        tickers = [tickers_list.upper()] if isinstance(tickers_list, str) else tickers_list

    if progress_container:
        custom_progress(progress_container, 0.0, "Fetching historical data...")
    
    full_data = fetch_backtest_data(start_date, end_date, tickers_list, progress_container)
    
    if full_data.empty:
        default_summary = {
            'Total_Signals': 0,
            'Win_Rate': 0.0,
            'Avg_PnL': 0.0,
            'Max_Drawdown': 0.0,
            'Sharpe': 0.0,
            'Avg_Days_Held': 0.0,
            'Avg_PnL_Dollar': 0.0,
            'Starting_Portfolio': portfolio_size,
            'Ending_Portfolio': portfolio_size
        }
        if progress_container:
            custom_progress(progress_container, 1.0, "100% - No data available")
        return pd.DataFrame(), default_summary
    
    # Use actual data dates/bars only - key fix for no results
    dates = sorted(full_data.index.unique())
    
    total_dates = len(dates)
    print(f"Starting backtest on {total_dates} dates from {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}.")

    all_signals = []
    processed_dates = 0
    if progress_container:
        custom_progress(progress_container, 0.5, "Initializing backtest...")
    for date in dates:
        try:
            print(f"Processing date: {date.strftime('%Y-%m-%d')} ({processed_dates + 1}/{total_dates})")
            # Use date as current_date for slicing
            df_day = process_data(date.strftime('%Y-%m-%d'), jw_percent, jw_mode, min_avg_vol, min_rel_vol, min_rvolat, tickers, None, full_data, date)
            if not df_day.empty:
                print(f"Found {len(df_day)} signals on {date.strftime('%Y-%m-%d')}.")
                for _, signal in df_day.iterrows():
                    ticker = signal['Ticker']
                    mode = signal['JW Mode']
                    entry_o = signal['Open']
                    entry_h = signal['High']
                    entry_l = signal['Low']
                    entry_c = signal['Close']
                    
                    # Compute SL price
                    sl_price = None
                    if sl_strategy != 'None':
                        if sl_strategy == 'Full Wick':
                            sl_price = entry_l if mode == 'Bullish' else entry_h
                        elif sl_strategy == 'Half Wick':
                            lower_oc = min(entry_o, entry_c)
                            higher_oc = max(entry_o, entry_c)
                            if mode == 'Bullish':
                                wick_size = (lower_oc - entry_l) / 2
                                sl_price = lower_oc - wick_size
                            else:
                                wick_size = (entry_h - higher_oc) / 2
                                sl_price = higher_oc + wick_size
                        elif sl_strategy == 'Custom':
                            if sl_percent > 0:
                                sl_pct = sl_percent / 100
                                if mode == 'Bullish':
                                    sl_price = entry_c * (1 - sl_pct)
                                else:
                                    sl_price = entry_c * (1 + sl_pct)
                    
                    # Compute TP price
                    tp_price = None
                    if sl_price is not None:
                        if mode == 'Bullish':
                            max_loss = entry_c - sl_price
                            tp_price = entry_c + (max_loss * rr)
                        else:
                            max_loss = sl_price - entry_c
                            tp_price = entry_c - (max_loss * rr)
                    
                    # Get all forward bars
                    forward_bars = get_forward_data(full_data, ticker, date)
                    if forward_bars.empty:
                        continue
                    
                    # Find first hit
                    hit_day = None
                    exit_price = None
                    outcome = None
                    hit = False
                    dir = 1 if mode == 'Bullish' else -1
                    for i, bar in enumerate(forward_bars.itertuples(index=False)):
                        hit_tp = (tp_price is not None) and ((mode == 'Bullish' and bar.High >= tp_price) or (mode == 'Bearish' and bar.Low <= tp_price))
                        hit_sl = (sl_price is not None) and ((mode == 'Bullish' and bar.Low <= sl_price) or (mode == 'Bearish' and bar.High >= sl_price))
                        if hit_tp:
                            hit_day = i + 1
                            exit_price = tp_price
                            outcome = 'Win'
                            hit = True
                            break
                        elif hit_sl:
                            hit_day = i + 1
                            exit_price = sl_price
                            outcome = 'Loss'
                            hit = True
                            break
                    
                    if not hit:
                        # To end
                        last_bar = forward_bars.iloc[-1]
                        exit_price = last_bar.Close
                        pnl_temp = dir * (exit_price - entry_c) / entry_c * 100
                        outcome = 'Win' if pnl_temp > 0 else 'Loss'
                        hit_day = len(forward_bars)
                    
                    num_shares = int(round(position_size / entry_c))
                    s_balance = round(num_shares * entry_c, 2)
                    pnl_per_share = dir * (exit_price - entry_c)
                    pnl_dollar = round(num_shares * pnl_per_share, 2)
                    e_balance = round(s_balance + pnl_dollar, 2)
                    pnl_percent = (pnl_per_share / entry_c) * 100
                    
                    all_signals.append({
                        'Ticker': ticker,
                        'Signal_Date': date,
                        'Mode': mode,
                        'Entry_Close': entry_c,
                        'Num_Shares': num_shares,
                        'S_Balance': s_balance,
                        'E_Balance': e_balance,
                        'PnL_$': pnl_dollar,
                        'PnL_%': pnl_percent,
                        'Days_Held': hit_day,
                        'Outcome': outcome,
                        'TP_Price': tp_price if tp_price is not None else np.nan,
                        'SL_Price': sl_price if sl_price is not None else np.nan,
                        'rVolume': signal['rVolume'],
                        'Strength': signal['Strength'],
                        'JW_Percent': signal['JW %'],
                        'rVolatility': signal['rVolatility']
                    })
            else:
                print(f"No signals found on {date.strftime('%Y-%m-%d')}.")
        except Exception as e:
            print(f"Error processing {date}: {e}")
        
        processed_dates += 1
        # Update progress less frequently to avoid multiple bars
        if progress_container and processed_dates % 5 == 0:
            progress_val = 0.5 + (processed_dates / total_dates) * 0.5
            custom_progress(progress_container, progress_val, f"Processing date {processed_dates}/{total_dates}: {date.strftime('%Y-%m-%d')}")
    
    # Always return full summary
    default_summary = {
        'Total_Signals': 0,
        'Win_Rate': 0.0,
        'Avg_PnL': 0.0,
        'Max_Drawdown': 0.0,
        'Sharpe': 0.0,
        'Avg_Days_Held': 0.0,
        'Avg_PnL_Dollar': 0.0,
        'Starting_Portfolio': portfolio_size,
        'Ending_Portfolio': portfolio_size
    }
    
    if not all_signals:
        if progress_container:
            custom_progress(progress_container, 1.0, "100% - No signals found")
        return pd.DataFrame(), default_summary
    
    df_bt = pd.DataFrame(all_signals)
    
    # Compute summary
    if len(df_bt) == 0:
        win_rate = 0.0
        avg_pnl = 0.0
        sharpe = 0.0
        max_dd = 0.0
        avg_days_held = 0.0
        avg_pnl_dollar = 0.0
        ending_portfolio = portfolio_size
    else:
        win_rate = (df_bt['Outcome'] == 'Win').mean() * 100
        pnls = df_bt['PnL_%']
        avg_pnl = pnls.mean()
        sharpe = avg_pnl / (pnls.std() + 1e-8)
        cum_returns = np.cumsum(pnls - avg_pnl)
        max_dd = np.min(cum_returns) if len(cum_returns) > 0 else 0.0
        avg_days_held = df_bt['Days_Held'].mean()
        avg_pnl_dollar = df_bt['PnL_$'].mean()
        ending_portfolio = portfolio_size + df_bt['PnL_$'].sum()
    
    summary = {
        'Total_Signals': len(df_bt),
        'Win_Rate': round(win_rate, 1),
        'Avg_PnL': round(avg_pnl, 2),
        'Max_Drawdown': round(max_dd, 2),
        'Sharpe': round(sharpe, 2),
        'Avg_Days_Held': round(avg_days_held, 1),
        'Avg_PnL_Dollar': round(avg_pnl_dollar, 2),
        'Starting_Portfolio': portfolio_size,
        'Ending_Portfolio': ending_portfolio
    }
    
    print(f"Backtest complete: {len(df_bt)} signals found.")
    if progress_container:
        custom_progress(progress_container, 1.0, f"100% - Backtest complete: {len(df_bt)} signals found")
    
    return df_bt, summary

# Template CSV for tickers
template_df = pd.DataFrame({'Ticker': ['AAPL', 'GOOGL']})
template_csv = template_df.to_csv(index=False).encode('utf-8')

# Top right controls
col_ctrls1, col_ctrls2, col_ctrls3 = st.columns([17, 2, 1])
with col_ctrls2:
    minimalist = st.checkbox("Minimalist View", key="minimalist")
with col_ctrls3:
    if st.button("⚙️", key="settings_icon", help="Settings"):
        st.session_state.show_settings = not st.session_state.show_settings

# Settings popup
if st.session_state.show_settings:
    st.markdown("## Settings")
    ticker_list_options = ['Default'] + list(st.session_state.ticker_lists.keys())
    current_list = st.selectbox("Select Ticker List", ticker_list_options, index=ticker_list_options.index(st.session_state.selected_ticker_list) if st.session_state.selected_ticker_list in ticker_list_options else 0)
    
    st.subheader("Default Tickers Cache")
    col_fetch1, col_fetch2 = st.columns(2)
    with col_fetch1:
        if st.button("Fetch Tickers"):
            with st.spinner("Fetching new tickers..."):
                new_tickers = get_tickers()
                old_len = len(st.session_state.default_tickers)
                st.session_state.default_tickers = list(set(st.session_state.default_tickers + new_tickers))
                save_tickers(st.session_state.default_tickers)
                if len(st.session_state.default_tickers) > old_len:
                    st.success(f"Added {len(st.session_state.default_tickers) - old_len} new tickers. Total: {len(st.session_state.default_tickers)}")
                    push_to_github()
                else:
                    st.info("No new tickers added.")
    
    with col_fetch2:
        st.info(f"Current cached tickers: {len(st.session_state.default_tickers)}")
    
    st.subheader("Historical Data Cache")
    now = datetime.now()
    last_day = monthrange(now.year, now.month)[1]
    max_hist_end = date(now.year, now.month, last_day)
    col_hist1, col_hist2 = st.columns(2)
    with col_hist1:
        hist_start = st.date_input("Start Date for Fetch", value=datetime(2020, 1, 1).date(), key="hist_start")
    with col_hist2:
        hist_end = st.date_input("End Date for Fetch", value=now.date(), max_value=max_hist_end, key="hist_end")
    
    progress_hist = st.empty()
    
    if st.button("Fetch Data"):
        with st.spinner("Fetching historical data..."):
            custom_progress(progress_hist, 0.0, "Initializing...")
            tickers = st.session_state.default_tickers
            if not tickers:
                st.warning("No tickers available. Fetch default tickers first.")
                custom_progress(progress_hist, 1.0, "100% - No tickers")
            else:
                batch_size = 50
                batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
                all_hist = pd.DataFrame()
                for batch_idx, batch in enumerate(batches):
                    custom_progress(progress_hist, (batch_idx / len(batches)) * 0.8, f"Downloading batch {batch_idx+1}/{len(batches)}")
                    batch_data = yf.download(batch, start=hist_start, end=(hist_end + timedelta(days=1)).strftime('%Y-%m-%d'), group_by='ticker', progress=False, auto_adjust=False)
                    all_hist = pd.concat([all_hist, batch_data], axis=1)
                
                custom_progress(progress_hist, 0.8, "Processing data...")
                all_data = []
                processed = 0
                total_tickers = len(tickers)
                for ticker in tickers:
                    processed += 1
                    if processed % 10 == 0 or processed == total_tickers:
                        custom_progress(progress_hist, 0.8 + (processed / total_tickers) * 0.2, f"Processing {processed}/{total_tickers} tickers")
                    if ticker in all_hist.columns.get_level_values(0):
                        ticker_df = all_hist[ticker].reset_index()
                        ticker_df['ticker'] = ticker
                        renamed_df = ticker_df[['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={
                            'Date': 'date',
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume'
                        })
                        all_data.append(renamed_df)
                if all_data:
                    full_df = pd.concat(all_data, ignore_index=True)
                    full_df['date'] = pd.to_datetime(full_df['date'])
                    update_parquet(full_df)
                    custom_progress(progress_hist, 1.0, "100% - Historical data cached!")
                else:
                    custom_progress(progress_hist, 1.0, "100% - No data to cache")
    
    st.subheader("Create New List")
    st.download_button("Download Template", template_csv, "ticker_template.csv", "text/csv")
    with st.form("new_list_form"):
        list_name = st.text_input("List Name")
        uploaded_file = st.file_uploader("Upload CSV (one column: Ticker)", type="csv")
        save_new_btn = st.form_submit_button("Save New List")
        if save_new_btn and uploaded_file and list_name:
            df_upload = pd.read_csv(uploaded_file)
            if 'Ticker' in df_upload.columns:
                tickers = df_upload['Ticker'].dropna().astype(str).tolist()
                st.session_state.ticker_lists[list_name] = tickers
                save_ticker_lists()
                st.rerun()
            else:
                pass

    if st.button("Save Settings & Close"):
        st.session_state.selected_ticker_list = current_list
        st.session_state.show_settings = False
        st.rerun()

# Tab structure - Backtest Engine as default primary tab on left
tab1, tab2 = st.tabs(["Backtest Engine", "Live Scan"])

with tab1:
    # Backtest inputs
    resolution = '1d'  # Fixed to daily
    col_bt1, col_bt2, col_bt3 = st.columns(3)
    with col_bt1:
        start_date = st.date_input("Start Date", value=st.session_state.bt_start_date, min_value=datetime(2010, 1, 1).date(), max_value=datetime.now().date(), key="bt_start")
        st.session_state.bt_start_date = start_date
        end_date = st.date_input("End Date", value=st.session_state.bt_end_date, min_value=start_date, max_value=datetime.now().date(), key="bt_end")
        st.session_state.bt_end_date = end_date
        ticker_choice = st.selectbox("Tickers", ['Full Cache', 'Manual'], index=0, key="bt_tickers_choice")
        if ticker_choice == 'Manual':
            manual_ticker = st.text_input("Enter Ticker", value="AAPL", key="bt_manual")
            bt_tickers = manual_ticker
        else:
            bt_tickers = 'Full Cache'
    with col_bt2:
        bt_jw_mode = st.selectbox("JW Mode", ['All', 'Bullish', 'Bearish'], index=0, key="bt_jw_mode")
        bt_rr = st.number_input("R/R", value=1.5, min_value=0.0, step=0.1, key="bt_rr")
        sl_strategy = st.selectbox("Stop Loss", ['None', 'Full Wick', 'Half Wick', 'Custom'], index=1, key="bt_sl")
        bt_sl_percent = 0.0
        if sl_strategy == 'Custom':
            bt_sl_percent = st.number_input("Stop Loss %", value=2.0, min_value=0.0, step=0.1, key="bt_sl_percent")
    with col_bt3:
        bt_jw_percent = st.number_input("JW %", value=20.0, min_value=0.0, step=1.0, key="bt_jw_percent")
        bt_min_avg_vol = st.number_input("Min aVolume (M)", value=0.1, min_value=0.0, step=0.1, key="bt_min_avg_vol")
        bt_min_rel_vol = st.number_input("Min rVolume", value=0.9, min_value=0.0, step=0.1, key="bt_min_rel_vol")
        bt_min_rvolat = st.number_input("Min rVolatility", value=1.0, min_value=0.0, step=0.1, key="bt_min_rvolat")
        bt_portfolio_size = st.number_input("Portfolio Size ($)", value=100000.0, min_value=0.0, step=10000.0, key="bt_portfolio")
        bt_position_size = st.number_input("Position Size ($)", value=10000.0, min_value=0.0, step=1000.0, key="bt_position")

    bt_progress = st.empty()

    if st.session_state.backtest_results is None:
        custom_progress(bt_progress, 0, "0%")

    if st.button("Run Backtest", key="run_bt"):
        # Removed st.spinner to eliminate the spinning wheel and text
        df_bt, summary = backtest_engine(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), bt_tickers, bt_jw_mode, sl_strategy, bt_sl_percent, bt_jw_percent, min_avg_vol=bt_min_avg_vol, min_rel_vol=bt_min_rel_vol, min_rvolat=bt_min_rvolat, rr=bt_rr, portfolio_size=bt_portfolio_size, position_size=bt_position_size, progress_container=bt_progress)
        st.session_state.backtest_results = (df_bt, summary)
        st.rerun()

    if st.session_state.backtest_results is not None:
        try:
            df_bt, summary = st.session_state.backtest_results
        except (ValueError, TypeError):
            st.error("Invalid backtest results format. Rerun the backtest.")
            st.session_state.backtest_results = None
            st.rerun()
        
        # Ensure summary has defaults
        default_keys = {'Total_Signals': 0, 'Win_Rate': 0.0, 'Avg_PnL': 0.0, 'Max_Drawdown': 0.0, 'Sharpe': 0.0, 'Avg_Days_Held': 0.0, 'Avg_PnL_Dollar': 0.0, 'Starting_Portfolio': bt_portfolio_size, 'Ending_Portfolio': bt_portfolio_size}
        for key, val in default_keys.items():
            if key not in summary:
                summary[key] = val
        
        if summary['Total_Signals'] == 0:
            st.info("🔍 No John Wicks identified in the backtest period. Try loosening filters (e.g., lower min rVolume) or extending the date range.")
        else:
            # Summary metrics
            col_sum1, col_sum2, col_sum3, col_sum4, col_sum5 = st.columns(5)
            with col_sum1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Signals</h3>
                    <p style="color: lime; font-size: 20px; font-weight: bold;">{summary['Total_Signals']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col_sum2:
                color = 'lime' if summary['Win_Rate'] > 50 else '#FF6B6B'
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Win Rate</h3>
                    <p style="color: {color}; font-size: 20px; font-weight: bold;">{summary['Win_Rate']}%</p>
                </div>
                """, unsafe_allow_html=True)
            with col_sum3:
                color = 'lime' if summary['Avg_PnL'] > 0 else '#FF6B6B'
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg P&L</h3>
                    <p style="color: {color}; font-size: 20px; font-weight: bold;">{summary['Avg_PnL']}%</p>
                </div>
                """, unsafe_allow_html=True)
            with col_sum4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Sharpe Ratio</h3>
                    <p style="color: lime; font-size: 20px; font-weight: bold;">{summary['Sharpe']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col_sum5:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg Days Held</h3>
                    <p style="color: lime; font-size: 20px; font-weight: bold;">{summary['Avg_Days_Held']}</p>
                </div>
                """, unsafe_allow_html=True)

            # New summary metrics
            col_new1, col_new2, col_new3 = st.columns(3)
            with col_new1:
                color_new1 = 'lime' if summary['Avg_PnL_Dollar'] > 0 else '#FF6B6B'
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg PnL $ / Trade</h3>
                    <p style="color: {color_new1}; font-size: 20px; font-weight: bold;">${summary['Avg_PnL_Dollar']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            with col_new2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Starting Portfolio</h3>
                    <p style="color: lime; font-size: 20px; font-weight: bold;">${summary['Starting_Portfolio']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            with col_new3:
                color_new3 = 'lime' if summary['Ending_Portfolio'] > summary['Starting_Portfolio'] else '#FF6B6B'
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Ending Portfolio</h3>
                    <p style="color: {color_new3}; font-size: 20px; font-weight: bold;">${summary['Ending_Portfolio']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

            # Table
            if not df_bt.empty:
                # Rename columns
                df_display = df_bt.rename(columns={'Signal_Date': 'Date', 'JW_Percent': 'JW %', 'Mode': 'JW Signal', 'Entry_Close': 'Close Price', 'Days_Held': 'Days Held', 'Outcome': 'Win/Loss', 'PnL_$': 'PnL $', 'PnL_%': 'PnL %', 'TP_Price': 'Take Profit Target Price', 'SL_Price': 'Stop Loss Price', 'Num_Shares': 'Num Shares', 'S_Balance': 'S Balance', 'E_Balance': 'E Balance'})
                df_display = df_display.sort_values('Strength', ascending=False)
                styled_bt = style_df(df_display, minimalist)
                st.dataframe(styled_bt, width='stretch', hide_index=True)
                
                csv_bt = df_bt.to_csv(index=False).encode('utf-8')
                st.download_button("EXPORT BACKTEST CSV", csv_bt, "jw_backtest.csv", "text/csv", key="export_bt")

            # Charts if not minimalist and signals exist
            if not minimalist and summary['Total_Signals'] > 0 and not df_bt.empty:
                try:
                    col_chart1, col_chart2 = st.columns(2)
                    with col_chart1:
                        # Avg P&L
                        if 'PnL_%' in df_bt.columns:
                            avg_pnl = df_bt['PnL_%'].mean()
                            fig_pnl = px.bar(x=['PnL'], y=[avg_pnl], title="Avg P&L")
                            fig_pnl.add_hline(y=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_pnl, use_container_width=True)
                        else:
                            st.warning("No PnL data for P&L chart.")
                    
                    with col_chart2:
                        # Win rate by mode
                        if 'Outcome' in df_bt.columns:
                            win_by_mode = df_bt.groupby('Mode')['Outcome'].apply(lambda x: (x == 'Win').mean() * 100).reset_index()
                            win_by_mode.columns = ['Mode', 'Win_Rate_%']
                            fig_win = px.bar(win_by_mode, x='Mode', y='Win_Rate_%', title="Win Rate by Mode")
                            st.plotly_chart(fig_win, use_container_width=True)
                        else:
                            st.warning("No Outcome data for win rate chart.")
                except Exception as e:
                    st.warning(f"Error generating charts: {e}")

with tab2:
    # Original inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        date = st.date_input("Date", value=datetime.now().date(), max_value=datetime.now().date(), key="date")
        date_str = date.strftime('%Y-%m-%d')
        min_rvolat = st.number_input("Min rVolatility", value=1.0, min_value=0.0, step=0.1, key="min_rvolat")
    with col2:
        jw_percent = st.number_input("JW %", value=20.0, min_value=0.0, step=1.0, key="jw_percent")
        min_avg_vol = st.number_input("Min aVolume (M)", value=5.0, min_value=0.0, step=0.5, key="min_avg_vol")
    with col3:
        jw_mode = st.selectbox("JW Mode", ['All', 'Bullish', 'Bearish'], index=0, key="jw_mode")
        min_rel_vol = st.number_input("Min rVolume", value=0.9, min_value=0.0, step=0.1, key="min_rel_vol")

    progress_container = st.empty()

    if st.session_state.analysis_run:
        if st.session_state.last_df.empty:
            custom_progress(progress_container, 1.0, "100%")
        else:
            custom_progress(progress_container, 1.0, "100%")
    else:
        custom_progress(progress_container, 0, "0%")

    # Analysis button
    col_btn = st.columns([1])
    selected = st.session_state.selected_ticker_list
    tickers_to_use = st.session_state.default_tickers if selected == 'Default' else st.session_state.ticker_lists.get(selected, [])

    with col_btn[0]:
        if st.button("Fortis Fortuna Adiuvat", key="run_analysis"):
            st.session_state.last_df = process_data(date_str, jw_percent, jw_mode, min_avg_vol, min_rel_vol, min_rvolat, tickers_to_use, progress_container)
            st.session_state.analysis_run = True
            st.rerun()

    # Results table
    if st.session_state.analysis_run:
        if not st.session_state.last_df.empty:
            styled_df = style_df(st.session_state.last_df, minimalist)
            st.dataframe(styled_df, width='stretch', hide_index=True)
            
            # Export
            csv = st.session_state.last_df.to_csv(index=False).encode('utf-8')
            st.download_button("EXPORT CSV", csv, "jw_terminal.csv", "text/csv", key="export")

# Load default tickers if not already loaded
if not st.session_state.default_tickers:
    st.session_state.default_tickers = load_cached_tickers()