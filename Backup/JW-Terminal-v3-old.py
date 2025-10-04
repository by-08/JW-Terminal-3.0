import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import io
import subprocess
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

# Top right controls
col_ctrls1, col_ctrls2, col_ctrls3 = st.columns([17, 2, 1])
with col_ctrls2:
    minimalist = st.checkbox("Minimalist View", key="minimalist")
with col_ctrls3:
    if st.button("⚙️", key="settings_icon", help="Settings"):
        st.session_state.show_settings = not st.session_state.show_settings

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
    st.session_state.default_tickers = load_cached_tickers()
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'bt_start_date' not in st.session_state:
    st.session_state.bt_start_date = datetime.now().date() - timedelta(days=365)
if 'bt_end_date' not in st.session_state:
    st.session_state.bt_end_date = datetime.now().date()

# Function to save ticker lists
def save_ticker_lists():
    with open('ticker_lists.json', 'w') as f:
        json.dump(st.session_state.ticker_lists, f)

def push_to_github():
    try:
        subprocess.run(['git', 'add', 'tickers_cache.json'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Update tickers cache'], check=True)
        subprocess.run(['git', 'push'], check=True)
        st.success("Pushed updated tickers_cache to GitHub.")
    except Exception as e:
        st.warning(f"Could not push to GitHub: {e}. Ensure git is configured with credentials.")

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
                mask = ticker_data.index == current_date
                single_data = ticker_data[mask]
                if not single_data.empty:
                    all_hist_single[ticker] = single_data
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
            hist_chunk = yf.download(chunk, start=date_str, end=end_date, group_by='ticker', threads=False, progress=False)
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
                    o = single_data['Open'].iloc[0]
                    h = single_data['High'].iloc[0]
                    l = single_data['Low'].iloc[0]
                    c = single_data['Close'].iloc[0]
                    v = single_data['Volume'].iloc[0]
                    
                    range_pct = ((h - l) / c * 100) if c != 0 else 0
                    
                    range_val = (h - l)
                    if range_val == 0:
                        close_pct = 0
                        open_pct = 0
                        signal = 'No'
                    else:
                        close_pct = ((h - c) / range_val * 100)
                        open_pct = ((h - o) / range_val * 100)
                        
                        if mode == 'Bullish':
                            if open_pct < 33 and close_pct < percentage:
                                signal = 'Yes'
                            else:
                                signal = 'No'
                        else:  # Bearish
                            bear_threshold = 100 - percentage
                            if open_pct > 66 and close_pct > bear_threshold:
                                signal = 'Yes'
                            else:
                                signal = 'No'
                    
                    # Adjust JW % for Bearish
                    if mode == 'Bearish':
                        close_pct = 100 - close_pct
                    
                    # Print for backtest
                    if full_data is not None and current_date is not None:
                        print(f"{ticker}, {current_date.strftime('%Y-%m-%d')}, {close_pct:.2f}, {open_pct:.2f}")
                    
                    all_data.append({
                        'Ticker': ticker,
                        'Open': round(o, 2),
                        'High': round(h, 2),
                        'Low': round(l, 2),
                        'Close': round(c, 2),
                        'Volume': int(v),
                        'Range %': round(range_pct, 2),
                        'JW %': round(close_pct, 2),
                        'Signal': signal,
                        'JW Mode': mode
                    })
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
                start_mask = ticker_data.index >= pd.to_datetime(start_30d)
                hist_30d = ticker_data[start_mask & (ticker_data.index < current_date)]
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
            hist_30d_chunk = yf.download(yes_chunk, start=start_30d, end=end_date, group_by='ticker', threads=False, progress=False)
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
    single_idx = current_date if current_date else pd.to_datetime(date_str)
    for idx, row in df_mode.iterrows():
        ticker = row['Ticker']
        try:
            if ticker in all_hist_30d:
                full_data_t = all_hist_30d[ticker]
                if not full_data_t.empty:
                    v = row['Volume']
                    
                    volumes = full_data_t['Volume'].dropna()
                    volumes_before = volumes[volumes.index < single_idx]
                    if len(volumes_before) >= 30:
                        avg_vol = volumes_before.tail(30).mean()
                    elif len(volumes_before) > 0:
                        avg_vol = volumes_before.mean()
                    else:
                        avg_vol = 0
                    
                    rel_vol = v / avg_vol if avg_vol > 0 else 0
                    
                    df_mode.at[idx, '30D Avg Vol'] = int(round(avg_vol, 0)) if avg_vol > 0 else 0
                    df_mode.at[idx, 'rVolume'] = round(rel_vol, 2)

                    # Compute rVolatility
                    before_data = full_data_t[full_data_t.index < single_idx]
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
        if progress_container:
            custom_progress(progress_container, 1.0, "100%")
        return pd.DataFrame()
    
    if progress_container is None:
        progress_container = st.empty()
    
    custom_progress(progress_container, 0, "0%")
    
    date = datetime.strptime(date_str, '%Y-%m-%d').date() if not current_date else pd.to_datetime(current_date).date()
    end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
    start_30d = (date - timedelta(days=45)).strftime('%Y-%m-%d')

    if filter_mode == 'All':
        df_bull = process_mode('Bullish', 0.0, 0.5, progress_container, tickers, date_str, percentage, start_30d, end_date, min_avg_vol, min_rel_vol, min_rvolat, full_data, current_date)
        df_bear = process_mode('Bearish', 0.5, 1.0, progress_container, tickers, date_str, percentage, start_30d, end_date, min_avg_vol, min_rel_vol, min_rvolat, full_data, current_date)
        df = pd.concat([df_bull, df_bear]) if not df_bear.empty else df_bull
    else:
        df = process_mode(filter_mode, 0.0, 1.0, progress_container, tickers, date_str, percentage, start_30d, end_date, min_avg_vol, min_rel_vol, min_rvolat, full_data, current_date)

    if df.empty:
        custom_progress(progress_container, 1.0, "100%")
        return pd.DataFrame()

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
        display_columns = ['Date', 'Ticker', 'Close Price', 'rVolume', 'rVolatility', 'JW %', 'JW Signal', 'Strength', 'Days Held', 'Win/Loss', 'PnL $', 'PnL %', 'Take Profit Target Price', 'Stop Loss Price']
        subset = subset[display_columns]
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

    float_cols = ['Open', 'High', 'Low', 'Close', 'Close Price', 'rVolume', 'rVolatility', 'JW %', 'Strength', 'PnL $', 'Take Profit Target Price', 'Stop Loss Price']
    format_dict = {col: '{:.2f}' for col in float_cols if col in subset.columns}

    # Format PnL columns
    if 'PnL %' in subset.columns:
        format_dict['PnL %'] = '{:.1f}%'
    if 'PnL $' in subset.columns:
        format_dict['PnL $'] = '{:.2f}'

    subset = subset.style.format(format_dict)

    # Apply styling to Strength column
    subset = subset.applymap(highlight_strength, subset=pd.IndexSlice[:, ['Strength']])
    # For backtest, highlight PnL if present
    pnl_cols = ['PnL $', 'PnL %']
    for col in pnl_cols:
        if col in df.columns:
            subset = subset.applymap(highlight_pnl, subset=pd.IndexSlice[:, [col]])
    
    # Highlight JW Signal
    if 'JW Signal' in subset.columns:
        subset = subset.applymap(highlight_jw_signal, subset=pd.IndexSlice[:, ['JW Signal']])
    
    # Highlight Outcome
    if 'Win/Loss' in subset.columns:
        subset = subset.applymap(highlight_outcome, subset=pd.IndexSlice[:, ['Win/Loss']])

    return subset

# Backtest helpers
def fetch_backtest_data(start_date, end_date, tickers_list):
    if isinstance(tickers_list, str) and tickers_list == 'Full Cache':
        tickers = st.session_state.default_tickers
    else:
        tickers = [tickers_list.upper()] if isinstance(tickers_list, str) else tickers_list
    full_data = yf.download(tickers, start=start_date, end=end_date, interval='1d', group_by='ticker', threads=False, progress=False)
    return full_data

def get_forward_data(full_data, ticker, current_date):
    if ticker not in full_data.columns.get_level_values(0):
        return pd.DataFrame()
    ticker_data = full_data[ticker]
    mask = ticker_data.index > current_date
    after = ticker_data[mask]
    return after

def backtest_engine(start_date, end_date, tickers_list, jw_mode, sl_strategy, jw_percent=20.0, min_avg_vol=5.0, min_rel_vol=0.9, min_rvolat=1.0, rr=1.5):
    # Handle tickers: full list or single
    if isinstance(tickers_list, str) and tickers_list == 'Full Cache':
        tickers = st.session_state.default_tickers
    else:
        tickers = [tickers_list.upper()] if isinstance(tickers_list, str) else tickers_list

    full_data = fetch_backtest_data(start_date, end_date, tickers_list)
    
    if full_data.empty:
        default_summary = {
            'Total_Signals': 0,
            'Win_Rate': 0.0,
            'Avg_PnL': 0.0,
            'Max_Drawdown': 0.0,
            'Sharpe': 0.0
        }
        return pd.DataFrame(), default_summary
    
    # Use actual data dates/bars only - key fix for no results
    dates = sorted(full_data.index.unique())
    
    total_dates = len(dates)

    all_signals = []
    processed_dates = 0
    for date in dates:
        try:
            # Use date as current_date for slicing
            df_day = process_data(date.strftime('%Y-%m-%d'), jw_percent, jw_mode, min_avg_vol, min_rel_vol, min_rvolat, tickers, None, full_data, date)
            if not df_day.empty:
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
                        hit_tp = tp_price is not None and ((mode == 'Bullish' and bar.High >= tp_price) or (mode == 'Bearish' and bar.Low <= tp_price))
                        hit_sl = sl_price is not None and ((mode == 'Bullish' and bar.Low <= sl_price) or (mode == 'Bearish' and bar.High >= sl_price))
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
                    
                    pnl_dollar = dir * (exit_price - entry_c)
                    pnl_percent = (pnl_dollar / entry_c) * 100
                    
                    all_signals.append({
                        'Ticker': ticker,
                        'Signal_Date': date,
                        'Mode': mode,
                        'Entry_Close': entry_c,
                        'PnL_$': pnl_dollar,
                        'PnL_%': pnl_percent,
                        'Days_Held': hit_day,
                        'Outcome': outcome,
                        'TP_Price': tp_price,
                        'SL_Price': sl_price,
                        'rVolume': signal['rVolume'],
                        'Strength': signal['Strength'],
                        'JW_Percent': signal['JW %'],
                        'rVolatility': signal['rVolatility']
                    })
        except Exception as e:
            print(f"Error processing {date}: {e}")
        
        processed_dates += 1
    
    # Always return full summary
    default_summary = {
        'Total_Signals': 0,
        'Win_Rate': 0.0,
        'Avg_PnL': 0.0,
        'Max_Drawdown': 0.0,
        'Sharpe': 0.0
    }
    
    if not all_signals:
        return pd.DataFrame(), default_summary
    
    df_bt = pd.DataFrame(all_signals)
    
    # Compute summary
    if len(df_bt) == 0:
        win_rate = 0.0
        avg_pnl = 0.0
        sharpe = 0.0
        max_dd = 0.0
    else:
        win_rate = (df_bt['Outcome'] == 'Win').mean() * 100
        pnls = df_bt['PnL_%']
        avg_pnl = pnls.mean()
        sharpe = avg_pnl / (pnls.std() + 1e-8)
        cum_returns = np.cumsum(pnls - avg_pnl)
        max_dd = np.min(cum_returns) if len(cum_returns) > 0 else 0.0
    
    summary = {
        'Total_Signals': len(df_bt),
        'Win_Rate': round(win_rate, 1),
        'Avg_PnL': round(avg_pnl, 2),
        'Max_Drawdown': round(max_dd, 2),
        'Sharpe': round(sharpe, 2)
    }
    
    return df_bt, summary

# Template CSV for tickers
template_df = pd.DataFrame({'Ticker': ['AAPL', 'GOOGL']})
template_csv = template_df.to_csv(index=False).encode('utf-8')

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

# Tab structure
tab1, tab2 = st.tabs(["Live Scan", "Backtest Engine"])

with tab1:
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
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Export
            csv = st.session_state.last_df.to_csv(index=False).encode('utf-8')
            st.download_button("EXPORT CSV", csv, "jw_terminal.csv", "text/csv", key="export")

with tab2:
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
        sl_strategy = st.selectbox("Stop Loss", ['None', 'Full Wick', 'Half Wick'], index=0, key="bt_sl")
    with col_bt3:
        bt_jw_percent = st.number_input("JW %", value=20.0, min_value=0.0, step=1.0, key="bt_jw_percent")
        bt_min_rel_vol = st.number_input("Min rVolume", value=0.9, min_value=0.0, step=0.1, key="bt_min_rel_vol")
        bt_min_rvolat = st.number_input("Min rVolatility", value=1.0, min_value=0.0, step=0.1, key="bt_min_rvolat")

    bt_progress = st.empty()

    if st.session_state.backtest_results is None:
        custom_progress(bt_progress, 0, "0%")

    if st.button("Run Backtest", key="run_bt"):
        with st.spinner("Running backtest..."):
            df_bt, summary = backtest_engine(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), bt_tickers, bt_jw_mode, sl_strategy, bt_jw_percent, min_avg_vol=5.0, min_rel_vol=bt_min_rel_vol, min_rvolat=bt_min_rvolat, rr=bt_rr)
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
        default_keys = {'Total_Signals': 0, 'Win_Rate': 0.0, 'Avg_PnL': 0.0, 'Max_Drawdown': 0.0, 'Sharpe': 0.0}
        for key, val in default_keys.items():
            if key not in summary:
                summary[key] = val
        
        if summary['Total_Signals'] == 0:
            st.info("🔍 No John Wicks identified in the backtest period. Try loosening filters (e.g., lower min rVolume) or extending the date range.")
        else:
            # Summary metrics
            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
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

            # Table
            if not df_bt.empty:
                # Rename columns
                df_display = df_bt.rename(columns={'Signal_Date': 'Date', 'JW_Percent': 'JW %', 'Mode': 'JW Signal', 'Entry_Close': 'Close Price', 'Days_Held': 'Days Held', 'Outcome': 'Win/Loss', 'PnL_$': 'PnL $', 'PnL_%': 'PnL %', 'TP_Price': 'Take Profit Target Price', 'SL_Price': 'Stop Loss Price'})
                df_display = df_display.sort_values('Strength', ascending=False)
                styled_bt = style_df(df_display, minimalist)
                st.dataframe(styled_bt, use_container_width=True, hide_index=True)
                
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