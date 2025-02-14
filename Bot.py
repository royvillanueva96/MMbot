# %%
# Configuration & Globals
###############################################################################

import os
import time
import math
import logging
import statistics
import datetime
from collections import deque

import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt

from dotenv import load_dotenv

import asyncio
import nest_asyncio
nest_asyncio.apply()

# For GARCH forecasting
from arch import arch_model

# For the websocket (improved latency + user data stream)
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException

EXCHANGE_ID = "binance"
SYMBOL = "BTC/USDT"

# Base parameters for spread & volatility usage
#BASE_SPREAD = 0.00005   # 0.005%
VOL_MULTIPLIER = 1.0
MAX_VOLATILITY_CAP = 0.15  # Cap the combined volatility at 15%

# Adaptive Notional
BASE_NOTIONAL = 250.0 
MAX_NOTIONAL = 500.0
MIN_NOTIONAL = 100.0

# Inventory & Exposure
MAX_EXPOSURE_BTC = 0.25
TARGET_BTC_EXPOSURE = 0.5 * MAX_EXPOSURE_BTC
EXPOSURE_SPREAD_MULTIPLIER_POSITIVE = 0.8
EXPOSURE_SPREAD_MULTIPLIER_NEGATIVE = 0.2

# Rolling volatility window
WINDOW_SIZE = 20
price_window = deque(maxlen=WINDOW_SIZE + 1)
returns_window = deque(maxlen=WINDOW_SIZE)

# Order Management
LOOP_INTERVAL = 1
STALE_TIME = 60
MAX_TRADE_TIME = 300  # close any open position after 300s (5 min)
MAX_ACTIVE_ORDERS = 20
KILL_SWITCH_THRESHOLD = -15  # -% from initial capital drawdown

# Depth offsets
DEPTH_LEVELS = 5
SPREAD_STEP = 0.0001 # 0.01%

# RSI Levels
RSI_UPPER = 80
RSI_LOWER = 20
RSI_DEPTH_LEVELS = 7
NOTIONAL_MULTIPLIER = 1.5

RISK_FREE_RATE = 0.03 # 3%

# Logging
logging.basicConfig(
    filename="enhanced_market_maker.log",
    filemode="a",
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)

###############################################################################
# Weighted-Average Cost (WAC) & Position Tracking
###############################################################################
net_position_btc = 0.0
net_cost_usdt = 0.0
initial_btc = 0.0
initial_usdt = 0.0
initial_price = 0.0
position_side = "flat"
position_size_btc = 0.0
position_cost_usdt = 0.0
realized_pnl = 0.0
position_open_timestamp = 0.0

###############################################################################
# Full Logging Lists
###############################################################################
trade_log = []                 # logs individual fills (PnL, time, etc.)
spread_log = []                # logs dynamic spread used
unrealized_pnl_log = []        # logs total portfolio unrealized PnL over time
fill_log = []                  # logs fill-level slippage + adverse selection
vol_log = []

# For counting orders
orders_posted = 0
orders_filled = 0
cumulative_traded_BTC = 0
inventory_levels = []

# Track active entry orders in a dictionary; once they fill, we place brackets.
active_entry_orders = {}  # order_id -> dict(side, amount, posted_price, bracket_placed)

###############################################################################
# Stop-Loss / Take-Profit Levels (for net position partial closes)
###############################################################################
STOP_LOSS_LEVELS = [
    (-0.10, 0.5),  # net PnL <= -10% => close half
    (-0.20, 1.0),  # net PnL <= -20% => close all
]
TAKE_PROFIT_LEVELS = [
    (0.2, 0.5),   # net PnL >= +20% => close half
    (0.4, 1.0),   # net PnL >= +40% => close all
]




# %%
# Exchange & WebSocket
###############################################################################
current_price = None
ws_manager = None
websocket_running = False

def initialize_exchange():
    load_dotenv("api.env")

    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')
    if not api_key or not api_secret:
        raise ValueError("Missing API_KEY or API_SECRET in environment variables.")

    exchange = getattr(ccxt, EXCHANGE_ID)({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
    })
    # For testnet usage
    exchange.set_sandbox_mode(True)
    return exchange

exchange = initialize_exchange()

def retry_api_call(func, retries=3, delay=1, *args, **kwargs):
    """
    Call a REST API function (e.g., exchange.fetch_balance) with retries.
    """
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__} (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)
    return None

def get_or_create_event_loop():
    """
    Returns the current asyncio event loop or creates a new one if it is closed.
    This can help avoid scheduling futures after shutdown.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop





# %%
# Slippage & Adverse Selection Reporting
###############################################################################
def record_slippage_and_adverse_selection(side: str,
                                         posted_price: float,
                                         fill_price: float,
                                         fill_amount: float,
                                         fill_timestamp: float):
    """
    Records slippage relative to posted price, defers adverse selection measurement
    until after some lookahead_seconds in 'update_adverse_selection_metrics()'.
    """
    global fill_log

    if fill_amount == 0:
        logging.warning(f"Order not filled, skipping slippage/adverse selection for price {posted_price}")
        return

    if side.lower() == 'buy':
        slippage_signed = fill_price - posted_price
    else:  # 'sell'
        slippage_signed = posted_price - fill_price

    fill_entry = {
        'timestamp': fill_timestamp,
        'side': side.lower(),
        'posted_price': posted_price,
        'fill_price': fill_price,
        'fill_amount': fill_amount,
        'slippage_signed': slippage_signed,
        'adverse_selection': None
    }

    fill_log.append(fill_entry)
    logging.info(f"Slippage recorded => {fill_entry}")
    print(f"Slippage => {slippage_signed:.4f} ({side.upper()}), "
          f"fill_price={fill_price:.4f}, posted={posted_price:.4f}")

def update_adverse_selection_metrics(lookahead_seconds=1):
    """
    After 'lookahead_seconds' have passed for each fill, we measure
    how the price moved vs. the fill price => 'adverse_selection'.
    """
    global fill_log
    current_time = time.time()
    for fill_entry in fill_log:
        if fill_entry['adverse_selection'] is None:
            elapsed = current_time - fill_entry['timestamp']
            if elapsed >= lookahead_seconds:
                try:
                    latest_price = safe_get_current_price()
                    side_factor = 1 if fill_entry['side'] == 'buy' else -1
                    fill_price = fill_entry['fill_price']
                    fill_entry['adverse_selection'] = side_factor * (latest_price - fill_price)

                    logging.info(
                        f"Adverse Selection => fill_time={fill_entry['timestamp']}, "
                        f"fill_price={fill_price:.4f}, current_price={latest_price:.4f}, "
                        f"adverse={fill_entry['adverse_selection']:.4f}"
                    )
                    print(f"Adverse selection => {fill_entry['adverse_selection']:.4f}")
                except Exception as e:
                    logging.error(f"Error updating adverse selection: {e}")

def generate_slippage_adverse_report():
    """
    Writes fill-level data (slippage & adverse selection) to CSV and prints averages.
    """
    if not fill_log:
        print("No fill data for slippage/adverse selection.")
        return

    df_fills = pd.DataFrame(fill_log)
    if df_fills.empty:
        print("No fill entries to report for slippage/adverse selection.")
        return

    avg_slippage = df_fills['slippage_signed'].mean()
    df_adverse = df_fills.dropna(subset=['adverse_selection'])
    avg_adverse = df_adverse['adverse_selection'].mean() if not df_adverse.empty else 0

    df_fills.to_csv("slippage_adverse_log.csv", index=False)
    logging.info(f"Slippage & Adverse => avg_slippage={avg_slippage:.4f}, avg_adverse={avg_adverse:.4f}")
    print(f"Slippage & Adverse => avg_slippage={avg_slippage:.4f}, avg_adverse={avg_adverse:.4f}")





# %%
# Additional Trade / Reporting
###############################################################################
def log_trade(entry_price, exit_price, amount, pnl, pnl_pct, timestamp):
    """
    Optionally log round-trip trades if you do "closed trades" analysis.
    For partial fills, you might prefer the fill-level log in 'trade_log'.
    """
    rt_trade = {
        'Timestamp': timestamp,
        'Entry Price': entry_price,
        'Exit Price': exit_price,
        'Amount': amount,
        'PnL (USDT)': pnl,
        'PnL (%)': pnl_pct
    }
    trade_log.append(rt_trade)
    logging.info(f"Round-Trip Trade => {rt_trade}")
    print(f"Round-Trip => {rt_trade}")

###############################################################################
# Chart Generation & KPI Reporting
###############################################################################

def log_spread(spread):
    spread_entry = {
        'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'Spread (%)': spread * 100
    }
    spread_log.append(spread_entry)
    logging.info(f"Spread => {spread_entry}")
    print(f"Spread => {spread_entry}")

def generate_spread_chart():
    if not spread_log:
        print("No spread data to plot.")
        return
    df_spread = pd.DataFrame(spread_log)
    if df_spread.empty or 'Timestamp' not in df_spread.columns or 'Spread (%)' not in df_spread.columns:
        print("Not enough data to plot spread.")
        return
    df_spread['Timestamp'] = pd.to_datetime(df_spread['Timestamp'], errors='coerce')
    df_spread.dropna(subset=['Timestamp'], inplace=True)
    df_spread.sort_values('Timestamp', inplace=True)

    plt.figure(figsize=(10, 5))
    plt.plot(df_spread['Timestamp'], df_spread['Spread (%)'], label='Dynamic Spread (%)')
    plt.xlabel('Time')
    plt.ylabel('Spread (%)')
    plt.title('Dynamic Spread Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('spread_chart.png')
    plt.close()
    print("Spread chart saved as 'spread_chart.png'.")

def plot_cumulative_portfolio_return(trade_log_local, initial_capital, save_path='realized_pnl_percentage.png'):
    if not trade_log_local:
        print("No trades logged yet, skipping realized PnL plot.")
        return
    df_trades = pd.DataFrame(trade_log_local)
    if df_trades.empty:
        print("No trades data to plot.")
        return
    required_cols = {'PnL (USDT)', 'Timestamp'}
    if not required_cols.issubset(df_trades.columns):
        print("Missing required columns in trade_log for cumulative PnL plot.")
        return

    df_trades['Timestamp'] = pd.to_datetime(df_trades['Timestamp'], errors='coerce')
    df_trades.dropna(subset=['Timestamp'], inplace=True)
    df_trades.sort_values('Timestamp', inplace=True)

    df_trades['Cumulative_PnL_USDT'] = df_trades['PnL (USDT)'].cumsum()
    df_trades['Cumulative_PnL(%)'] = (df_trades['Cumulative_PnL_USDT'] / initial_capital) * 100

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cumulative PnL (%)', color=color)
    ax1.plot(df_trades['Timestamp'], df_trades['Cumulative_PnL(%)'], color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Cumulative PnL (USDT)', color=color)
    ax2.plot(df_trades['Timestamp'], df_trades['Cumulative_PnL_USDT'], color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Cumulative Realized PnL Over Time')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Cumulative Realized PnL chart saved => {save_path}.")

def plot_unrealized_pnl(unrealized_pnl_log_local, save_path='unrealized_pnl_percentage.png'):
    if not unrealized_pnl_log_local:
        print("No unrealized PnL data to plot.")
        return
    df = pd.DataFrame(unrealized_pnl_log_local)
    if df.empty:
        print("No unrealized PnL entries to plot.")
        return
    if 'Timestamp' not in df.columns or 'Unrealized_PnL (USDT)' not in df.columns:
        print("Missing columns in unrealized_pnl_log for plotting.")
        return

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df.dropna(subset=['Timestamp'], inplace=True)
    df.drop_duplicates('Timestamp', inplace=True)
    df.sort_values('Timestamp', inplace=True)

    initial_capital = compute_initial_capital()
    if initial_capital <= 0:
        print("Invalid initial capital for percentage calculation.")
        return

    df['PnL_pct'] = (df['Unrealized_PnL (USDT)'] / initial_capital) * 100

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Unrealized PnL (%)', color=color)
    ax1.plot(df['Timestamp'], df['PnL_pct'], color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Unrealized PnL (USDT)', color=color)
    ax2.plot(df['Timestamp'], df['Unrealized_PnL (USDT)'], color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Unrealized PnL Over Time')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Unrealized PnL chart saved => {save_path}")

def plot_trade_pnl(trade_log_local, save_path='trade_pnl.png'):
    if not trade_log_local:
        print("No trades logged yet, skipping trade PnL plot.")
        return
    df_trades = pd.DataFrame(trade_log_local)
    if df_trades.empty:
        print("No trades data to plot.")
        return
    required_cols = {'PnL (USDT)', 'PnL (%)', 'Timestamp'}
    if not required_cols.issubset(df_trades.columns):
        print("Missing columns in trade_log for trade PnL plot.")
        return

    df_trades['Timestamp'] = pd.to_datetime(df_trades['Timestamp'], errors='coerce')
    df_trades.dropna(subset=['Timestamp'], inplace=True)
    df_trades.sort_values('Timestamp', inplace=True)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('PnL (%)', color=color)
    ax1.plot(df_trades['Timestamp'], df_trades['PnL (%)'], color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('PnL (USDT)', color=color)
    ax2.plot(df_trades['Timestamp'], df_trades['PnL (USDT)'], color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Trade PnL Over Time')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Trade PnL chart saved => {save_path}")


def log_volatility(rolling_vol, garch_vol):
    """
    Logs the current rolling volatility, GARCH volatility, and combined volatility.
    The combined volatility is calculated in the same way as in compute_dynamic_spread and get_adaptive_order_notional.
    """
    if rolling_vol is None and garch_vol is None:
        combined_vol = 0
    elif rolling_vol is None:
        combined_vol = garch_vol
    elif garch_vol is None:
        combined_vol = rolling_vol
    else:
        combined_vol = 0.8 * rolling_vol + 0.2 * garch_vol

    rolling_vol_pct = (rolling_vol * 100) if rolling_vol is not None else 0
    garch_vol_pct   = (garch_vol * 100) if garch_vol is not None else 0
    combined_vol_pct = combined_vol * 100

    vol_entry = {
        'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'rolling_vol_pct': rolling_vol_pct,
        'garch_vol_pct': garch_vol_pct,
        'combined_vol_pct': combined_vol_pct
    }
    vol_log.append(vol_entry)
    logging.info(f"Volatility Log => {vol_entry}")
    print(f"Volatility Log => {vol_entry}")

def generate_volatility_chart():
    """
    Generates a chart plotting rolling volatility, GARCH volatility, and combined volatility over time.
    Saves the chart as 'volatility_chart.png'.
    """
    if not vol_log:
        print("No volatility data to plot.")
        return

    df_vol = pd.DataFrame(vol_log)
    if df_vol.empty:
        print("No volatility entries to plot.")
        return

    # Ensure the timestamp is converted and sorted correctly
    df_vol['Timestamp'] = pd.to_datetime(df_vol['Timestamp'], errors='coerce')
    df_vol.dropna(subset=['Timestamp'], inplace=True)
    df_vol.sort_values('Timestamp', inplace=True)

    plt.figure(figsize=(10, 5))
    plt.plot(df_vol['Timestamp'], df_vol['rolling_vol_pct'], label='Rolling Vol (%)', marker='o')
    plt.plot(df_vol['Timestamp'], df_vol['garch_vol_pct'], label='GARCH Vol (%)', marker='x')
    plt.plot(df_vol['Timestamp'], df_vol['combined_vol_pct'], label='Combined Vol (%)', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.title('Volatility: Rolling, GARCH, and Combined')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('volatility_chart.png')
    plt.close()
    print("Volatility chart saved as 'volatility_chart.png'.")

def generate_kpi_report(current_price):
    """
    Example KPI: fill ratio, average inventory, inventory turnover,
    realized/unrealized PnL, etc.
    """
    global orders_posted, orders_filled, realized_pnl

    current_btc, current_usdt = get_current_balance()
    unrealized_pnl_val, unrealized_pnl_pct = get_unrealized_pnl_overall(current_price, current_btc, current_usdt)
    invested_amount = (initial_usdt + initial_btc * initial_price)
    overall_return = (realized_pnl / invested_amount) * 100 if invested_amount else 0

    fill_ratio = (orders_filled / orders_posted) * 100 if orders_posted > 0 else 0
    avg_inventory = sum(inventory_levels)/len(inventory_levels) if inventory_levels else 0
    inv_turnover = (cumulative_traded_BTC / avg_inventory) if avg_inventory > 0 else 0

    summary = {
        'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"), 
        'RealizedPnL(USDT)': realized_pnl,
        'UnrealizedPnL(USDT)': unrealized_pnl_val,
        'UnrealizedPnL(%)': unrealized_pnl_pct,
        'OverallReturn(%)': overall_return,
        'OrdersPosted': orders_posted,
        'OrdersFilled': orders_filled,
        'FillRatio(%)': fill_ratio,
        'AvgInventory(BTC)': avg_inventory,
        'InventoryTurnover': inv_turnover
    }

    df_summary = pd.DataFrame([summary])
    df_summary.to_csv('kpi_report.csv', mode='a',
                      header=not os.path.exists('kpi_report.csv'), index=False)
    print("KPI report updated.")
    logging.info(f"KPI => {summary}")

def generate_summary_report(current_price):
    global realized_pnl, spread_log, trade_log

    current_btc, current_usdt = get_current_balance()
    unrealized_pnl_val, unrealized_pnl_pct = get_unrealized_pnl_overall(current_price, current_btc, current_usdt)
    invested_amount = initial_usdt + initial_btc * initial_price
    overall_return = (realized_pnl / invested_amount) * 100 if invested_amount != 0 else 0
    initial_capital = compute_initial_capital()
    total_btc_volume, total_usdt_volume = calculate_total_volume()

    # Convert logs to DataFrames
    df_spread = pd.DataFrame(spread_log)
    df_trades = pd.DataFrame(trade_log)
    df_vol    = pd.DataFrame(vol_log)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1) Build a DataFrame from the unrealized_pnl_log
    df_unreal = pd.DataFrame(unrealized_pnl_log)
    if not df_unreal.empty and 'Unrealized_PnL (USDT)' in df_unreal.columns:
        # Sort by time so cummax() works chronologically
        df_unreal['Timestamp'] = pd.to_datetime(df_unreal['Timestamp'], errors='coerce')
        df_unreal.dropna(subset=['Timestamp'], inplace=True)
        df_unreal.sort_values('Timestamp', inplace=True)

        # 2) Convert "Unrealized_PnL (USDT)" to total "Equity" = initial_capital + unrealized_pnl
        df_unreal['Equity'] = initial_capital + df_unreal['Unrealized_PnL (USDT)']

        # 3) Compute a running max of the equity for each time
        df_unreal['RunningMax'] = df_unreal['Equity'].cummax()

        # 4) Calculate the drawdown as a fraction of the running max
        df_unreal['Drawdown'] = (df_unreal['Equity'] - df_unreal['RunningMax']) / df_unreal['RunningMax']

        # 5) The minimum (most negative) drawdown is our worst peak-to-trough drop
        max_unrealized_drawdown = df_unreal['Drawdown'].min()  # negative number (e.g. -0.10 => -10%)

        # For readability, convert to percentage (likely a negative value)
        max_unrealized_drawdown_pct = 100.0 * max_unrealized_drawdown
    else:
        max_unrealized_drawdown_pct = 0.0

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Compute stats from df_trades (already in your code)
    if not df_trades.empty and 'PnL (USDT)' in df_trades.columns and 'Timestamp' in df_trades.columns:
        df_trades['Timestamp'] = pd.to_datetime(df_trades['Timestamp'], errors='coerce')
        df_trades.dropna(subset=['Timestamp'], inplace=True)
        df_trades.sort_values('Timestamp', inplace=True)

        df_trades['Cumulative PnL'] = df_trades['PnL (USDT)'].cumsum()
        df_trades['Cumulative Return (%)'] = (df_trades['Cumulative PnL'] / initial_capital) * 100
        total_trades = len(df_trades)
        winning_trades = df_trades[df_trades['PnL (USDT)'] > 0]
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0

        # Example simplified approach to Sharpe
        returns = df_trades['PnL (USDT)']
        if len(returns) > 1 and returns.std() != 0:
            sharpe_ratio = (returns.mean() - RISK_FREE_RATE) / returns.std()
        else:
            sharpe_ratio = 0

        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() != 0:
            sortino_ratio = ((returns.mean() - RISK_FREE_RATE) / downside_returns.std())
        else:
            sortino_ratio = 0
    else:
        total_trades = 0
        win_rate = 0
        sharpe_ratio = 0
        sortino_ratio = 0

    # Summarize
    summary = {
        'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'Initial BTC': initial_btc,
        'Initial USDT': initial_usdt,
        'Current BTC': current_btc,
        'Current USDT': current_usdt,
        'Realized PnL (USDT)': realized_pnl,
        'Unrealized PnL (USDT)': unrealized_pnl_val,
        'Unrealized PnL (%)': unrealized_pnl_pct,
        'Overall Return (%)': overall_return,
        'Total Trades': total_trades,
        'Win Rate (%)': win_rate,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'MaxUnrealizedDrawdown(%)': max_unrealized_drawdown_pct,
        'Total Volume (BTC)': total_btc_volume,
        'Total Volume (USDT)': total_usdt_volume
    }

    df_summary = pd.DataFrame([summary])
    df_summary.to_csv('summary_report.csv', mode='w', header=True, index=False)

    # Optionally save your other logs
    if not df_spread.empty:
        df_spread.to_csv('spread_log.csv', mode='w', header=True, index=False)
    if not df_trades.empty:
        df_trades.to_csv('trade_log.csv', mode='w', header=True, index=False)
    if not df_vol.empty:
        df_vol.to_csv('vol_log.csv', mode='w', header=True, index=False)

    logging.info("Summary report generated.")
    print("Summary report => 'summary_report.csv' updated.")






# %%
# Handlers (Ticker + User Data)
###############################################################################
best_bid_global = 0.0
best_ask_global = 0.0

def handle_ticker_stream(msg):
    """
    Handler for live ticker data from the websocket (bookTicker).
    We update the global current_price in real-time.
    """
    global current_price, best_bid_global, best_ask_global
    if 'b' in msg and 'a' in msg:
        best_bid = float(msg['b'])
        best_ask = float(msg['a'])
        best_bid_global = best_bid
        best_ask_global = best_ask
        current_price = (best_bid + best_ask) / 2.0

def user_data_handler(msg):
    """
    Handler for user data events: captures real-time fills from Binance.
    We'll detect fill events (executionReports) and parse partial or full fills.
    """
    global orders_filled

    if 'e' not in msg:
        return
    event_type = msg['e']
    if event_type == 'executionReport':
        exec_report = msg
        side = exec_report['S'].lower()    # BUY or SELL
        order_id = str(exec_report['i'])   # order ID
        execution_type = exec_report['x']  # e.g. TRADE if it's a fill
        order_status = exec_report['X']    # FILLED, PARTIALLY_FILLED, CANCELED, etc.

         # Skip slippage recording for orders that aren't filled
        if order_status not in ['FILLED', 'PARTIALLY_FILLED']:
            logging.info(f"Order {order_id} not filled, skipping slippage/adverse reporting.")
            return
        
        filled_qty = float(exec_report['l'])  # last filled trade qty
        fill_price = float(exec_report['L'])  # last fill price
        commission_amount = float(exec_report['n']) if 'n' in exec_report else 0.0

        # 1) KeyError fix: check if this order_id is known
        if order_id not in active_entry_orders:
            logging.warning(f"Unknown order_id={order_id} in user_data_handler. Attempting cancellation.")
            try:
                # Fetch the order details to check its current status
                order_info = exchange.fetch_order(order_id, SYMBOL)
                status = order_info.get('status', '').lower() if order_info else ''
                if status not in ['closed', 'canceled']:
                    # Attempt cancellation if the order is still active
                    if cancel_order_if_open(order_id, SYMBOL):
                        logging.info(f"Successfully canceled unknown order_id={order_id}")
                    else:
                        logging.warning(f"Cancellation attempted but unknown order_id={order_id} might still be active.")
                else:
                        logging.info(f"Unknown order_id={order_id} is already {status}.")
            except Exception as e:
                logging.error(f"Error while attempting to cancel unknown order_id={order_id}: {e}")
            return

        entry_info = active_entry_orders[order_id]

        if execution_type == 'TRADE' and filled_qty > 0:
            # We have a fill event
            logging.info(f"UserData Fill => side={side}, qty={filled_qty}, px={fill_price}")
            print(f"UserData => Fill: side={side}, qty={filled_qty}, px={fill_price:.2f}")

            # 1) Weighted-Average Cost update
            handle_fill(side, fill_price, filled_qty, commission_amount)

            if order_status == 'FILLED' and not entry_info.get('filled', False):
                orders_filled += 1
                entry_info['filled'] = True

             # If you want to track cumulative:
            filled_so_far = entry_info.get('cumulative_filled', 0.0)
            new_cum = filled_so_far + filled_qty
            entry_info['cumulative_filled'] = new_cum  ### FIXED ###

            # 2) Slippage & adverse selection:
            #    If this is from one of our known entry orders, we can record posted_price
        if order_status in ['FILLED' , 'CANCELED']:
            #if order_status == 'FILLED' and not entry_info.get('bracket_placed', False):
             #   manage_bracket_orders(SYMBOL)  # Ensure we don't exceed the algo order limit
              #  final_filled_amount = entry_info.get('cumulative_filled', entry_info['amount'])
               # bracket_ids = place_bracket_orders(
                #    symbol=SYMBOL,
                 #   side=side,
                  #  filled_amount=final_filled_amount,
                   # fill_price=fill_price
                #)

                #entry_info['bracket_ids'] = bracket_ids or []
                #entry_info['bracket_placed'] = True

                if order_id in active_entry_orders:
                    posted_price = active_entry_orders[order_id]['posted_price']
                    record_slippage_and_adverse_selection(
                        side=side,
                        posted_price=posted_price,
                        fill_price=fill_price,
                        fill_amount=filled_qty,
                        fill_timestamp=time.time()
                    )
                
                logging.info(f"Removing order_id={order_id} from active_entry_orders. status={order_status}")
                if order_id in active_entry_orders:
                    del active_entry_orders[order_id]
                    

def start_websocket_stream():
    """
    Initializes and starts the binance ThreadedWebsocketManager.
    Subscribes to:
      1) Book Ticker for real-time best bid/ask
      2) User Data Stream for real-time fill notifications
    """
    global ws_manager, websocket_running
    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')
    if not api_key or not api_secret:
        raise ValueError("Missing API_KEY or API_SECRET for WebSocket usage.")

    ws_manager = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret, testnet=True)
    ws_manager.start()

    # BookTicker => real-time best bid/ask
    sym_no_slash = SYMBOL.replace("/", "")
    ws_manager.start_symbol_ticker_socket(callback=handle_ticker_stream, symbol=sym_no_slash)

    # User Data => real-time fills
    ws_manager.start_user_socket(callback=user_data_handler)

    websocket_running = True
    print("WebSocket streams => real-time ticker + user data started.")

def safe_fetch_balance():
    """
    Wrap the fetch_balance call with retry logic.
    """
    return retry_api_call(exchange.fetch_balance, retries=3, delay=1)

def safe_start_websocket():
    """
    Attempts to start the WebSocket stream with exponential backoff.
    Replace start_websocket_stream() with your existing function that starts the WS.
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            start_websocket_stream()  # your original function that starts the WS
            logging.info("WebSocket started successfully.")
            return True
        except Exception as e:
            logging.error(f"Error starting websocket (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)  # exponential backoff: 1, 2, 4, 8 seconds, etc.
    return False




# %%
# Utility: Balances & Price
###############################################################################
def fetch_current_price_fallback():
    """
    Fallback to REST if WebSocket is unavailable or current_price is None.
    """
    try:
        order_book = exchange.fetch_order_book(SYMBOL)
        best_bid = order_book['bids'][0][0] if order_book['bids'] else None
        best_ask = order_book['asks'][0][0] if order_book['asks'] else None
        if best_bid is None or best_ask is None:
            raise ValueError("Order book is empty or unavailable.")
        return (best_bid + best_ask) / 2.0
    except Exception as e:
        logging.error(f"Error fetching current price via REST: {e}")
        return 0.0

def safe_get_current_price():
    global current_price
    if current_price is not None:
        return current_price
    else:
        return fetch_current_price_fallback()

def get_current_balance():
    try:
        balance = safe_fetch_balance()
        if balance is None:
            return 0.0, 0.0
        current_btc_local = float(balance.get('total', {}).get('BTC', 0.0))
        current_usdt_local = float(balance.get('total', {}).get('USDT', 0.0))
        return current_btc_local, current_usdt_local
    except Exception as e:
        logging.error(f"Error fetching current balance: {e}")
        return 0.0, 0.0

def get_net_btc_position():
    btc, _ = get_current_balance()
    return btc

def get_net_usdt_position():
    _, usdt = get_current_balance()
    return usdt






# %%
# Volatility, GARCH, RSI
###############################################################################
def get_log_return(price_old, price_new):
    if price_old > 0 and price_new > 0:
        return math.log(price_new / price_old)
    return 0.0

def update_rolling_volatility(new_price):
    price_window.append(new_price)
    if len(price_window) > 1:
        r = get_log_return(price_window[-2], price_window[-1])
        returns_window.append(r)
    if len(returns_window) == WINDOW_SIZE:
        return statistics.pstdev(returns_window)
    return None

def forecast_volatility_garch(returns_list):
    if len(returns_list) < WINDOW_SIZE:
        return None
    scaled_returns = [r * 1e5 for r in returns_list]
    sr = pd.Series(scaled_returns).dropna()
    if len(sr) < WINDOW_SIZE:
        return None
    try:
        model = arch_model(sr, vol='Garch', p=1, q=1, dist='normal')
        res = model.fit(disp='off', last_obs=len(sr)-1)
        forecast_res = res.forecast(horizon=1, reindex=False)
        next_var = forecast_res.variance.values[-1, 0]
        stdev_scaled = math.sqrt(next_var)
        stdev_unscaled = stdev_scaled / 1e5
        return stdev_unscaled
    except Exception as e:
        logging.error(f"GARCH forecasting error: {e}")
        return None

def compute_rsi(prices, window=14):
    """
    Compute the RSI for a list of prices using the classic Wilder’s RSI approach.
    Return the most recent RSI value, or None if not enough data.
    """
    if len(prices) < window + 1:
        return None  # not enough data for RSI

    deltas = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
    gains = []
    losses = []
    
    for delta in deltas[-window:]:
        if delta > 0:
            gains.append(delta)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(delta))

    avg_gain = sum(gains) / window
    avg_loss = sum(losses) / window

    if avg_loss == 0:
        return 100.0  # RSI max (strong bullish)

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def adjust_params_for_rsi(rsi_value):
    """
    Example: 
    - If RSI > 70 => bullish, be more aggressive:
        - Increase notional by 50%
        - Increase depth levels to place more orders
    - If RSI < 30 => bearish, also be more aggressive on the short side:
        - For a standard approach, we might also adjust notional up or down
        - Or we reduce the notional if we want to scale back risk
    - Otherwise => normal
    Returns: (adjusted_notional_multiplier, adjusted_depth_levels)
    """
    # Default: no changes
    notional_multiplier = 1.0
    depth_levels = DEPTH_LEVELS  # your existing global

    if rsi_value is None:
        return (notional_multiplier, depth_levels)  # skip if not enough data

    if rsi_value > RSI_UPPER:
        # Bullish => place more orders & bigger notional
        notional_multiplier = NOTIONAL_MULTIPLIER
        depth_levels = RSI_DEPTH_LEVELS   # up from 5
    elif rsi_value < RSI_LOWER:
        # Bearish => you could do the same or invert
        notional_multiplier = NOTIONAL_MULTIPLIER
        depth_levels = RSI_DEPTH_LEVELS
        # Or if you want less risk on the buy side, you could do 0.8
        # depending on your preference.

    return (notional_multiplier, depth_levels)





# %%
# Position Sizing & Spread
###############################################################################

def get_market_spread_fraction():
    """
    Returns the current spread as a fraction of the mid-price:
      spread_fraction = (best_ask - best_bid) / mid_price
    If best_bid or best_ask is invalid, return 0.0
    """
    if best_bid_global <= 0 or best_ask_global <= best_bid_global:
        return 0.0  # abnormal or uninitialized

    mid = (best_ask_global + best_bid_global) / 2.0
    if mid <= 0:
        return 0.0

    raw_spread = best_ask_global - best_bid_global
    spread_fraction = raw_spread / mid
    return spread_fraction


def compute_dynamic_spread(rolling_vol: float, garch_vol: float, current_btc: float) -> float:
    if rolling_vol is None and garch_vol is None:
        combined_vol = 0
    elif rolling_vol is None:
        combined_vol = garch_vol
    elif garch_vol is None:
        combined_vol = rolling_vol
    else:
        combined_vol = 0.8 * rolling_vol + 0.2 * garch_vol

    combined_vol = min(combined_vol, MAX_VOLATILITY_CAP)
    market_spread_fraction = get_market_spread_fraction()
    base = market_spread_fraction + (VOL_MULTIPLIER * combined_vol)

    # Skew if near exposure extremes
    if current_btc > MAX_EXPOSURE_BTC * EXPOSURE_SPREAD_MULTIPLIER_POSITIVE:
        return base * 1.2
    elif current_btc < MAX_EXPOSURE_BTC * EXPOSURE_SPREAD_MULTIPLIER_NEGATIVE:
        return base * 1.2
    else:
        return base

def get_adaptive_order_notional(rolling_vol, garch_vol, current_btc, rsi_value=None):
    if rolling_vol is None and garch_vol is None:
        combined_vol = 0
    elif rolling_vol is None:
        combined_vol = garch_vol
    elif garch_vol is None:
        combined_vol = rolling_vol
    else:
        combined_vol = 0.5 * rolling_vol + 0.5 * garch_vol

    vol_factor = max(1.0 - combined_vol * 10, 0.0)

    raw_notional = BASE_NOTIONAL * vol_factor

    # If RSI is bullish or bearish, we apply an additional multiplier
    rsi_mult = 1.0
    if rsi_value is not None:
        if rsi_value > RSI_UPPER:
            rsi_mult = NOTIONAL_MULTIPLIER
        elif rsi_value < RSI_LOWER:
            rsi_mult = NOTIONAL_MULTIPLIER

    raw_notional *= rsi_mult

    final_notional = min(max(raw_notional, MIN_NOTIONAL), MAX_NOTIONAL)
    return final_notional

def calculate_order_size(notional: float, price: float) -> float:
    return notional / price if price else 0

def within_exposure_limit(current_btc: float, additional: float = 0.0) -> bool:
    return abs(current_btc + additional) <= MAX_EXPOSURE_BTC





# %%
# Orders; Entry and Cancels
###############################################################################
def place_limit_order(symbol: str, side: str, amount: float, price: float):
    global orders_posted
    try:
        order = exchange.create_order(symbol, 'limit', side, amount, price)
        orders_posted += 1
        # Track it so we know to place bracket after fill
        active_entry_orders[order['id']] = {
            'side': side,
            'amount': amount,
            'posted_price': price,
            'filled': False,
            'cumulative_filled': 0.0,
            'placed_timestamp': time.time()
        }
        logging.info(f"Placed {side.upper()} limit => ID={order['id']}, px={price}, amt={amount}")
        print(f"Placed {side.upper()} limit => ID={order['id']}, px={price:.2f}, amt={amount:.4f}")
        return order
    except Exception as e:
        logging.error(f"Error placing {side} limit order: {e}")
        print(f"Error => {e}")
        return None

def cancel_order_if_open(order_id, symbol):
    try:
        order_info = exchange.fetch_order(order_id, symbol)
        if order_info is None:
            return True
        status = order_info.get('status', '').lower()
        if status in ['closed', 'canceled']:
            return True
        exchange.cancel_order(order_id, symbol)
        logging.info(f"Canceled => {order_id}")
        print(f"Canceled => {order_id}")
        return True
    except BinanceAPIException as e:
        if 'Unknown order sent' in str(e):
            return True
        logging.error(f"BinanceAPIException => {e}")
        return False
    except Exception as e:
        logging.error(f"Error canceling order {order_id}: {e}")
        return False

def cancel_stale_orders(stale_time_seconds=STALE_TIME):
    """
    Cancel any orders that have been open for more than `stale_time_seconds`.
    """
    global active_entry_orders

    now = time.time()
    try:
        # Fetch open orders from the exchange to verify status
        open_orders = exchange.fetch_open_orders(SYMBOL)
        open_order_ids = set(o['id'] for o in open_orders)  # IDs from the exchange perspective

        for order_id, entry_info in list(active_entry_orders.items()):
            # If this order is no longer open on the exchange, skip
            # (It might already be FILLED or CANCELED.)
            if order_id not in open_order_ids:
                continue

            placed_ts = entry_info.get('placed_timestamp', 0.0)
            age = now - placed_ts

            if age >= stale_time_seconds:
                logging.info(f"Order {order_id} is stale (age={age:.2f}s), canceling...")
                print(f"Canceling stale order => {order_id} (age={age:.2f}s)")

                # Attempt to cancel
                canceled = cancel_order_if_open(order_id, SYMBOL)
                if canceled:
                    # Clean up from active_entry_orders if needed
                    if order_id in active_entry_orders:
                        del active_entry_orders[order_id]

    except Exception as e:
        logging.error(f"Error in cancel_stale_orders: {e}")
        print(f"Error canceling stale orders: {e}")


def close_stale_position(max_position_time=MAX_TRADE_TIME):
    """
    Force-closes any open position if it has been open longer than `max_position_time` seconds.
    """
    global position_side, position_size_btc, position_cost_usdt
    global position_open_timestamp

    if position_side == "flat":
        return  # No open position to close

    now = time.time()
    position_age = now - position_open_timestamp

    if position_age >= max_position_time:
        # Time-based exit triggered
        logging.info(f"Time-based exit => position side={position_side}, size={position_size_btc}, age={position_age:.2f}s")
        print(f"Time-based exit => side={position_side}, size={position_size_btc:.4f}, age={position_age:.2f}s")

        try:
            # Flatten the entire position using a market order
            if position_side == "long":
                exchange.create_market_sell_order(SYMBOL, position_size_btc)
            elif position_side == "short":
                exchange.create_market_buy_order(SYMBOL, position_size_btc)

            # Reset position tracking
            position_side = "flat"
            position_size_btc = 0.0
            position_cost_usdt = 0.0
            position_open_timestamp = 0.0

        except Exception as e:
            logging.error(f"Error closing stale position: {e}")
            print(f"Error closing stale position: {e}")



def cancel_all_open_orders(symbol):
    """
    Cancels all open orders for the specified symbol.
    """
    try:
        open_orders = exchange.fetch_open_orders(symbol)
        for order in open_orders:
            order_id = order['id']
            print(f"Canceling open order => {order_id}")
            cancel_order_if_open(order_id, symbol)
        print("All open orders canceled successfully.")
    except Exception as e:
        logging.error(f"Error canceling all open orders: {e}")
        print(f"Error canceling all open orders: {e}")





# %%
# Weighted Avg Cost & Realized PnL

def update_position_from_fill(fill_side, fill_price, fill_amount, fee=0.0):
    global realized_pnl, position_side, position_size_btc, position_cost_usdt, entry_price, position_open_timestamp

    if fill_amount == 0:
        logging.warning(f"Order not filled, skipping position update and PnL calculation.")
        return

    # 1) If position_side is "flat", we simply open a new position:
    if position_side == "flat":
        if fill_side == "buy":
            position_side = "long"
            position_size_btc = fill_amount
            position_cost_usdt = fill_price * fill_amount
            # Record entry price when opening position
            entry_price = fill_price
            position_open_timestamp = time.time() # Record entry timestamp
            logging.info(f"Opened long position at entry price {entry_price:.2f}")
        else:  # fill_side == "sell"
            position_side = "short"
            position_size_btc = fill_amount
            position_cost_usdt = fill_price * fill_amount
            # Record entry price when opening position
            entry_price = fill_price
            position_open_timestamp = time.time() # Record entry timestamp
            logging.info(f"Opened short position at entry price {entry_price:.2f}")
        realized_pnl -= fee
        return

    # 2) If position is already open:
    if position_side == "long" and fill_side == "sell":
        # Exit long position, calculate exit PnL
        exit_price = fill_price
        avg_cost_long = position_cost_usdt / position_size_btc
        partial_pnl = (exit_price - avg_cost_long) * fill_amount
        realized_pnl += partial_pnl
        logging.info(f"Closed long position at exit price {exit_price:.2f}, PnL={partial_pnl:.2f} USDT")
        position_side = "flat"  # Position is now flat

    elif position_side == "short" and fill_side == "buy":
        # Exit short position, calculate exit PnL
        exit_price = fill_price
        avg_cost_short = position_cost_usdt / position_size_btc
        partial_pnl = (avg_cost_short - exit_price) * fill_amount
        realized_pnl += partial_pnl
        logging.info(f"Closed short position at exit price {exit_price:.2f}, PnL={partial_pnl:.2f} USDT")
        position_side = "flat"  # Position is now flat

    # Handle fee
    realized_pnl -= fee


def handle_fill(side, fill_price, fill_amount, fee):
    global realized_pnl, cumulative_traded_BTC

    cumulative_traded_BTC += fill_amount
    old_realized = realized_pnl
    update_position_from_fill(side, fill_price, fill_amount, fee)
    fill_realized = realized_pnl - old_realized

    if fill_realized != 0:  # Only log trades with non-zero PnL
        fill_pnl_pct = 0.0
        if fill_price > 0 and fill_amount > 0:
            fill_pnl_pct = (fill_realized / (fill_price * fill_amount)) * 100

        # If position was closed (exit price)
        if position_side == "flat":
            # Record entry and exit prices
            entry_exit_log = {
                'Entry Price': entry_price if 'entry_price' in globals() else None,
                'Exit Price': fill_price,
                'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'Side': side,
                'Price': fill_price,
                'Amount': fill_amount,
                'PnL (USDT)': fill_realized,
                'PnL (%)': fill_pnl_pct
            }
            trade_log.append(entry_exit_log)

        logging.info(f"Trade Log => {side.upper()} at {fill_price:.2f}, PnL={fill_realized:.2f} USDT")
        print(f"Trade Log => {side.upper()} at {fill_price:.2f}, PnL={fill_realized:.2f} USDT")


# %%
# PnL, Inventory, Volume
###############################################################################
def get_initial_balance():
    global initial_btc, initial_usdt
    try:
        balance = exchange.fetch_balance()
        initial_btc = float(balance.get('total', {}).get('BTC', 0.0))
        initial_usdt = float(balance.get('total', {}).get('USDT', 0.0))
        return True
    except Exception as e:
        logging.error(f"Error fetching initial balance: {e}")
        return False

def compute_initial_capital():
    return initial_usdt + (initial_btc * initial_price)

def get_position_pnl_pct(curr_price: float):
    """
    Computes the percentage of unrealized profit/loss based on the current position.
    Uses the new tracking variables: position_side, position_size_btc, and position_cost_usdt.
    Returns 0 if flat or if cost basis is zero.
    """
    # If flat or no size, there's no position to compute
    if position_side == "flat" or position_size_btc < 1e-8:
        return 0.0

    # Calculate the weighted average cost (average entry price)
    avg_cost = position_cost_usdt / position_size_btc

    # For a long, profit is when current price exceeds avg cost.
    # For a short, profit is when current price is lower than avg cost.
    if position_side == "long":
        net_unreal = (curr_price - avg_cost) * position_size_btc
    else:  # position_side == "short"
        net_unreal = (avg_cost - curr_price) * position_size_btc

    cost_basis = avg_cost * position_size_btc
    if cost_basis == 0:
        return 0.0

    pnl_pct = net_unreal / cost_basis
    return pnl_pct

def get_unrealized_pnl_overall(curr_price, curr_btc, curr_usdt):
    invested_amount = (initial_usdt + initial_btc * initial_price)
    current_value = (curr_btc * curr_price) + curr_usdt
    unreal_pnl = current_value - invested_amount
    unreal_pnl_pct = (unreal_pnl / invested_amount) * 100 if invested_amount != 0 else 0.0
    return unreal_pnl, unreal_pnl_pct

def print_pnl():
    cp = safe_get_current_price()
    cb_btc, cb_usdt = get_current_balance()
    unreal_pnl, unreal_pnl_pct = get_unrealized_pnl_overall(cp, cb_btc, cb_usdt)
    invested_amount = compute_initial_capital()
    overall_return = (realized_pnl / invested_amount) * 100 if invested_amount != 0 else 0
    print(f"Realized PnL: {realized_pnl:.2f} USDT | "
          f"Unrealized: {unreal_pnl:.2f} USDT ({unreal_pnl_pct:.2f}%) | "
          f"Overall Return: {overall_return:.2f}%")
    logging.info(f"PnL => Realized: {realized_pnl:.2f}, Unreal: {unreal_pnl:.2f} ({unreal_pnl_pct:.2f}%), Return: {overall_return:.2f}%")

def log_unrealized_pnl():
    cp = safe_get_current_price()
    cb_btc, cb_usdt = get_current_balance()
    unreal_pnl, _ = get_unrealized_pnl_overall(cp, cb_btc, cb_usdt)
    unrealized_pnl_log.append({
        'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'Unrealized_PnL (USDT)': unreal_pnl
    })



# Inventory & Exposure Management
###############################################################################
def adjust_exposure_if_needed(current_btc):
    threshold = 0.8 * MAX_EXPOSURE_BTC
    if current_btc < -threshold:
        reduce_amount = abs(current_btc) - 0.5 * MAX_EXPOSURE_BTC
        if reduce_amount > 0:
            try:
                print(f"Exposure too low(short) => Flatten {reduce_amount:.4f} via market buy.")
                exchange.create_market_buy_order(SYMBOL, reduce_amount)
            except Exception as e:
                logging.error(f"Error flattening short: {e}")
        return

    if current_btc > threshold:
        reduce_amount = current_btc - 0.5 * MAX_EXPOSURE_BTC
        if reduce_amount > 0:
            try:
                print(f"Exposure too high => Flatten {reduce_amount:.4f} via market sell.")
                exchange.create_market_sell_order(SYMBOL, reduce_amount)
            except Exception as e:
                logging.error(f"Error flattening: {e}")
        return

    low_threshold = 0.1 * MAX_EXPOSURE_BTC
    if current_btc < low_threshold:
        buy_amount = TARGET_BTC_EXPOSURE - current_btc
        if buy_amount > 0:
            try:
                print(f"Inventory too low => Market-buy {buy_amount:.4f}.")
                exchange.create_market_buy_order(SYMBOL, buy_amount)
            except Exception as e:
                logging.error(f"Error low inventory fix: {e}")

    high_threshold = 0.9 * MAX_EXPOSURE_BTC
    if current_btc > high_threshold:
        sell_amount = current_btc - TARGET_BTC_EXPOSURE
        if sell_amount > 0:
            try:
                print(f"Inventory too high => Market-sell {sell_amount:.4f}.")
                exchange.create_market_sell_order(SYMBOL, sell_amount)
            except Exception as e:
                logging.error(f"Error high inventory fix: {e}")

# Volume
###############################################################################
def calculate_total_volume():
    """
    Returns:
      total_btc_volume    = sum of all fill amounts in BTC
      total_usdt_volume   = sum of (fill_amount * fill_price) across all fills
    """
    if not fill_log:
        return 0.0, 0.0

    df_fills = pd.DataFrame(fill_log)
    if df_fills.empty:
        return 0.0, 0.0

    # Each fill's USDT notional is fill_amount * fill_price
    df_fills['Notional'] = df_fills['fill_amount'] * df_fills['fill_price']

    total_btc_volume = df_fills['fill_amount'].sum()
    total_usdt_volume = df_fills['Notional'].sum()
    
    return total_btc_volume, total_usdt_volume



            



# %%
# Partial SL/ TP & Depth Orders 
###############################################################################
def check_stop_loss_take_profit():
    global position_side, position_size_btc, position_cost_usdt
    # If we are "flat", there's no position
    if position_side == "flat":
        return

    cp = safe_get_current_price()
    # Compute unrealized PnL % from the new approach
    net_unreal = 0.0
    avg_cost = position_cost_usdt / position_size_btc
    if position_side == "long":
        net_unreal = (cp - avg_cost) * position_size_btc
    else:  # "short"
        net_unreal = (avg_cost - cp) * position_size_btc

    cost_basis = avg_cost * position_size_btc
    net_pnl_pct = (net_unreal / cost_basis) if cost_basis != 0 else 0.0

    # Stop-loss
    for (threshold_pnl, fraction_to_close) in STOP_LOSS_LEVELS:
        if net_pnl_pct <= threshold_pnl:
            qty_to_close = abs(position_size_btc) * fraction_to_close
            print(f"Stop-Loss => net PnL%={net_pnl_pct*100:.2f}%, close fraction={fraction_to_close}")
            try:
                open_orders = exchange.fetch_open_orders(SYMBOL)
                for o in open_orders:
                    cancel_order_if_open(o['id'], SYMBOL)
            except:
                pass
            try:
                if position_side == "long":
                    exchange.create_market_sell_order(SYMBOL, qty_to_close)
                elif position_side == "short":
                    exchange.create_market_buy_order(SYMBOL, qty_to_close)
            except Exception as e:
                logging.error(f"Error partial SL close: {e}")
            return

    # Take-profit
    for (threshold_pnl, fraction_to_close) in TAKE_PROFIT_LEVELS:
        if net_pnl_pct >= threshold_pnl:
            qty_to_close = abs(position_size_btc) * fraction_to_close
            print(f"Take-Profit => net PnL%={net_pnl_pct*100:.2f}%, close fraction={fraction_to_close}")
            try:
                open_orders = exchange.fetch_open_orders(SYMBOL)
                for o in open_orders:
                    cancel_order_if_open(o['id'], SYMBOL)
            except:
                pass
            try:
                if position_side == "long":
                    exchange.create_market_sell_order(SYMBOL, qty_to_close)
                elif position_side == "short":
                    exchange.create_market_buy_order(SYMBOL, qty_to_close)
            except Exception as e:
                logging.error(f"Error partial TP close: {e}")
            return


# Depth Orders 
###############################################################################
def place_depth_orders(
    mid_price: float,
    base_spread: float,
    notional: float,
    current_btc: float,
    depth_levels: int
):
    """
    Places layered (depth) limit orders on both buy and sell sides.
      - `mid_price`: the reference (mid) price
      - `base_spread`: baseline spread for level 1
      - `notional`: how many USDT we want to trade per order
      - `current_btc`: how many BTC we currently hold
      - `depth_levels`: how many levels to place in each direction
    """
    try:
        open_orders = exchange.fetch_open_orders(SYMBOL)
    except Exception as e:
        logging.error(f"Error fetching open orders: {e}")
        open_orders = []

    open_orders_count = len(open_orders)

    for level in range(1, depth_levels + 1):
        buy_spread = base_spread + (level * SPREAD_STEP)
        buy_price = mid_price * (1 - buy_spread / 2)
        buy_amount = calculate_order_size(notional, buy_price)

        if open_orders_count < MAX_ACTIVE_ORDERS:
            if within_exposure_limit(current_btc, buy_amount):
                place_limit_order(SYMBOL, 'buy', buy_amount, buy_price)
                open_orders_count += 1

        sell_spread = base_spread + (level * SPREAD_STEP)
        sell_price = mid_price * (1 + sell_spread / 2)
        sell_amount = calculate_order_size(notional, sell_price)

        if open_orders_count < MAX_ACTIVE_ORDERS:
            if within_exposure_limit(current_btc, -sell_amount):
                place_limit_order(SYMBOL, 'sell', sell_amount, sell_price)
                open_orders_count += 1



# %%
# Main Loop
###############################################################################
def run_market_maker():
    print("Starting Market Maker with Full Reporting, User Data Fills, and Brackets!")
    logging.info("Market Maker Started.")

    # A) Get initial balances
    success = get_initial_balance()
    if not success:
        print("Could not fetch initial balance, aborting.")
        return

    # Fallback price in case we don't have WebSocket yet
    fallback_price = fetch_current_price_fallback()
    global initial_price
    initial_price = fallback_price if fallback_price > 0 else 20000

    # Possibly update initial_price from the live ticker if available
    cp = safe_get_current_price()
    if cp and cp > 0:
        initial_price = cp

    initial_capital = compute_initial_capital()
    print(f"Initial => BTC={initial_btc:.4f}, USDT={initial_usdt:.2f}, Price={initial_price:.2f}, "
          f"Capital={initial_capital:.2f}")
    logging.info(f"Initial => BTC={initial_btc:.4f}, USDT={initial_usdt:.2f}, Price={initial_price:.2f}, "
          f"Capital={initial_capital:.2f}")

    last_report_time = datetime.datetime.now()
    REPORT_INTERVAL = 60
    adverse_selection_lookahead = 1

    cancel_interval = STALE_TIME # check every 60 seconds
    last_time_check = time.time()

    while True:
        try:
            # 1) Update adverse selection
            update_adverse_selection_metrics(lookahead_seconds=adverse_selection_lookahead)

            # 2) Current price & volatility
            mp = safe_get_current_price()
            rolling_vol = update_rolling_volatility(mp)
            garch_vol = forecast_volatility_garch(returns_window)

            # Compute RSI
            rsi_value = compute_rsi(list(price_window), window=14)
            if rsi_value is not None:
                print(f"RSI => {rsi_value:.2f}")
                logging.info(f"RSI => {rsi_value:.2f}")


            # Log the volatility data
            log_volatility(rolling_vol, garch_vol)

            # 3) Net positions
            current_btc_exchange = get_net_btc_position()
            current_usdt_exchange = get_net_usdt_position()
            inventory_levels.append(abs(current_btc_exchange))

            # 4) Adjust exposure
            adjust_exposure_if_needed(current_btc_exchange)

            # 5) Dynamic spread & place depth orders
            dynamic_spread = compute_dynamic_spread(rolling_vol, garch_vol, current_btc_exchange)
            log_spread(dynamic_spread)

            adaptive_notional = get_adaptive_order_notional(rolling_vol, garch_vol, current_btc_exchange,rsi_value)

            # Decide how many levels we want
            if rsi_value is not None and (rsi_value > RSI_UPPER or rsi_value < RSI_LOWER):
                rsi_depth_levels = RSI_DEPTH_LEVELS
            else:
                rsi_depth_levels = DEPTH_LEVELS

            place_depth_orders(mp, dynamic_spread, adaptive_notional, current_btc_exchange, rsi_depth_levels)

            # 6) Print PnL & log
            print_pnl()
            log_unrealized_pnl()

            # 7) Partial stop-loss / take-profit on net position
            check_stop_loss_take_profit()

            # Periodically cancel stale orders
            if time.time() - last_time_check >= cancel_interval:
                cancel_stale_orders(stale_time_seconds=STALE_TIME)
            # Time-based exit on the position:
                close_stale_position(max_position_time=MAX_TRADE_TIME)
                last_time_check = time.time()

            # 8) Kill switch check
            unreal_portfolio, unreal_portfolio_pct = get_unrealized_pnl_overall(
                mp, current_btc_exchange, current_usdt_exchange
            )
            if unreal_portfolio_pct <= KILL_SWITCH_THRESHOLD:
                print("Kill switch => Big drawdown, flattening positions.")
                logging.warning("Kill switch => Flattening all positions.")
                try:
                    open_orders = exchange.fetch_open_orders(SYMBOL)
                    for o in open_orders:
                        cancel_order_if_open(o['id'], SYMBOL)
                except:
                    pass
                # flatten
                if current_btc_exchange > 0:
                    exchange.create_market_sell_order(SYMBOL, current_btc_exchange)
                elif current_btc_exchange < 0:
                    exchange.create_market_buy_order(SYMBOL, abs(current_btc_exchange))
                break

            # 9) Periodic reporting (plots, CSV logs)
            current_time = datetime.datetime.now()
            if (current_time - last_report_time).total_seconds() >= REPORT_INTERVAL:
                generate_spread_chart()
                plot_cumulative_portfolio_return(trade_log, initial_capital)
                plot_unrealized_pnl(unrealized_pnl_log)
                plot_trade_pnl(trade_log)
                generate_kpi_report(mp)
                generate_summary_report(mp)
                generate_slippage_adverse_report()
                generate_volatility_chart()

                last_report_time = current_time

        except Exception as e:
            logging.error(f"Main loop error: {e}")
            print(f"Main loop error: {e}") 

        time.sleep(LOOP_INTERVAL)

def main():

    loop = get_or_create_event_loop()
    # Start the WebSocket safely with reconnection logic.
    if not safe_start_websocket():
        logging.error("Failed to start websocket after maximum retries. Exiting.")
        return
    
    cancel_all_open_orders(SYMBOL)

    run_market_maker()

if __name__ == "__main__":
    main()



