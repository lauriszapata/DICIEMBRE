"""
BACKTEST SIMPLIFICADO - ROI FIJO + COOLDOWN
=============================================
Versión optimizada que pre-descarga todos los datos

CARACTERÍSTICAS:
- SL/TP ROI fijo: 5% y 10%
- Cooldown 1 hora post-pérdida
- Filtros multi-timeframe (15min, 5min, 1min)
- Pre-descarga de datos (más rápido)
- Límite 2 trades/hora
- Filtro lateral
- ADX 30
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuración
CONFIG = {
    'MARGIN_USD': 100,
    'LEVERAGE': 10,
    'MAX_OPEN_POSITIONS': 3,
    'SL_ROI': 0.05,
    'TP_ROI': 0.10,
    'ADX_MIN': 30,
    'RSI_LONG_MIN': 55,
    'RSI_SHORT_MAX': 70,
    'VOLUME_RATIO': 1.2,
    'EMA_EXTENSION_ATR_MULT': 3.0,
    'ATR_MIN_PCT': 0.002,
    'ATR_MAX_PCT': 0.15,
    'PIVOT_LOOKBACK': 50,
    'COOLDOWN_HOURS': 1,
    'MAX_TRADES_PER_HOUR': 2,
    'LATERAL_RANGE_PCT': 0.015,
    'COMMISSION_PCT': 0.0005,
}

SYMBOLS = [
    'DOGE/USDT', 'OP/USDT', 'ATOM/USDT', 'FIL/USDT', 'ADA/USDT',
    'TRX/USDT', 'DOT/USDT', 'LINK/USDT', 'ARB/USDT', 'APT/USDT'
]

print("=" * 80)
print("BACKTEST SIMPLIFICADO - NOVIEMBRE 2025")
print("=" * 80)
print("\nDescargando datos de todos los timeframes al inicio...\n")

exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})

start_data = datetime(2025, 10, 1)
start_backtest = datetime(2025, 11, 1)
end_backtest = datetime(2025, 11, 30, 23, 59, 59)

since_ms = int(start_data.timestamp() * 1000)
until_ms = int(end_backtest.timestamp() * 1000)

# PRE-DESCARGA DE TODOS LOS DATOS
all_data = {}

for symbol in SYMBOLS:
    print(f"Descargando {symbol}...")
    all_data[symbol] = {}
    
    for tf in ['1h', '15m', '5m', '1m']:
        try:
            ohlcv = []
            current = since_ms
            
            while current < until_ms:
                data = exchange.fetch_ohlcv(symbol, tf, since=current, limit=1000)
                if not data:
                    break
                ohlcv.extend(data)
                current = data[-1][0] + 1
                if len(data) < 1000:
                    break
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            df = df[df['timestamp'] < pd.to_datetime(until_ms, unit='ms')]
            
            # Calcular EMAs para este timeframe
            df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
            df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
            
            all_data[symbol][tf] = df
            print(f"  {tf}: {len(df)} velas")
            
        except Exception as e:
            print(f"  Error {tf}: {e}")
            all_data[symbol][tf] = pd.DataFrame()

print("\n✅ Descarga completa\n")
print("=" * 80)

# Calcular indicadores para 1h
def calc_indicators(df):
    if len(df) < 60:
        return df
    
    df = df.copy()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14, adjust=False).mean()
    
    plus_dm = high.diff().where((high.diff() > (-low.diff())) & (high.diff() > 0), 0)
    minus_dm = (-low.diff()).where(((-low.diff()) > high.diff()) & ((-low.diff()) > 0), 0)
    atr_adx = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_adx)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_adx)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.ewm(span=14, adjust=False).mean()
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    df['macd_hist'] = (ema_fast - ema_slow) - (ema_fast - ema_slow).ewm(span=9, adjust=False).mean()
    
    df['vol_sma'] = df['volume'].rolling(window=20).mean()
    df['atr_pct'] = df['atr'] / df['close']
    df['ema20_dist_atr'] = abs(df['close'] - df['ema20']) / df['atr']
    
    return df

for symbol in SYMBOLS:
    if '1h' in all_data[symbol]:
        all_data[symbol]['1h'] = calc_indicators(all_data[symbol]['1h'])

# Funciones de filtros
def detect_hl(df, idx):
    if idx < 53:
        return False
    pivot_idx = idx - 2
    pivot_low = df['low'].iloc[pivot_idx]
    if df['low'].iloc[pivot_idx - 1] <= pivot_low or df['low'].iloc[pivot_idx + 1] <= pivot_low:
        return False
    for i in range(pivot_idx - 3, max(0, pivot_idx - 50), -1):
        if i <= 0 or i >= len(df) - 1:
            continue
        prev_low = df['low'].iloc[i]
        if df['low'].iloc[i-1] > prev_low and df['low'].iloc[i+1] > prev_low:
            return pivot_low > prev_low
    return False

def detect_lh(df, idx):
    if idx < 53:
        return False
    pivot_idx = idx - 2
    pivot_high = df['high'].iloc[pivot_idx]
    if df['high'].iloc[pivot_idx - 1] >= pivot_high or df['high'].iloc[pivot_idx + 1] >= pivot_high:
        return False
    for i in range(pivot_idx - 3, max(0, pivot_idx - 50), -1):
        if i <= 0 or i >= len(df) - 1:
            continue
        prev_high = df['high'].iloc[i]
        if df['high'].iloc[i-1] < prev_high and df['high'].iloc[i+1] < prev_high:
            return pivot_high < prev_high
    return False

def check_15m_filter(symbol, current_time, direction):
    df = all_data[symbol]['15m']
    if df.empty or len(df) < 50:
        return False
    
    df_before = df[df['timestamp'] < current_time]
    if len(df_before) < 5:
        return False
    
    is_long = direction == 'LONG'
    consecutive = 0
    
    for i in range(-5, -1):
        ema8 = df_before['ema8'].iloc[i]
        ema21 = df_before['ema21'].iloc[i]
        if (is_long and ema8 > ema21) or (not is_long and ema8 < ema21):
            consecutive += 1
        else:
            consecutive = 0
    
    current_close = df_before['close'].iloc[-2]
    ema8_current = df_before['ema8'].iloc[-2]
    price_ok = (current_close > ema8_current) if is_long else (current_close < ema8_current)
    
    return consecutive >= 3 and price_ok

def check_5m_filter(symbol, current_time, direction):
    df = all_data[symbol]['5m']
    if df.empty or len(df) < 50:
        return False
    
    df_before = df[df['timestamp'] < current_time]
    if len(df_before) < 8:
        return False
    
    is_long = direction == 'LONG'
    checks = 0
    
    bullish = sum(1 for i in range(-4, -1) if df_before['close'].iloc[i] > df_before['open'].iloc[i])
    bearish = sum(1 for i in range(-4, -1) if df_before['close'].iloc[i] < df_before['open'].iloc[i])
    if (is_long and bullish >= 2) or (not is_long and bearish >= 2):
        checks += 1
    
    ema8 = df_before['ema8'].iloc[-2]
    ema21 = df_before['ema21'].iloc[-2]
    if (is_long and ema8 > ema21) or (not is_long and ema8 < ema21):
        checks += 1
    
    slope = ema8 - df_before['ema8'].iloc[-5]
    if (is_long and slope > 0) or (not is_long and slope < 0):
        checks += 1
    
    current_close = df_before['close'].iloc[-2]
    if (is_long and current_close > ema21) or (not is_long and current_close < ema21):
        checks += 1
    
    gap_current = abs(ema8 - ema21)
    gap_prev = abs(df_before['ema8'].iloc[-5] - df_before['ema21'].iloc[-5])
    if gap_current > gap_prev:
        checks += 1
    
    return checks >= 3

def check_1m_filter(symbol, current_time, direction):
    df = all_data[symbol]['1m']
    if df.empty or len(df) < 30:
        return False
    
    df_before = df[df['timestamp'] < current_time]
    if len(df_before) < 10:
        return False
    
    is_long = direction == 'LONG'
    checks = 0
    
    bullish = sum(1 for i in range(-4, -1) if df_before['close'].iloc[i] > df_before['open'].iloc[i])
    bearish = sum(1 for i in range(-4, -1) if df_before['close'].iloc[i] < df_before['open'].iloc[i])
    if (is_long and bullish >= 2) or (not is_long and bearish >= 2):
        checks += 1
    
    ema8 = df_before['ema8'].iloc[-2]
    ema21 = df_before['ema21'].iloc[-2]
    if (is_long and ema8 > ema21) or (not is_long and ema8 < ema21):
        checks += 1
    
    slope = ema8 - df_before['ema8'].iloc[-5]
    if (is_long and slope > 0) or (not is_long and slope < 0):
        checks += 1
    
    current_close = df_before['close'].iloc[-2]
    if (is_long and current_close > ema8) or (not is_long and current_close < ema8):
        checks += 1
    
    momentum = current_close - df_before['close'].iloc[-7]
    if (is_long and momentum > 0) or (not is_long and momentum < 0):
        checks += 1
    
    return checks >= 4

# Estado del backtest
cooldowns = {}
trades_hourly = {}

def is_in_cooldown(symbol, time):
    if symbol not in cooldowns:
        return False
    if time >= cooldowns[symbol] + timedelta(hours=1):
        del cooldowns[symbol]
        return False
    return True

def check_hourly_limit(symbol, time):
    if symbol not in trades_hourly:
        trades_hourly[symbol] = []
    one_hour_ago = time - timedelta(hours=1)
    trades_hourly[symbol] = [t for t in trades_hourly[symbol] if t > one_hour_ago]
    return len(trades_hourly[symbol]) < 2

def add_trade(symbol, time):
    if symbol not in trades_hourly:
        trades_hourly[symbol] = []
    trades_hourly[symbol].append(time)

# Buscar señales
print("Buscando señales...\n")
all_signals = []

for symbol in SYMBOLS:
    df = all_data[symbol]['1h']
    if df.empty:
        continue
    
    nov_start = df[df['timestamp'] >= start_backtest].index[0] if any(df['timestamp'] >= start_backtest) else len(df)
    
    for idx in range(nov_start, len(df)):
        row = df.iloc[idx]
        current_time = row['timestamp']
        
        if idx < 60:
            continue
        
        if is_in_cooldown(symbol, current_time) or not check_hourly_limit(symbol, current_time):
            continue
        
        # Filtro lateral
        if idx >= 4:
            recent = df.iloc[idx-4:idx]
            range_pct = (recent['high'].max() - recent['low'].min()) / recent['close'].mean()
            if range_pct < CONFIG['LATERAL_RANGE_PCT']:
                continue
        
        # Condiciones 1h
        conds_base = (
            CONFIG['ATR_MIN_PCT'] <= row['atr_pct'] <= CONFIG['ATR_MAX_PCT'] and
            row['adx'] >= CONFIG['ADX_MIN'] and
            row['volume'] >= CONFIG['VOLUME_RATIO'] * row['vol_sma'] and
            row['ema20_dist_atr'] < CONFIG['EMA_EXTENSION_ATR_MULT']
        )
        
        if not conds_base:
            continue
        
        # LONG
        if (row['ema8'] > row['ema21'] and row['close'] > row['ema50'] and row['ema20'] > row['ema50'] and
            row['rsi'] > CONFIG['RSI_LONG_MIN'] and row['macd_hist'] > 0 and detect_hl(df, idx)):
            
            if (check_15m_filter(symbol, current_time, 'LONG') and
                check_5m_filter(symbol, current_time, 'LONG') and
                check_1m_filter(symbol, current_time, 'LONG')):
                
                all_signals.append({
                    'timestamp': current_time,
                    'symbol': symbol,
                    'direction': 'LONG',
                    'idx': idx,
                    'entry_price': row['close']
                })
        
        # SHORT
        if (row['ema8'] < row['ema21'] and row['close'] < row['ema50'] and row['ema20'] < row['ema50'] and
            row['rsi'] < CONFIG['RSI_SHORT_MAX'] and row['macd_hist'] < 0 and detect_lh(df, idx)):
            
            if (check_15m_filter(symbol, current_time, 'SHORT') and
                check_5m_filter(symbol, current_time, 'SHORT') and
                check_1m_filter(symbol, current_time, 'SHORT')):
                
                all_signals.append({
                    'timestamp': current_time,
                    'symbol': symbol,
                    'direction': 'SHORT',
                    'idx': idx,
                    'entry_price': row['close']
                })

all_signals.sort(key=lambda x: x['timestamp'])
print(f"✅ {len(all_signals)} señales encontradas\n")
print("=" * 80)

# Ejecutar trades
print("\nEjecutando trades...\n")
all_trades = []
open_positions = []

for signal in all_signals:
    open_positions = [t for t in open_positions if t > signal['timestamp']]
    if len(open_positions) >= 3:
        continue
    
    symbol = signal['symbol']
    df = all_data[symbol]['1h']
    idx = signal['idx']
    direction = signal['direction']
    entry_price = signal['entry_price']
    
    margin = CONFIG['MARGIN_USD']
    leverage = CONFIG['LEVERAGE']
    exposure = margin * leverage
    
    sl_move = CONFIG['SL_ROI'] / leverage
    tp_move = CONFIG['TP_ROI'] / leverage
    
    if direction == 'LONG':
        sl_price = entry_price * (1 - sl_move)
        tp_price = entry_price * (1 + tp_move)
    else:
        sl_price = entry_price * (1 + sl_move)
        tp_price = entry_price * (1 - tp_move)
    
    entry_comm = exposure * CONFIG['COMMISSION_PCT']
    
    # Simular
    for i in range(idx + 1, len(df)):
        row = df.iloc[i]
        
        if direction == 'LONG':
            if row['low'] <= sl_price:
                pnl_gross = exposure * ((sl_price - entry_price) / entry_price)
                pnl_net = pnl_gross - entry_comm - (exposure * CONFIG['COMMISSION_PCT'])
                
                trade = {
                    'symbol': symbol, 'direction': direction,
                    'entry_time': signal['timestamp'], 'entry_price': entry_price,
                    'sl_price': sl_price, 'tp_price': tp_price,
                    'exit_time': df.iloc[i]['timestamp'], 'exit_price': sl_price,
                    'exit_type': 'SL', 'pnl_net': pnl_net,
                    'roi_pct': (pnl_net / margin) * 100,
                    'duration_hours': i - idx
                }
                all_trades.append(trade)
                open_positions.append(trade['exit_time'])
                add_trade(symbol, signal['timestamp'])
                cooldowns[symbol] = trade['exit_time']  # Cooldown por pérdida
                break
            
            if row['high'] >= tp_price:
                pnl_gross = exposure * ((tp_price - entry_price) / entry_price)
                pnl_net = pnl_gross - entry_comm - (exposure * CONFIG['COMMISSION_PCT'])
                
                trade = {
                    'symbol': symbol, 'direction': direction,
                    'entry_time': signal['timestamp'], 'entry_price': entry_price,
                    'sl_price': sl_price, 'tp_price': tp_price,
                    'exit_time': df.iloc[i]['timestamp'], 'exit_price': tp_price,
                    'exit_type': 'TP', 'pnl_net': pnl_net,
                    'roi_pct': (pnl_net / margin) * 100,
                    'duration_hours': i - idx
                }
                all_trades.append(trade)
                open_positions.append(trade['exit_time'])
                add_trade(symbol, signal['timestamp'])
                break
        
        else:  # SHORT
            if row['high'] >= sl_price:
                pnl_gross = exposure * ((entry_price - sl_price) / entry_price)
                pnl_net = pnl_gross - entry_comm - (exposure * CONFIG['COMMISSION_PCT'])
                
                trade = {
                    'symbol': symbol, 'direction': direction,
                    'entry_time': signal['timestamp'], 'entry_price': entry_price,
                    'sl_price': sl_price, 'tp_price': tp_price,
                    'exit_time': df.iloc[i]['timestamp'], 'exit_price': sl_price,
                    'exit_type': 'SL', 'pnl_net': pnl_net,
                    'roi_pct': (pnl_net / margin) * 100,
                    'duration_hours': i - idx
                }
                all_trades.append(trade)
                open_positions.append(trade['exit_time'])
                add_trade(symbol, signal['timestamp'])
                cooldowns[symbol] = trade['exit_time']
                break
            
            if row['low'] <= tp_price:
                pnl_gross = exposure * ((entry_price - tp_price) / entry_price)
                pnl_net = pnl_gross - entry_comm - (exposure * CONFIG['COMMISSION_PCT'])
                
                trade = {
                    'symbol': symbol, 'direction': direction,
                    'entry_time': signal['timestamp'], 'entry_price': entry_price,
                    'sl_price': sl_price, 'tp_price': tp_price,
                    'exit_time': df.iloc[i]['timestamp'], 'exit_price': tp_price,
                    'exit_type': 'TP', 'pnl_net': pnl_net,
                    'roi_pct': (pnl_net / margin) * 100,
                    'duration_hours': i - idx
                }
                all_trades.append(trade)
                open_positions.append(trade['exit_time'])
                add_trade(symbol, signal['timestamp'])
                break
    
    emoji = "🟢" if all_trades[-1]['pnl_net'] > 0 else "🔴"
    t = all_trades[-1]
    print(f"{emoji} {t['entry_time'].strftime('%m/%d %H:%M')} {t['symbol']} {t['direction']} -> {t['exit_type']} ${t['pnl_net']:.2f} (ROI: {t['roi_pct']:.1f}%)")

# Resultados
print("\n" + "=" * 80)
print("RESULTADOS FINALES")
print("=" * 80)

if not all_trades:
    print("\n❌ No se ejecutaron trades")
else:
    df_trades = pd.DataFrame(all_trades)
    
    total = len(df_trades)
    winners = len(df_trades[df_trades['pnl_net'] > 0])
    losers = len(df_trades[df_trades['pnl_net'] <= 0])
    win_rate = (winners / total * 100) if total > 0 else 0
    
    total_pnl = df_trades['pnl_net'].sum()
    avg_pnl = df_trades['pnl_net'].mean()
    avg_roi = df_trades['roi_pct'].mean()
    
    gross_profit = df_trades[df_trades['pnl_net'] > 0]['pnl_net'].sum()
    gross_loss = abs(df_trades[df_trades['pnl_net'] <= 0]['pnl_net'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    longs = df_trades[df_trades['direction'] == 'LONG']
    shorts = df_trades[df_trades['direction'] == 'SHORT']
    
    print(f"""
📊 Trades:           {total}
   Ganadores:        {winners} ({win_rate:.1f}%)
   Perdedores:       {losers}

💰 PnL Neto:         ${total_pnl:.2f}
   Promedio/Trade:   ${avg_pnl:.2f}
   ROI Promedio:     {avg_roi:.2f}%
   Profit Factor:    {pf:.2f}

📈 LONG:  {len(longs)} trades | ${longs['pnl_net'].sum():.2f}
   SHORT: {len(shorts)} trades | ${shorts['pnl_net'].sum():.2f}

🎯 TP: {len(df_trades[df_trades['exit_type']=='TP'])} | SL: {len(df_trades[df_trades['exit_type']=='SL'])}
    """)
    
    df_trades.to_csv('/Users/laurazapata/Desktop/DICIEMBRE/trades_roi_simplificado_nov_2025.csv', index=False)
    print("📁 Guardado: trades_roi_simplificado_nov_2025.csv")

print("\n" + "=" * 80)
