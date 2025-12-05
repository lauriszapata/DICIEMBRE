"""
BACKTEST ANUAL 2025 - CONFIGURACIÓN GANADORA ATR
=================================================
Usando la configuración que generó +$593 en noviembre 2025
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'MARGIN_USD': 100,
    'LEVERAGE': 20,              # ⚡ CAMBIADO A 20x
    'MAX_OPEN_POSITIONS': 3,
    'TIMEFRAME': '1h',
    
    # SL/TP ATR-based (configuración ganadora actual)
    'SL_ATR_MULT': 1.3,
    'TP_ATR_MULT': 3.5,
    
    # Indicadores
    'ADX_MIN': 28,
    'RSI_LONG_MIN': 55,
    'RSI_SHORT_MAX': 70,
    'VOLUME_RATIO': 1.2,
    'EMA_EXTENSION_ATR_MULT': 3.0,
    
    # Filtros
    'ATR_MIN_PCT': 0.002,
    'ATR_MAX_PCT': 0.15,
    
    # Filtro mercado lateral (igual que bot real)
    'USE_LATERAL_FILTER': True,
    'LATERAL_RANGE_PCT': 0.015,  # 1.5%
    
    # Filtro de días y horas (basado en backtest 2025)
    'USE_TIME_FILTER': True,
    'BLOCKED_DAYS': [1],            # Martes bloqueado
    'BLOCKED_HOURS_UTC': [1, 12, 23],  # Peores horas
    
    'COMMISSION_PCT': 0.0005,
}

# SÍMBOLOS RENTABLES EN 2025 (sin ATOM, DOGE, DOT, APT)
SYMBOLS = [
    'ADA/USDT',   # +$532.52 en 2025
    'FIL/USDT',   # +$292.54 en 2025
    'ARB/USDT',   # +$266.45 en 2025
    'LINK/USDT',  # +$176.35 en 2025
    'OP/USDT',    # +$114.99 en 2025
    'TRX/USDT',   # +$107.79 en 2025
]

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = high.diff().where((high.diff() > (-low.diff())) & (high.diff() > 0), 0)
    minus_dm = (-low.diff()).where(((-low.diff()) > high.diff()) & ((-low.diff()) > 0), 0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return dx.rolling(window=period).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    ema_fast = series.ewm(span=12, adjust=False).mean()
    ema_slow = series.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line - signal_line

def calculate_indicators(df):
    if len(df) < 60:
        return df
    df = df.copy()
    df['ema8'] = calculate_ema(df['close'], 8)
    df['ema20'] = calculate_ema(df['close'], 20)
    df['ema21'] = calculate_ema(df['close'], 21)
    df['ema50'] = calculate_ema(df['close'], 50)
    df['atr'] = calculate_atr(df, 14)
    df['adx'] = calculate_adx(df, 14)
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['macd_hist'] = calculate_macd(df['close'])
    df['vol_sma20'] = df['volume'].rolling(window=20).mean()
    df['atr_pct'] = df['atr'] / df['close']
    df['ema20_dist_atr'] = abs(df['close'] - df['ema20']) / df['atr']
    return df

def detect_pivot_low(df, idx, lookback=50):
    if idx < lookback + 3:
        return False
    pivot_idx = idx - 2
    pivot_low = df['low'].iloc[pivot_idx]
    if df['low'].iloc[pivot_idx - 1] <= pivot_low or df['low'].iloc[pivot_idx + 1] <= pivot_low:
        return False
    for i in range(pivot_idx - 3, max(pivot_idx - lookback, 0), -1):
        if i <= 0 or i >= len(df) - 1:
            continue
        prev_low = df['low'].iloc[i]
        if df['low'].iloc[i-1] > prev_low and df['low'].iloc[i+1] > prev_low:
            return pivot_low > prev_low
    return False

def detect_pivot_high(df, idx, lookback=50):
    if idx < lookback + 3:
        return False
    pivot_idx = idx - 2
    pivot_high = df['high'].iloc[pivot_idx]
    if df['high'].iloc[pivot_idx - 1] >= pivot_high or df['high'].iloc[pivot_idx + 1] >= pivot_high:
        return False
    for i in range(pivot_idx - 3, max(pivot_idx - lookback, 0), -1):
        if i <= 0 or i >= len(df) - 1:
            continue
        prev_high = df['high'].iloc[i]
        if df['high'].iloc[i-1] < prev_high and df['high'].iloc[i+1] < prev_high:
            return pivot_high < prev_high
    return False

def check_lateral(df, idx):
    """Verifica si el mercado está lateral (rango últimas 4 velas < 1.5%)"""
    if not CONFIG.get('USE_LATERAL_FILTER', True):
        return False
    if idx < 5:
        return False
    
    # Últimas 4 velas cerradas (hasta idx, excluyendo la actual)
    last_4 = df.iloc[idx-4:idx]
    high = last_4['high'].max()
    low = last_4['low'].min()
    range_pct = (high - low) / low
    
    return range_pct < CONFIG.get('LATERAL_RANGE_PCT', 0.015)

def check_time_filter(timestamp):
    """Verifica si el día/hora están bloqueados"""
    if not CONFIG.get('USE_TIME_FILTER', True):
        return False
    day_of_week = timestamp.weekday()  # 0=Lunes, 1=Martes
    hour_utc = timestamp.hour
    if day_of_week in CONFIG.get('BLOCKED_DAYS', [1]):
        return True
    if hour_utc in CONFIG.get('BLOCKED_HOURS_UTC', [1, 12, 23]):
        return True
    return False

def check_long_entry(df, idx):
    row = df.iloc[idx]
    if idx < 60:
        return False
    # Filtro lateral
    if check_lateral(df, idx):
        return False
    # Filtro de días/horas
    if check_time_filter(df['timestamp'].iloc[idx]):
        return False
    return (row['ema8'] > row['ema21'] and row['close'] > row['ema50'] and row['ema20'] > row['ema50'] and
            row['adx'] >= CONFIG['ADX_MIN'] and row['rsi'] > CONFIG['RSI_LONG_MIN'] and
            row['macd_hist'] > 0 and row['volume'] >= CONFIG['VOLUME_RATIO'] * row['vol_sma20'] and
            detect_pivot_low(df, idx) and row['ema20_dist_atr'] < CONFIG['EMA_EXTENSION_ATR_MULT'] and
            CONFIG['ATR_MIN_PCT'] <= row['atr_pct'] <= CONFIG['ATR_MAX_PCT'])

def check_short_entry(df, idx):
    row = df.iloc[idx]
    if idx < 60:
        return False
    # Filtro lateral
    if check_lateral(df, idx):
        return False
    # Filtro de días/horas
    if check_time_filter(df['timestamp'].iloc[idx]):
        return False
    return (row['ema8'] < row['ema21'] and row['close'] < row['ema50'] and row['ema20'] < row['ema50'] and
            row['adx'] >= CONFIG['ADX_MIN'] and row['rsi'] < CONFIG['RSI_SHORT_MAX'] and
            row['macd_hist'] < 0 and row['volume'] >= CONFIG['VOLUME_RATIO'] * row['vol_sma20'] and
            detect_pivot_high(df, idx) and row['ema20_dist_atr'] < CONFIG['EMA_EXTENSION_ATR_MULT'] and
            CONFIG['ATR_MIN_PCT'] <= row['atr_pct'] <= CONFIG['ATR_MAX_PCT'])

def simulate_trade(df, entry_idx, direction, entry_price, atr):
    exposure = CONFIG['MARGIN_USD'] * CONFIG['LEVERAGE']
    
    if direction == 'LONG':
        sl_price = entry_price - (CONFIG['SL_ATR_MULT'] * atr)
        tp_price = entry_price + (CONFIG['TP_ATR_MULT'] * atr)
    else:
        sl_price = entry_price + (CONFIG['SL_ATR_MULT'] * atr)
        tp_price = entry_price - (CONFIG['TP_ATR_MULT'] * atr)
    
    for i in range(entry_idx + 1, len(df)):
        row = df.iloc[i]
        
        if direction == 'LONG':
            if row['low'] <= sl_price:
                pnl_pct = (sl_price - entry_price) / entry_price
                pnl = exposure * pnl_pct - (exposure * CONFIG['COMMISSION_PCT'] * 2)
                return {'exit_idx': i, 'exit_price': sl_price, 'exit_type': 'SL', 'pnl': pnl, 'duration_hours': i - entry_idx}
            if row['high'] >= tp_price:
                pnl_pct = (tp_price - entry_price) / entry_price
                pnl = exposure * pnl_pct - (exposure * CONFIG['COMMISSION_PCT'] * 2)
                return {'exit_idx': i, 'exit_price': tp_price, 'exit_type': 'TP', 'pnl': pnl, 'duration_hours': i - entry_idx}
        else:
            if row['high'] >= sl_price:
                pnl_pct = (entry_price - sl_price) / entry_price
                pnl = exposure * pnl_pct - (exposure * CONFIG['COMMISSION_PCT'] * 2)
                return {'exit_idx': i, 'exit_price': sl_price, 'exit_type': 'SL', 'pnl': pnl, 'duration_hours': i - entry_idx}
            if row['low'] <= tp_price:
                pnl_pct = (entry_price - tp_price) / entry_price
                pnl = exposure * pnl_pct - (exposure * CONFIG['COMMISSION_PCT'] * 2)
                return {'exit_idx': i, 'exit_price': tp_price, 'exit_type': 'TP', 'pnl': pnl, 'duration_hours': i - entry_idx}
    
    last_price = df['close'].iloc[-1]
    pnl_pct = ((last_price - entry_price) / entry_price) if direction == 'LONG' else ((entry_price - last_price) / entry_price)
    pnl = exposure * pnl_pct - (exposure * CONFIG['COMMISSION_PCT'] * 2)
    return {'exit_idx': len(df) - 1, 'exit_price': last_price, 'exit_type': 'OPEN', 'pnl': pnl, 'duration_hours': len(df) - 1 - entry_idx}

print("=" * 80)
print("BACKTEST ANUAL 2025 - CONFIGURACIÓN GANADORA ATR")
print("=" * 80)
print("\n📅 Período: Enero - Diciembre 2025")
print(f"🎯 SL: {CONFIG['SL_ATR_MULT']}× ATR | TP: {CONFIG['TP_ATR_MULT']}× ATR")
print("💰 Margin: $100 | Leverage: 20x\n")

exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})

# TODO EL AÑO 2025
start_data = datetime(2024, 12, 1)  # Warmup desde diciembre 2024
start_backtest = datetime(2025, 1, 1)
end_backtest = datetime(2025, 12, 4)  # Hasta hoy

since_ms = int(start_data.timestamp() * 1000)
until_ms = int(end_backtest.timestamp() * 1000)

print("📥 Descargando datos...\n")
symbol_data = {}

for symbol in SYMBOLS:
    print(f"   {symbol}...", end=" ")
    try:
        all_ohlcv = []
        current = since_ms
        while current < until_ms:
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=current, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        df = df[df['timestamp'] < pd.to_datetime(until_ms, unit='ms')]
        
        if len(df) >= 100:
            df = calculate_indicators(df)
            symbol_data[symbol] = df
            print(f"✅ {len(df)} velas")
        else:
            print(f"⚠️ Solo {len(df)} velas")
    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\n✅ {len(symbol_data)} símbolos cargados\n")
print("-" * 80)

# Buscar señales
all_signals = []
for symbol, df in symbol_data.items():
    start_idx = df[df['timestamp'] >= start_backtest].index[0] if any(df['timestamp'] >= start_backtest) else len(df)
    
    for idx in range(start_idx, len(df)):
        row = df.iloc[idx]
        if check_long_entry(df, idx):
            all_signals.append({
                'timestamp': row['timestamp'], 'symbol': symbol, 'direction': 'LONG',
                'idx': idx, 'entry_price': row['close'], 'atr': row['atr']
            })
        if check_short_entry(df, idx):
            all_signals.append({
                'timestamp': row['timestamp'], 'symbol': symbol, 'direction': 'SHORT',
                'idx': idx, 'entry_price': row['close'], 'atr': row['atr']
            })

all_signals.sort(key=lambda x: x['timestamp'])
print(f"🔍 {len(all_signals)} señales encontradas\n")

# Ejecutar trades
all_trades = []
open_positions = []

for signal in all_signals:
    open_positions = [t for t in open_positions if t > signal['timestamp']]
    if len(open_positions) >= CONFIG['MAX_OPEN_POSITIONS']:
        continue
    
    df = symbol_data[signal['symbol']]
    result = simulate_trade(df, signal['idx'], signal['direction'], signal['entry_price'], signal['atr'])
    
    if signal['direction'] == 'LONG':
        sl_price = signal['entry_price'] - (CONFIG['SL_ATR_MULT'] * signal['atr'])
        tp_price = signal['entry_price'] + (CONFIG['TP_ATR_MULT'] * signal['atr'])
    else:
        sl_price = signal['entry_price'] + (CONFIG['SL_ATR_MULT'] * signal['atr'])
        tp_price = signal['entry_price'] - (CONFIG['TP_ATR_MULT'] * signal['atr'])
    
    trade = {
        'symbol': signal['symbol'], 'direction': signal['direction'],
        'entry_time': signal['timestamp'], 'entry_price': signal['entry_price'],
        'sl_price': sl_price, 'tp_price': tp_price,
        'exit_time': df.iloc[result['exit_idx']]['timestamp'],
        'exit_price': result['exit_price'], 'exit_type': result['exit_type'],
        'pnl': result['pnl'], 'duration_hours': result['duration_hours']
    }
    
    all_trades.append(trade)
    open_positions.append(trade['exit_time'])

print(f"✅ {len(all_trades)} trades ejecutados\n")
print("=" * 80)

# Análisis por mes
trades_df = pd.DataFrame(all_trades)
trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')

print("\n📅 RENDIMIENTO MENSUAL")
print("=" * 80)
print(f"\n{'Mes':<12} {'Trades':<8} {'Ganadores':<11} {'Win%':<8} {'PnL':<12} {'PnL Acum':<12}")
print("-" * 80)

cumulative_pnl = 0
monthly_stats = []

for month in sorted(trades_df['month'].unique()):
    month_trades = trades_df[trades_df['month'] == month]
    total = len(month_trades)
    winners = len(month_trades[month_trades['pnl'] > 0])
    win_rate = (winners / total * 100) if total > 0 else 0
    pnl = month_trades['pnl'].sum()
    cumulative_pnl += pnl
    
    emoji = "🟢" if pnl > 0 else "🔴"
    print(f"{str(month):<12} {total:<8} {winners:<11} {win_rate:<7.1f}% {emoji} ${pnl:<10.2f} ${cumulative_pnl:<10.2f}")
    
    monthly_stats.append({
        'month': str(month), 'trades': total, 'winners': winners,
        'win_rate': win_rate, 'pnl': pnl, 'cumulative': cumulative_pnl
    })

# Resumen final
print("\n" + "=" * 80)
print("RESUMEN ANUAL 2025")
print("=" * 80)

total_trades = len(trades_df)
winners = len(trades_df[trades_df['pnl'] > 0])
win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
total_pnl = trades_df['pnl'].sum()

gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

roi_anual = (total_pnl / CONFIG['MARGIN_USD']) * 100

print(f"""
📊 Total Trades:        {total_trades}
   Ganadores:           {winners} ({win_rate:.1f}%)
   Perdedores:          {total_trades - winners}

💰 PnL Total:           ${total_pnl:.2f}
   ROI Anual:           {roi_anual:.1f}%
   Profit Factor:       {pf:.2f}
   
   Promedio/Trade:      ${total_pnl / total_trades:.2f}
   Mejor mes:           {max(monthly_stats, key=lambda x: x['pnl'])['month']} (${max(monthly_stats, key=lambda x: x['pnl'])['pnl']:.2f})
   Peor mes:            {min(monthly_stats, key=lambda x: x['pnl'])['month']} (${min(monthly_stats, key=lambda x: x['pnl'])['pnl']:.2f})

🎯 TP: {len(trades_df[trades_df['exit_type']=='TP'])} | SL: {len(trades_df[trades_df['exit_type']=='SL'])}
""")

trades_df.to_csv('/Users/laurazapata/Desktop/DICIEMBRE/trades_anual_2025.csv', index=False)
print("📁 Guardado: trades_anual_2025.csv")
print("\n" + "=" * 80)
