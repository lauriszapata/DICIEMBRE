"""
BACKTEST OPTIMIZACIÓN DE STOP LOSS
==================================
Prueba diferentes valores de SL (multiplicador ATR) para encontrar el mínimo rentable.

Configuración base tomada de bot_ganadora_v3.py:
- SL actual: 1.5× ATR
- TP actual: 3.0× ATR
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración base
CONFIG = {
    'MARGIN_USD': 100,
    'LEVERAGE': 10,
    'MAX_OPEN_POSITIONS': 3,
    'TIMEFRAME': '1h',
    
    # TP fijo
    'TP_ATR_MULT': 3.0,
    
    # Indicadores
    'ADX_MIN': 28,
    'RSI_LONG_MIN': 55,
    'RSI_SHORT_MAX': 70,
    'VOLUME_RATIO': 1.2,
    'EMA_EXTENSION_ATR_MULT': 3.0,
    
    # Filtros
    'ATR_MIN_PCT': 0.002,
    'ATR_MAX_PCT': 0.15,
    
    'COMMISSION_PCT': 0.0005,
}

# Símbolos rentables según bot actual
SYMBOLS = [
    'ADA/USDT', 'FIL/USDT', 'ARB/USDT', 'LINK/USDT', 'OP/USDT', 'TRX/USDT'
]

# Rango de SL a probar (multiplicador ATR)
SL_RANGE = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0, 2.5, 3.0]


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

def check_long_entry(df, idx):
    row = df.iloc[idx]
    if idx < 60:
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
    return (row['ema8'] < row['ema21'] and row['close'] < row['ema50'] and row['ema20'] < row['ema50'] and
            row['adx'] >= CONFIG['ADX_MIN'] and row['rsi'] < CONFIG['RSI_SHORT_MAX'] and
            row['macd_hist'] < 0 and row['volume'] >= CONFIG['VOLUME_RATIO'] * row['vol_sma20'] and
            detect_pivot_high(df, idx) and row['ema20_dist_atr'] < CONFIG['EMA_EXTENSION_ATR_MULT'] and
            CONFIG['ATR_MIN_PCT'] <= row['atr_pct'] <= CONFIG['ATR_MAX_PCT'])

def simulate_trade(df, entry_idx, direction, entry_price, atr, sl_mult, tp_mult):
    """Simula un trade con SL y TP específicos"""
    exposure = CONFIG['MARGIN_USD'] * CONFIG['LEVERAGE']
    
    if direction == 'LONG':
        sl_price = entry_price - (sl_mult * atr)
        tp_price = entry_price + (tp_mult * atr)
    else:
        sl_price = entry_price + (sl_mult * atr)
        tp_price = entry_price - (tp_mult * atr)
    
    for i in range(entry_idx + 1, len(df)):
        row = df.iloc[i]
        
        if direction == 'LONG':
            if row['low'] <= sl_price:
                pnl_pct = (sl_price - entry_price) / entry_price
                pnl = exposure * pnl_pct - (exposure * CONFIG['COMMISSION_PCT'] * 2)
                return {'exit_type': 'SL', 'pnl': pnl, 'duration': i - entry_idx}
            if row['high'] >= tp_price:
                pnl_pct = (tp_price - entry_price) / entry_price
                pnl = exposure * pnl_pct - (exposure * CONFIG['COMMISSION_PCT'] * 2)
                return {'exit_type': 'TP', 'pnl': pnl, 'duration': i - entry_idx}
        else:
            if row['high'] >= sl_price:
                pnl_pct = (entry_price - sl_price) / entry_price
                pnl = exposure * pnl_pct - (exposure * CONFIG['COMMISSION_PCT'] * 2)
                return {'exit_type': 'SL', 'pnl': pnl, 'duration': i - entry_idx}
            if row['low'] <= tp_price:
                pnl_pct = (entry_price - tp_price) / entry_price
                pnl = exposure * pnl_pct - (exposure * CONFIG['COMMISSION_PCT'] * 2)
                return {'exit_type': 'TP', 'pnl': pnl, 'duration': i - entry_idx}
    
    # Trade abierto al final
    last_price = df['close'].iloc[-1]
    pnl_pct = ((last_price - entry_price) / entry_price) if direction == 'LONG' else ((entry_price - last_price) / entry_price)
    pnl = exposure * pnl_pct - (exposure * CONFIG['COMMISSION_PCT'] * 2)
    return {'exit_type': 'OPEN', 'pnl': pnl, 'duration': len(df) - 1 - entry_idx}


def run_backtest_with_sl(symbol_data, signals, sl_mult, tp_mult):
    """Ejecuta backtest con un SL específico"""
    trades = []
    open_positions = []
    
    for signal in signals:
        open_positions = [t for t in open_positions if t > signal['timestamp']]
        if len(open_positions) >= CONFIG['MAX_OPEN_POSITIONS']:
            continue
        
        df = symbol_data[signal['symbol']]
        result = simulate_trade(df, signal['idx'], signal['direction'], 
                               signal['entry_price'], signal['atr'], sl_mult, tp_mult)
        
        exit_idx = min(signal['idx'] + result['duration'], len(df) - 1)
        trades.append({
            'symbol': signal['symbol'],
            'direction': signal['direction'],
            'entry_time': signal['timestamp'],
            'exit_time': df.iloc[exit_idx]['timestamp'],
            'exit_type': result['exit_type'],
            'pnl': result['pnl'],
            'duration': result['duration']
        })
        open_positions.append(df.iloc[exit_idx]['timestamp'])
    
    return trades


def main():
    print("=" * 80)
    print("🔍 OPTIMIZACIÓN DE STOP LOSS - BACKTEST ANUAL 2025")
    print("=" * 80)
    print("\n📅 Período: Enero - Diciembre 2025")
    print(f"🎯 TP fijo: {CONFIG['TP_ATR_MULT']}× ATR")
    print(f"📊 Probando SL: {SL_RANGE}")
    print(f"💰 Margen: ${CONFIG['MARGIN_USD']} | Leverage: {CONFIG['LEVERAGE']}x\n")
    
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    
    # Período de backtest
    start_data = datetime(2024, 12, 1)
    start_backtest = datetime(2025, 1, 1)
    end_backtest = datetime(2025, 12, 3)
    
    since_ms = int(start_data.timestamp() * 1000)
    until_ms = int(end_backtest.timestamp() * 1000)
    
    # Descargar datos
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
    
    # Buscar señales (una sola vez, el SL no cambia las señales)
    print("🔍 Buscando señales...")
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
    print(f"   {len(all_signals)} señales encontradas\n")
    
    # ========================================================================
    # PRUEBA CADA SL
    # ========================================================================
    print("=" * 80)
    print("📊 RESULTADOS POR STOP LOSS")
    print("=" * 80)
    print(f"\n{'SL (ATR)':<10} {'Trades':<8} {'Win%':<8} {'PnL':<12} {'PF':<8} {'Avg/Trade':<12} {'Status'}")
    print("-" * 80)
    
    results = []
    tp_mult = CONFIG['TP_ATR_MULT']
    
    for sl_mult in SL_RANGE:
        trades = run_backtest_with_sl(symbol_data, all_signals, sl_mult, tp_mult)
        
        if not trades:
            continue
        
        trades_df = pd.DataFrame(trades)
        total = len(trades_df)
        winners = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winners / total * 100) if total > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = total_pnl / total if total > 0 else 0
        
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Status
        if total_pnl > 0:
            status = "✅ RENTABLE"
        else:
            status = "❌ PÉRDIDA"
        
        emoji = "🟢" if total_pnl > 0 else "🔴"
        print(f"{sl_mult:<10.1f} {total:<8} {win_rate:<7.1f}% {emoji}${total_pnl:<10.2f} {pf:<8.2f} ${avg_pnl:<10.2f} {status}")
        
        results.append({
            'sl_mult': sl_mult,
            'tp_mult': tp_mult,
            'trades': total,
            'win_rate': win_rate,
            'pnl': total_pnl,
            'profit_factor': pf,
            'avg_per_trade': avg_pnl,
            'profitable': total_pnl > 0
        })
    
    # ========================================================================
    # ANÁLISIS FINAL
    # ========================================================================
    print("\n" + "=" * 80)
    print("📈 ANÁLISIS DE RESULTADOS")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    
    # Encontrar SL mínimo rentable
    profitable = results_df[results_df['profitable'] == True]
    
    if len(profitable) > 0:
        min_sl = profitable['sl_mult'].min()
        best_sl = profitable.loc[profitable['pnl'].idxmax()]
        
        print(f"""
🎯 SL MÍNIMO RENTABLE: {min_sl}× ATR

📊 COMPARACIÓN:
   - SL mínimo rentable: {min_sl}× ATR → PnL: ${profitable[profitable['sl_mult'] == min_sl]['pnl'].values[0]:.2f}
   - SL actual (1.5×):   PnL: ${results_df[results_df['sl_mult'] == 1.5]['pnl'].values[0]:.2f}
   - SL óptimo:          {best_sl['sl_mult']}× ATR → PnL: ${best_sl['pnl']:.2f}

📋 RANGOS RENTABLES:
""")
        for _, row in profitable.iterrows():
            print(f"   SL {row['sl_mult']}× ATR: ${row['pnl']:.2f} | WR: {row['win_rate']:.1f}% | PF: {row['profit_factor']:.2f}")
    else:
        print("\n❌ NINGÚN SL ES RENTABLE CON LA CONFIGURACIÓN ACTUAL")
        best_sl = results_df.loc[results_df['pnl'].idxmax()]
        print(f"\n   Mejor (menos pérdida): SL {best_sl['sl_mult']}× ATR → PnL: ${best_sl['pnl']:.2f}")
    
    # Guardar resultados
    results_df.to_csv('/Users/laurazapata/Desktop/DICIEMBRE/sl_optimization_results.csv', index=False)
    print(f"\n📁 Resultados guardados en: sl_optimization_results.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
