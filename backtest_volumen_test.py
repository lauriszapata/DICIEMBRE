"""
BACKTEST - COMPARACIÓN DE DIFERENTES NIVELES DE VOLUMEN
========================================================
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

SYMBOLS = ['DOGE/USDT', 'OP/USDT', 'ATOM/USDT', 'FIL/USDT', 'ADA/USDT',
           'TRX/USDT', 'DOT/USDT', 'LINK/USDT', 'ARB/USDT', 'APT/USDT']

def calculate_indicators(df):
    df = df.copy()
    df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = high.diff(); plus_dm[plus_dm < 0] = 0
    minus_dm = low.diff().abs() * -1; minus_dm[minus_dm > 0] = 0; minus_dm = minus_dm.abs()
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr_adx = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_adx)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_adx)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df['adx'] = dx.ewm(span=14, adjust=False).mean()
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    df['macd_hist'] = (ema_fast - ema_slow) - (ema_fast - ema_slow).ewm(span=9, adjust=False).mean()
    
    df['atr'] = tr.ewm(span=14, adjust=False).mean()
    df['atr_pct'] = df['atr'] / df['close']
    df['vol_sma'] = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma']
    df['ema20_dist'] = (df['close'] - df['ema20']).abs() / df['atr']
    return df

def detect_higher_low(df, eval_idx):
    pivot_idx = eval_idx - 2
    if pivot_idx < 3: return False
    pivot_low = df['low'].iloc[pivot_idx]
    if df['low'].iloc[pivot_idx - 1] <= pivot_low: return False
    if df['low'].iloc[pivot_idx + 1] <= pivot_low: return False
    for i in range(pivot_idx - 3, max(0, pivot_idx - 50), -1):
        if i <= 0 or i >= len(df) - 1: continue
        prev_low = df['low'].iloc[i]
        if df['low'].iloc[i-1] > prev_low and df['low'].iloc[i+1] > prev_low:
            return pivot_low > prev_low
    return False

def detect_lower_high(df, eval_idx):
    pivot_idx = eval_idx - 2
    if pivot_idx < 3: return False
    pivot_high = df['high'].iloc[pivot_idx]
    if df['high'].iloc[pivot_idx - 1] >= pivot_high: return False
    if df['high'].iloc[pivot_idx + 1] >= pivot_high: return False
    for i in range(pivot_idx - 3, max(0, pivot_idx - 50), -1):
        if i <= 0 or i >= len(df) - 1: continue
        prev_high = df['high'].iloc[i]
        if df['high'].iloc[i-1] < prev_high and df['high'].iloc[i+1] < prev_high:
            return pivot_high < prev_high
    return False

def check_signal(df, idx, vol_ratio_min):
    if idx < 60: return None
    row = df.iloc[idx]
    required = ['ema8', 'ema20', 'ema21', 'ema50', 'adx', 'rsi', 'macd_hist', 'atr', 'vol_ratio', 'atr_pct', 'ema20_dist']
    if any(pd.isna(row[col]) for col in required): return None
    if not (0.002 <= row['atr_pct'] <= 0.15): return None
    if row['adx'] < 28: return None
    if row['ema20_dist'] >= 3.0: return None
    if row['vol_ratio'] < vol_ratio_min: return None
    
    if all([row['ema8'] > row['ema21'], row['close'] > row['ema50'], row['ema20'] > row['ema50'],
            row['rsi'] > 55, row['macd_hist'] > 0, detect_higher_low(df, idx)]):
        return 'LONG'
    if all([row['ema8'] < row['ema21'], row['close'] < row['ema50'], row['ema20'] < row['ema50'],
            row['rsi'] < 70, row['macd_hist'] < 0, detect_lower_high(df, idx)]):
        return 'SHORT'
    return None

def run_backtest(all_data, vol_ratio_min, start_date, end_date):
    trades = []
    position = None
    current = start_date
    
    while current <= end_date:
        if position:
            symbol = position['symbol']
            df = all_data[symbol]
            mask = df['timestamp'] == pd.Timestamp(current)
            if mask.any():
                idx = df[mask].index[0]
                candle = df.iloc[idx]
                hit_sl = hit_tp = False
                exit_price = None
                
                if position['direction'] == 'LONG':
                    if candle['low'] <= position['sl']: hit_sl, exit_price = True, position['sl']
                    elif candle['high'] >= position['tp']: hit_tp, exit_price = True, position['tp']
                else:
                    if candle['high'] >= position['sl']: hit_sl, exit_price = True, position['sl']
                    elif candle['low'] <= position['tp']: hit_tp, exit_price = True, position['tp']
                
                if hit_sl or hit_tp:
                    if position['direction'] == 'LONG':
                        pnl_pct = (exit_price - position['entry']) / position['entry']
                    else:
                        pnl_pct = (position['entry'] - exit_price) / position['entry']
                    pnl_usd = pnl_pct * 100 * 20 - 2  # $100 margin, 20x leverage, -$2 fees
                    trades.append({'pnl': pnl_usd, 'result': 'TP' if hit_tp else 'SL'})
                    position = None
        
        if position is None:
            for symbol in SYMBOLS:
                df = all_data[symbol]
                mask = df['timestamp'] == pd.Timestamp(current)
                if not mask.any(): continue
                idx = df[mask].index[0]
                if idx < 2: continue
                signal = check_signal(df, idx - 1, vol_ratio_min)
                if signal:
                    row = df.iloc[idx - 1]
                    entry = df.iloc[idx]['open']
                    atr = row['atr']
                    if signal == 'LONG':
                        sl, tp = entry - atr * 1.5, entry + atr * 3.0
                    else:
                        sl, tp = entry + atr * 1.5, entry - atr * 3.0
                    position = {'symbol': symbol, 'direction': signal, 'entry': entry, 'sl': sl, 'tp': tp}
                    break
        current += timedelta(hours=1)
    
    return trades

def main():
    print("=" * 70)
    print("📊 BACKTEST NOVIEMBRE 2025 - DIFERENTES NIVELES DE VOLUMEN")
    print("   Leverage: 20x | Margen: $100")
    print("=" * 70)
    
    # Conectar y descargar datos
    print("\n📥 Descargando datos...")
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    exchange.load_markets()
    
    start_date = datetime(2025, 11, 1)
    end_date = datetime(2025, 11, 30, 23, 59, 59)
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    all_data = {}
    for symbol in SYMBOLS:
        print(f"   {symbol}...", end=" ", flush=True)
        ohlcv = []
        since = start_ts - (200 * 60 * 60 * 1000)
        while since < end_ts:
            batch = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=1000)
            if not batch: break
            ohlcv.extend(batch)
            since = batch[-1][0] + 1
            if len(batch) < 1000: break
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
        df = calculate_indicators(df)
        all_data[symbol] = df
        print("✓")
    
    # Probar diferentes niveles de volumen
    print("\n" + "=" * 70)
    print("📊 RESULTADOS:")
    print("=" * 70)
    print(f"\n{'Vol Mínimo':<12} | {'Trades':>7} | {'WinRate':>8} | {'PnL Total':>12} | {'Promedio':>10}")
    print("-" * 70)
    
    vol_levels = [1.2, 1.0, 0.8, 0.6, 0.5, 0.0]
    results = []
    
    for vol in vol_levels:
        trades = run_backtest(all_data, vol, start_date, end_date)
        if trades:
            df_t = pd.DataFrame(trades)
            total = df_t['pnl'].sum()
            wins = len(df_t[df_t['pnl'] > 0])
            wr = wins / len(trades) * 100
            avg = total / len(trades)
            results.append({'vol': vol, 'trades': len(trades), 'wr': wr, 'pnl': total, 'avg': avg})
            
            # Emoji según resultados
            if total > 200:
                emoji = "🔥"
            elif total > 0:
                emoji = "✅"
            else:
                emoji = "❌"
            
            print(f">= {vol:.1f}x       | {len(trades):>7} | {wr:>7.1f}% | ${total:>10.2f} | ${avg:>9.2f} {emoji}")
        else:
            print(f">= {vol:.1f}x       | {'0':>7} | {'N/A':>8} | {'$0.00':>12} | {'N/A':>10}")
    
    # Mejor resultado
    print("\n" + "=" * 70)
    if results:
        best = max(results, key=lambda x: x['pnl'])
        print(f"🏆 MEJOR CONFIGURACIÓN: Vol >= {best['vol']:.1f}x")
        print(f"   Trades: {best['trades']} | WinRate: {best['wr']:.1f}% | PnL: ${best['pnl']:.2f}")
        
        # Comparar con 1.2 (original)
        original = next((r for r in results if r['vol'] == 1.2), None)
        if original and best['vol'] != 1.2:
            diff = best['pnl'] - original['pnl']
            print(f"\n   Diferencia vs original (1.2x): ${diff:+.2f}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
