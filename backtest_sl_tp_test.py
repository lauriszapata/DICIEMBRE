"""
BACKTEST - COMPARACIÓN DE DIFERENTES NIVELES DE SL/TP
======================================================
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

def check_signal(df, idx):
    if idx < 60: return None
    row = df.iloc[idx]
    required = ['ema8', 'ema20', 'ema21', 'ema50', 'adx', 'rsi', 'macd_hist', 'atr', 'vol_ratio', 'atr_pct', 'ema20_dist']
    if any(pd.isna(row[col]) for col in required): return None
    if not (0.002 <= row['atr_pct'] <= 0.15): return None
    if row['adx'] < 28: return None
    if row['ema20_dist'] >= 3.0: return None
    if row['vol_ratio'] < 1.2: return None
    
    if all([row['ema8'] > row['ema21'], row['close'] > row['ema50'], row['ema20'] > row['ema50'],
            row['rsi'] > 55, row['macd_hist'] > 0, detect_higher_low(df, idx)]):
        return 'LONG'
    if all([row['ema8'] < row['ema21'], row['close'] < row['ema50'], row['ema20'] < row['ema50'],
            row['rsi'] < 70, row['macd_hist'] < 0, detect_lower_high(df, idx)]):
        return 'SHORT'
    return None

def run_backtest(all_data, sl_mult, tp_mult, start_date, end_date, leverage=20):
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
                    pnl_usd = pnl_pct * 100 * leverage - 2
                    trades.append({'pnl': pnl_usd, 'result': 'TP' if hit_tp else 'SL'})
                    position = None
        
        if position is None:
            for symbol in SYMBOLS:
                df = all_data[symbol]
                mask = df['timestamp'] == pd.Timestamp(current)
                if not mask.any(): continue
                idx = df[mask].index[0]
                if idx < 2: continue
                signal = check_signal(df, idx - 1)
                if signal:
                    row = df.iloc[idx - 1]
                    entry = df.iloc[idx]['open']
                    atr = row['atr']
                    if signal == 'LONG':
                        sl, tp = entry - atr * sl_mult, entry + atr * tp_mult
                    else:
                        sl, tp = entry + atr * sl_mult, entry - atr * tp_mult
                    position = {'symbol': symbol, 'direction': signal, 'entry': entry, 'sl': sl, 'tp': tp}
                    break
        current += timedelta(hours=1)
    
    return trades

def main():
    print("=" * 80)
    print("📊 BACKTEST NOVIEMBRE 2025 - DIFERENTES NIVELES DE SL/TP")
    print("   Leverage: 20x | Margen: $100 | Volumen >= 1.2x")
    print("=" * 80)
    
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
    
    # Configuraciones a probar
    # (SL multiplier, TP multiplier, descripción)
    configs = [
        # SL más cercano
        (0.8, 1.6, "SL 0.8x, TP 1.6x (R:R 1:2)"),
        (1.0, 2.0, "SL 1.0x, TP 2.0x (R:R 1:2)"),
        (1.0, 2.5, "SL 1.0x, TP 2.5x (R:R 1:2.5)"),
        (1.0, 3.0, "SL 1.0x, TP 3.0x (R:R 1:3)"),
        
        # Original
        (1.5, 3.0, "SL 1.5x, TP 3.0x (ORIGINAL)"),
        
        # SL más amplio
        (1.5, 4.0, "SL 1.5x, TP 4.0x (R:R 1:2.7)"),
        (2.0, 4.0, "SL 2.0x, TP 4.0x (R:R 1:2)"),
        (2.0, 5.0, "SL 2.0x, TP 5.0x (R:R 1:2.5)"),
    ]
    
    print("\n" + "=" * 80)
    print("📊 RESULTADOS:")
    print("=" * 80)
    print(f"\n{'Configuración':<35} | {'Trades':>6} | {'WinRate':>7} | {'PnL':>10} | {'Avg':>8}")
    print("-" * 80)
    
    results = []
    
    for sl_mult, tp_mult, desc in configs:
        trades = run_backtest(all_data, sl_mult, tp_mult, start_date, end_date)
        if trades:
            df_t = pd.DataFrame(trades)
            total = df_t['pnl'].sum()
            wins = len(df_t[df_t['pnl'] > 0])
            wr = wins / len(trades) * 100
            avg = total / len(trades)
            results.append({'sl': sl_mult, 'tp': tp_mult, 'desc': desc, 'trades': len(trades), 'wr': wr, 'pnl': total, 'avg': avg})
            
            # Marcar original y mejor
            marker = ""
            if sl_mult == 1.5 and tp_mult == 3.0:
                marker = " ⭐"
            if total > 300:
                emoji = "🔥"
            elif total > 0:
                emoji = "✅"
            else:
                emoji = "❌"
            
            print(f"{desc:<35} | {len(trades):>6} | {wr:>6.1f}% | ${total:>8.2f} | ${avg:>6.2f} {emoji}{marker}")
    
    # Mejor resultado
    print("\n" + "=" * 80)
    if results:
        best = max(results, key=lambda x: x['pnl'])
        original = next((r for r in results if r['sl'] == 1.5 and r['tp'] == 3.0), None)
        
        print(f"🏆 MEJOR CONFIGURACIÓN: {best['desc']}")
        print(f"   Trades: {best['trades']} | WinRate: {best['wr']:.1f}% | PnL: ${best['pnl']:.2f}")
        
        if original:
            diff = best['pnl'] - original['pnl']
            print(f"\n⭐ ORIGINAL (SL 1.5x, TP 3.0x):")
            print(f"   Trades: {original['trades']} | WinRate: {original['wr']:.1f}% | PnL: ${original['pnl']:.2f}")
            
            if best['sl'] != 1.5 or best['tp'] != 3.0:
                print(f"\n   Diferencia vs mejor: ${diff:+.2f}")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
