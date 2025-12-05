"""
BACKTEST NOVIEMBRE 2025 - LEVERAGE 20x
======================================
Usando EXACTAMENTE la misma lógica del bot ganadora v3
Período: 1 Nov 2025 - 30 Nov 2025
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List

# =============================================================================
# CONFIGURACIÓN - IGUAL QUE BOT GANADORA (excepto leverage 20x)
# =============================================================================
CONFIG = {
    'MARGIN_USD': 100,
    'LEVERAGE': 20,              # ⚡ CAMBIADO A 20x
    'MAX_OPEN_POSITIONS': 1,
    
    # Períodos de indicadores
    'EMA_FAST': 8,
    'EMA_MEDIUM': 20,
    'EMA_SIGNAL': 21,
    'EMA_SLOW': 50,
    'ADX_PERIOD': 14,
    'RSI_PERIOD': 14,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'ATR_PERIOD': 14,
    'VOLUME_SMA_PERIOD': 20,
    'PIVOT_LOOKBACK': 50,
    
    # Umbrales
    'ADX_MIN': 28,
    'RSI_LONG_MIN': 55,
    'RSI_SHORT_MAX': 70,
    'VOLUME_RATIO': 1.2,
    'EMA_EXTENSION_ATR_MULT': 3.0,
    
    # SL/TP
    'SL_ATR_MULT': 1.5,
    'TP_ATR_MULT': 3.0,
    
    # Filtros
    'ATR_MIN_PCT': 0.002,
    'ATR_MAX_PCT': 0.15,
    'MAX_SPREAD_PCT': 0.001,
}

SYMBOLS = [
    'DOGE/USDT', 'OP/USDT', 'ATOM/USDT', 'FIL/USDT', 'ADA/USDT',
    'TRX/USDT', 'DOT/USDT', 'LINK/USDT', 'ARB/USDT', 'APT/USDT'
]

# =============================================================================
# INDICADORES - EXACTAMENTE IGUAL QUE EL BOT
# =============================================================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # EMAs
    df['ema8'] = df['close'].ewm(span=CONFIG['EMA_FAST'], adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=CONFIG['EMA_MEDIUM'], adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=CONFIG['EMA_SIGNAL'], adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=CONFIG['EMA_SLOW'], adjust=False).mean()
    
    # ADX
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = minus_dm.abs()
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_adx = tr.ewm(span=CONFIG['ADX_PERIOD'], adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=CONFIG['ADX_PERIOD'], adjust=False).mean() / atr_adx)
    minus_di = 100 * (minus_dm.ewm(span=CONFIG['ADX_PERIOD'], adjust=False).mean() / atr_adx)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df['adx'] = dx.ewm(span=CONFIG['ADX_PERIOD'], adjust=False).mean()
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(span=CONFIG['RSI_PERIOD'], adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=CONFIG['RSI_PERIOD'], adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    
    # MACD
    ema_fast = close.ewm(span=CONFIG['MACD_FAST'], adjust=False).mean()
    ema_slow = close.ewm(span=CONFIG['MACD_SLOW'], adjust=False).mean()
    df['macd_line'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd_line'].ewm(span=CONFIG['MACD_SIGNAL'], adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']
    
    # ATR
    df['atr'] = tr.ewm(span=CONFIG['ATR_PERIOD'], adjust=False).mean()
    df['atr_pct'] = df['atr'] / df['close']
    
    # Volume
    df['vol_sma'] = df['volume'].rolling(window=CONFIG['VOLUME_SMA_PERIOD']).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma']
    
    # Extension
    df['ema20_dist'] = (df['close'] - df['ema20']).abs() / df['atr']
    
    return df

# =============================================================================
# DETECCIÓN DE PIVOTS - EXACTAMENTE IGUAL QUE EL BOT
# =============================================================================
def detect_higher_low(df: pd.DataFrame, eval_idx: int) -> bool:
    lookback = CONFIG['PIVOT_LOOKBACK']
    pivot_idx = eval_idx - 2
    
    if pivot_idx < 3:
        return False
    
    pivot_low = df['low'].iloc[pivot_idx]
    
    if df['low'].iloc[pivot_idx - 1] <= pivot_low:
        return False
    if df['low'].iloc[pivot_idx + 1] <= pivot_low:
        return False
    
    for i in range(pivot_idx - 3, max(0, pivot_idx - lookback), -1):
        if i <= 0 or i >= len(df) - 1:
            continue
        prev_low = df['low'].iloc[i]
        if df['low'].iloc[i-1] > prev_low and df['low'].iloc[i+1] > prev_low:
            if pivot_low > prev_low:
                return True
            else:
                return False
    
    return False

def detect_lower_high(df: pd.DataFrame, eval_idx: int) -> bool:
    lookback = CONFIG['PIVOT_LOOKBACK']
    pivot_idx = eval_idx - 2
    
    if pivot_idx < 3:
        return False
    
    pivot_high = df['high'].iloc[pivot_idx]
    
    if df['high'].iloc[pivot_idx - 1] >= pivot_high:
        return False
    if df['high'].iloc[pivot_idx + 1] >= pivot_high:
        return False
    
    for i in range(pivot_idx - 3, max(0, pivot_idx - lookback), -1):
        if i <= 0 or i >= len(df) - 1:
            continue
        prev_high = df['high'].iloc[i]
        if df['high'].iloc[i-1] < prev_high and df['high'].iloc[i+1] < prev_high:
            if pivot_high < prev_high:
                return True
            else:
                return False
    
    return False

# =============================================================================
# ANÁLISIS DE SEÑALES - EXACTAMENTE IGUAL QUE EL BOT
# =============================================================================
def check_signal(df: pd.DataFrame, idx: int) -> Optional[str]:
    """Verifica si hay señal en el índice dado"""
    
    if idx < CONFIG['EMA_SLOW'] + 10:
        return None
    
    row = df.iloc[idx]
    
    # Verificar NaN
    required = ['ema8', 'ema20', 'ema21', 'ema50', 'adx', 'rsi', 'macd_hist', 'atr', 'vol_ratio', 'atr_pct', 'ema20_dist']
    if any(pd.isna(row[col]) for col in required):
        return None
    
    # Filtro ATR%
    if not (CONFIG['ATR_MIN_PCT'] <= row['atr_pct'] <= CONFIG['ATR_MAX_PCT']):
        return None
    
    # Filtro ADX
    if row['adx'] < CONFIG['ADX_MIN']:
        return None
    
    # Filtro Extension
    if row['ema20_dist'] >= CONFIG['EMA_EXTENSION_ATR_MULT']:
        return None
    
    # Filtro Volumen
    if row['vol_ratio'] < CONFIG['VOLUME_RATIO']:
        return None
    
    # === Condiciones LONG ===
    long_ema8_21 = row['ema8'] > row['ema21']
    long_close_ema50 = row['close'] > row['ema50']
    long_ema20_50 = row['ema20'] > row['ema50']
    long_rsi = row['rsi'] > CONFIG['RSI_LONG_MIN']
    long_macd = row['macd_hist'] > 0
    long_hl = detect_higher_low(df, idx)
    
    if all([long_ema8_21, long_close_ema50, long_ema20_50, long_rsi, long_macd, long_hl]):
        return 'LONG'
    
    # === Condiciones SHORT ===
    short_ema8_21 = row['ema8'] < row['ema21']
    short_close_ema50 = row['close'] < row['ema50']
    short_ema20_50 = row['ema20'] < row['ema50']
    short_rsi = row['rsi'] < CONFIG['RSI_SHORT_MAX']
    short_macd = row['macd_hist'] < 0
    short_lh = detect_lower_high(df, idx)
    
    if all([short_ema8_21, short_close_ema50, short_ema20_50, short_rsi, short_macd, short_lh]):
        return 'SHORT'
    
    return None

# =============================================================================
# BACKTEST
# =============================================================================
def run_backtest():
    print("=" * 80)
    print("🔥 BACKTEST NOVIEMBRE 2025 - LEVERAGE 20x")
    print("=" * 80)
    print(f"   Período: 1 Nov 2025 - 30 Nov 2025")
    print(f"   Margen por trade: ${CONFIG['MARGIN_USD']}")
    print(f"   Leverage: {CONFIG['LEVERAGE']}x")
    print(f"   Exposición: ${CONFIG['MARGIN_USD'] * CONFIG['LEVERAGE']}")
    print(f"   Max posiciones: {CONFIG['MAX_OPEN_POSITIONS']}")
    print(f"   SL: {CONFIG['SL_ATR_MULT']}x ATR | TP: {CONFIG['TP_ATR_MULT']}x ATR")
    print("=" * 80)
    
    # Conectar a Binance - usar binance normal para datos históricos
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    exchange.load_markets()
    
    # Fechas
    start_date = datetime(2025, 11, 1, 0, 0, 0)
    end_date = datetime(2025, 11, 30, 23, 59, 59)
    
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    # Descargar datos de todos los símbolos
    print("\n📥 Descargando datos históricos...")
    all_data = {}
    
    for symbol in SYMBOLS:
        print(f"   {symbol}...", end=" ")
        try:
            ohlcv = []
            since = start_ts - (200 * 60 * 60 * 1000)  # 200 horas antes para indicadores
            
            while since < end_ts:
                batch = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=1000)
                if not batch:
                    break
                ohlcv.extend(batch)
                since = batch[-1][0] + 1
                if len(batch) < 1000:
                    break
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
            df = calculate_indicators(df)
            all_data[symbol] = df
            print(f"✓ {len(df)} velas")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Simular trading
    print("\n🎯 Ejecutando backtest...")
    
    trades = []
    balance = 0  # Solo tracking de PnL
    position = None
    
    # Iterar por cada hora del período
    current = start_date
    
    while current <= end_date:
        current_ts = current.timestamp() * 1000
        
        # Si hay posición abierta, verificar SL/TP
        if position:
            symbol = position['symbol']
            df = all_data[symbol]
            
            # Buscar la vela actual
            mask = df['timestamp'] == pd.Timestamp(current)
            if mask.any():
                idx = df[mask].index[0]
                candle = df.iloc[idx]
                
                hit_sl = False
                hit_tp = False
                exit_price = None
                
                if position['direction'] == 'LONG':
                    if candle['low'] <= position['sl']:
                        hit_sl = True
                        exit_price = position['sl']
                    elif candle['high'] >= position['tp']:
                        hit_tp = True
                        exit_price = position['tp']
                else:  # SHORT
                    if candle['high'] >= position['sl']:
                        hit_sl = True
                        exit_price = position['sl']
                    elif candle['low'] <= position['tp']:
                        hit_tp = True
                        exit_price = position['tp']
                
                if hit_sl or hit_tp:
                    # Calcular PnL
                    if position['direction'] == 'LONG':
                        pnl_pct = (exit_price - position['entry']) / position['entry']
                    else:
                        pnl_pct = (position['entry'] - exit_price) / position['entry']
                    
                    pnl_usd = pnl_pct * CONFIG['MARGIN_USD'] * CONFIG['LEVERAGE']
                    pnl_usd -= 2  # Comisiones estimadas
                    
                    balance += pnl_usd
                    
                    result = 'TP ✅' if hit_tp else 'SL ❌'
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current,
                        'symbol': symbol,
                        'direction': position['direction'],
                        'entry': position['entry'],
                        'exit': exit_price,
                        'pnl': pnl_usd,
                        'result': result
                    })
                    
                    position = None
        
        # Si no hay posición, buscar señales
        if position is None:
            for symbol in SYMBOLS:
                df = all_data[symbol]
                
                # Buscar índice de la vela actual
                mask = df['timestamp'] == pd.Timestamp(current)
                if not mask.any():
                    continue
                
                idx = df[mask].index[0]
                
                # Verificar señal (evaluamos idx-1 que sería la vela cerrada)
                if idx < 2:
                    continue
                    
                signal = check_signal(df, idx - 1)
                
                if signal:
                    row = df.iloc[idx - 1]
                    entry_price = df.iloc[idx]['open']  # Entramos en apertura siguiente
                    atr = row['atr']
                    
                    if signal == 'LONG':
                        sl = entry_price - atr * CONFIG['SL_ATR_MULT']
                        tp = entry_price + atr * CONFIG['TP_ATR_MULT']
                    else:
                        sl = entry_price + atr * CONFIG['SL_ATR_MULT']
                        tp = entry_price - atr * CONFIG['TP_ATR_MULT']
                    
                    position = {
                        'symbol': symbol,
                        'direction': signal,
                        'entry': entry_price,
                        'entry_time': current,
                        'sl': sl,
                        'tp': tp,
                        'atr': atr
                    }
                    break  # Solo 1 posición
        
        current += timedelta(hours=1)
    
    # Resultados
    print("\n" + "=" * 80)
    print("📊 RESULTADOS BACKTEST NOVIEMBRE 2025 - 20x LEVERAGE")
    print("=" * 80)
    
    if trades:
        df_trades = pd.DataFrame(trades)
        
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] <= 0]
        
        total_pnl = df_trades['pnl'].sum()
        win_rate = len(wins) / len(df_trades) * 100
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        
        print(f"\n   📈 RESUMEN:")
        print(f"   {'─' * 40}")
        print(f"   Total trades:     {len(trades)}")
        print(f"   Ganadores:        {len(wins)} ({win_rate:.1f}%)")
        print(f"   Perdedores:       {len(losses)} ({100-win_rate:.1f}%)")
        print(f"   {'─' * 40}")
        print(f"   Ganancia promedio: ${avg_win:.2f}")
        print(f"   Pérdida promedio:  ${avg_loss:.2f}")
        print(f"   {'─' * 40}")
        print(f"   💰 PnL TOTAL:      ${total_pnl:.2f}")
        print(f"   {'─' * 40}")
        
        # Por símbolo
        print(f"\n   📊 POR SÍMBOLO:")
        by_symbol = df_trades.groupby('symbol').agg({
            'pnl': ['count', 'sum', 'mean']
        }).round(2)
        by_symbol.columns = ['Trades', 'PnL Total', 'PnL Promedio']
        by_symbol = by_symbol.sort_values('PnL Total', ascending=False)
        print(by_symbol.to_string())
        
        # Últimos trades
        print(f"\n   📋 ÚLTIMOS 10 TRADES:")
        print(f"   {'─' * 70}")
        for _, t in df_trades.tail(10).iterrows():
            print(f"   {t['entry_time'].strftime('%m/%d %H:%M')} | {t['symbol']:<12} | {t['direction']:<5} | ${t['pnl']:>7.2f} | {t['result']}")
        
        # Comparativa 10x vs 20x
        pnl_10x = total_pnl / 2  # Aproximación
        print(f"\n   ⚖️ COMPARATIVA:")
        print(f"   {'─' * 40}")
        print(f"   Con 10x leverage: ~${pnl_10x:.2f}")
        print(f"   Con 20x leverage:  ${total_pnl:.2f}")
        print(f"   Diferencia:        ${total_pnl - pnl_10x:.2f} ({((total_pnl/pnl_10x)-1)*100:.0f}% más)")
        
    else:
        print("   ❌ No se ejecutaron trades")
    
    print("\n" + "=" * 80)
    print("✅ BACKTEST COMPLETADO")
    print("=" * 80)

if __name__ == "__main__":
    run_backtest()
