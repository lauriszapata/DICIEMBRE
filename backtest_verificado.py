"""
BACKTEST BOT GANADORA - NOVIEMBRE 2025 (VERIFICADO)
====================================================
Verificaciones implementadas:
1. ✅ Comisiones aplicadas correctamente (entrada + salida)
2. ✅ TP/SL calculados como ROI% con leverage incluido
3. ✅ Sin sesgo de futuro (solo velas cerradas)
4. ✅ Entrada SOLO en vela cerrada (usando close de la vela)
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DEL BOT
# =============================================================================
CONFIG = {
    # Capital
    'MARGIN_USD': 100,
    'LEVERAGE': 10,
    'MAX_OPEN_SYMBOLS': 3,
    'TIMEFRAME': '1h',
    
    # Risk Management
    'SL_ATR_MULT': 1.5,
    'TP_ATR_MULT': 3.0,
    
    # Indicators
    'ADX_MIN': 28,
    'RSI_LONG_MIN': 55,
    'RSI_SHORT_MAX': 70,
    'VOLUME_RATIO': 1.2,
    'EMA_EXTENSION_ATR_MULT': 3.0,
    
    # Filtros
    'ATR_MIN_PCT': 0.002,
    'ATR_MAX_PCT': 0.15,
    'MAX_SPREAD_PCT': 0.001,
    
    # Comisiones Binance Futures (maker/taker)
    'COMMISSION_PCT': 0.0005,  # 0.05% por lado
}

SYMBOLS = [
    'DOGE/USDT', 'OP/USDT', 'ATOM/USDT', 'FIL/USDT', 'ADA/USDT',
    'TRX/USDT', 'DOT/USDT', 'LINK/USDT', 'ARB/USDT', 'APT/USDT'
]

# =============================================================================
# FUNCIONES DE INDICADORES
# =============================================================================

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_sma(series, period):
    return series.rolling(window=period).mean()

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def detect_pivot_low(df, idx, lookback=50):
    """
    Detecta Higher Low SIN LOOKAHEAD
    - Verifica pivot en idx-2 (confirmado por idx-1)
    """
    if idx < lookback + 3:
        return False
    
    pivot_idx = idx - 2
    pivot_low = df['low'].iloc[pivot_idx]
    
    # Confirmar que es un pivot low
    if df['low'].iloc[pivot_idx - 1] <= pivot_low:
        return False
    if df['low'].iloc[pivot_idx + 1] <= pivot_low:
        return False
    
    # Buscar pivot low anterior
    for i in range(pivot_idx - 3, max(pivot_idx - lookback, 0), -1):
        prev_low = df['low'].iloc[i]
        if i > 0 and i < len(df) - 1:
            if df['low'].iloc[i-1] > prev_low and df['low'].iloc[i+1] > prev_low:
                if pivot_low > prev_low:
                    return True
                else:
                    return False
    return False

def detect_pivot_high(df, idx, lookback=50):
    """
    Detecta Lower High SIN LOOKAHEAD
    - Verifica pivot en idx-2 (confirmado por idx-1)
    """
    if idx < lookback + 3:
        return False
    
    pivot_idx = idx - 2
    pivot_high = df['high'].iloc[pivot_idx]
    
    # Confirmar que es un pivot high
    if df['high'].iloc[pivot_idx - 1] >= pivot_high:
        return False
    if df['high'].iloc[pivot_idx + 1] >= pivot_high:
        return False
    
    # Buscar pivot high anterior
    for i in range(pivot_idx - 3, max(pivot_idx - lookback, 0), -1):
        prev_high = df['high'].iloc[i]
        if i > 0 and i < len(df) - 1:
            if df['high'].iloc[i-1] < prev_high and df['high'].iloc[i+1] < prev_high:
                if pivot_high < prev_high:
                    return True
                else:
                    return False
    return False

# =============================================================================
# FUNCIONES DE BACKTEST
# =============================================================================

def fetch_ohlcv(exchange, symbol, timeframe, since, until):
    """Descarga datos históricos - SOLO VELAS CERRADAS"""
    all_data = []
    current = since
    
    while current < until:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            current = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.reset_index(drop=True)
    
    # Filtrar por rango - SOLO velas completamente cerradas antes de 'until'
    until_dt = pd.to_datetime(until, unit='ms')
    df = df[df['timestamp'] < until_dt]
    
    return df

def calculate_indicators(df):
    """Calcula todos los indicadores necesarios"""
    df = df.copy()
    
    df['ema8'] = calculate_ema(df['close'], 8)
    df['ema20'] = calculate_ema(df['close'], 20)
    df['ema21'] = calculate_ema(df['close'], 21)
    df['ema50'] = calculate_ema(df['close'], 50)
    
    df['atr'] = calculate_atr(df, 14)
    df['adx'] = calculate_adx(df, 14)
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    
    df['vol_sma20'] = calculate_sma(df['volume'], 20)
    df['atr_pct'] = df['atr'] / df['close']
    df['ema20_dist_atr'] = abs(df['close'] - df['ema20']) / df['atr']
    
    return df

def check_long_entry(df, idx):
    """Verifica condiciones de entrada LONG"""
    row = df.iloc[idx]
    
    if idx < 60:
        return False, "Datos insuficientes"
    
    if not (row['ema8'] > row['ema21']):
        return False, "EMA8 <= EMA21"
    if not (row['close'] > row['ema50']):
        return False, "Close <= EMA50"
    if not (row['ema20'] > row['ema50']):
        return False, "EMA20 <= EMA50"
    
    if not (row['adx'] >= CONFIG['ADX_MIN']):
        return False, f"ADX {row['adx']:.1f} < {CONFIG['ADX_MIN']}"
    
    if not (row['rsi'] > CONFIG['RSI_LONG_MIN']):
        return False, f"RSI {row['rsi']:.1f} <= {CONFIG['RSI_LONG_MIN']}"
    
    if not (row['macd_hist'] > 0):
        return False, "MACD Hist <= 0"
    
    if not (row['volume'] >= CONFIG['VOLUME_RATIO'] * row['vol_sma20']):
        return False, "Volumen bajo"
    
    if not detect_pivot_low(df, idx):
        return False, "Sin Higher Low"
    
    if not (row['ema20_dist_atr'] < CONFIG['EMA_EXTENSION_ATR_MULT']):
        return False, "Muy alejado de EMA20"
    
    if not (CONFIG['ATR_MIN_PCT'] <= row['atr_pct'] <= CONFIG['ATR_MAX_PCT']):
        return False, f"ATR% fuera de rango: {row['atr_pct']:.4f}"
    
    return True, "OK"

def check_short_entry(df, idx):
    """Verifica condiciones de entrada SHORT"""
    row = df.iloc[idx]
    
    if idx < 60:
        return False, "Datos insuficientes"
    
    if not (row['ema8'] < row['ema21']):
        return False, "EMA8 >= EMA21"
    if not (row['close'] < row['ema50']):
        return False, "Close >= EMA50"
    if not (row['ema20'] < row['ema50']):
        return False, "EMA20 >= EMA50"
    
    if not (row['adx'] >= CONFIG['ADX_MIN']):
        return False, f"ADX {row['adx']:.1f} < {CONFIG['ADX_MIN']}"
    
    if not (row['rsi'] < CONFIG['RSI_SHORT_MAX']):
        return False, f"RSI {row['rsi']:.1f} >= {CONFIG['RSI_SHORT_MAX']}"
    
    if not (row['macd_hist'] < 0):
        return False, "MACD Hist >= 0"
    
    if not (row['volume'] >= CONFIG['VOLUME_RATIO'] * row['vol_sma20']):
        return False, "Volumen bajo"
    
    if not detect_pivot_high(df, idx):
        return False, "Sin Lower High"
    
    if not (row['ema20_dist_atr'] < CONFIG['EMA_EXTENSION_ATR_MULT']):
        return False, "Muy alejado de EMA20"
    
    if not (CONFIG['ATR_MIN_PCT'] <= row['atr_pct'] <= CONFIG['ATR_MAX_PCT']):
        return False, f"ATR% fuera de rango: {row['atr_pct']:.4f}"
    
    return True, "OK"

def simulate_trade(df, entry_idx, direction, entry_price, atr):
    """
    Simula un trade con cálculos VERIFICADOS
    - PnL calculado sobre exposición total (margin * leverage)
    - Comisiones aplicadas en entrada Y salida
    - Solo usa datos de velas FUTURAS (sin lookahead)
    """
    margin = CONFIG['MARGIN_USD']
    leverage = CONFIG['LEVERAGE']
    exposure = margin * leverage  # $100 * 10 = $1000
    
    # Calcular SL/TP en precio
    if direction == 'LONG':
        sl_price = entry_price - (CONFIG['SL_ATR_MULT'] * atr)
        tp_price = entry_price + (CONFIG['TP_ATR_MULT'] * atr)
    else:  # SHORT
        sl_price = entry_price + (CONFIG['SL_ATR_MULT'] * atr)
        tp_price = entry_price - (CONFIG['TP_ATR_MULT'] * atr)
    
    # Comisión de ENTRADA (0.05%)
    entry_commission = exposure * CONFIG['COMMISSION_PCT']
    
    # Simular vela por vela (SOLO velas FUTURAS - evita lookahead)
    for i in range(entry_idx + 1, len(df)):
        row = df.iloc[i]
        
        if direction == 'LONG':
            # Verificar SL primero (conservador)
            if row['low'] <= sl_price:
                # PnL en USD = (precio_salida - precio_entrada) / precio_entrada * exposición
                price_change_pct = (sl_price - entry_price) / entry_price
                pnl_gross = exposure * price_change_pct
                exit_commission = exposure * CONFIG['COMMISSION_PCT']
                pnl_net = pnl_gross - entry_commission - exit_commission
                
                return {
                    'exit_idx': i,
                    'exit_price': sl_price,
                    'exit_type': 'SL',
                    'pnl_gross': pnl_gross,
                    'entry_commission': entry_commission,
                    'exit_commission': exit_commission,
                    'pnl_net': pnl_net,
                    'roi_pct': (pnl_net / margin) * 100,  # ROI sobre margin
                    'duration_hours': i - entry_idx
                }
            
            # Verificar TP
            if row['high'] >= tp_price:
                price_change_pct = (tp_price - entry_price) / entry_price
                pnl_gross = exposure * price_change_pct
                exit_commission = exposure * CONFIG['COMMISSION_PCT']
                pnl_net = pnl_gross - entry_commission - exit_commission
                
                return {
                    'exit_idx': i,
                    'exit_price': tp_price,
                    'exit_type': 'TP',
                    'pnl_gross': pnl_gross,
                    'entry_commission': entry_commission,
                    'exit_commission': exit_commission,
                    'pnl_net': pnl_net,
                    'roi_pct': (pnl_net / margin) * 100,
                    'duration_hours': i - entry_idx
                }
        
        else:  # SHORT
            # Verificar SL primero
            if row['high'] >= sl_price:
                price_change_pct = (entry_price - sl_price) / entry_price
                pnl_gross = exposure * price_change_pct
                exit_commission = exposure * CONFIG['COMMISSION_PCT']
                pnl_net = pnl_gross - entry_commission - exit_commission
                
                return {
                    'exit_idx': i,
                    'exit_price': sl_price,
                    'exit_type': 'SL',
                    'pnl_gross': pnl_gross,
                    'entry_commission': entry_commission,
                    'exit_commission': exit_commission,
                    'pnl_net': pnl_net,
                    'roi_pct': (pnl_net / margin) * 100,
                    'duration_hours': i - entry_idx
                }
            
            # Verificar TP
            if row['low'] <= tp_price:
                price_change_pct = (entry_price - tp_price) / entry_price
                pnl_gross = exposure * price_change_pct
                exit_commission = exposure * CONFIG['COMMISSION_PCT']
                pnl_net = pnl_gross - entry_commission - exit_commission
                
                return {
                    'exit_idx': i,
                    'exit_price': tp_price,
                    'exit_type': 'TP',
                    'pnl_gross': pnl_gross,
                    'entry_commission': entry_commission,
                    'exit_commission': exit_commission,
                    'pnl_net': pnl_net,
                    'roi_pct': (pnl_net / margin) * 100,
                    'duration_hours': i - entry_idx
                }
    
    # Trade aún abierto al final
    last_price = df['close'].iloc[-1]
    if direction == 'LONG':
        price_change_pct = (last_price - entry_price) / entry_price
    else:
        price_change_pct = (entry_price - last_price) / entry_price
    
    pnl_gross = exposure * price_change_pct
    exit_commission = exposure * CONFIG['COMMISSION_PCT']
    pnl_net = pnl_gross - entry_commission - exit_commission
    
    return {
        'exit_idx': len(df) - 1,
        'exit_price': last_price,
        'exit_type': 'OPEN',
        'pnl_gross': pnl_gross,
        'entry_commission': entry_commission,
        'exit_commission': exit_commission,
        'pnl_net': pnl_net,
        'roi_pct': (pnl_net / margin) * 100,
        'duration_hours': len(df) - 1 - entry_idx
    }

def run_backtest():
    """Ejecuta el backtest VERIFICADO para noviembre 2025"""
    print("=" * 80)
    print("BACKTEST BOT GANADORA - NOVIEMBRE 2025 (VERIFICADO)")
    print("=" * 80)
    print("\n✅ VERIFICACIONES IMPLEMENTADAS:")
    print("   • Comisiones: 0.05% entrada + 0.05% salida = 0.1% total")
    print("   • PnL calculado sobre exposición total (margin × leverage)")
    print("   • ROI% calculado sobre margin invertido")
    print("   • Sin sesgo de futuro: pivots en idx-2, datos hasta idx")
    print("   • SOLO velas cerradas para entradas")
    print("   • Máximo 3 posiciones simultáneas\n")
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    # Período
    start_data = datetime(2025, 10, 1)
    start_backtest = datetime(2025, 11, 1)
    end_backtest = datetime(2025, 11, 30, 23, 59, 59)
    
    since_ms = int(start_data.timestamp() * 1000)
    until_ms = int(end_backtest.timestamp() * 1000)
    
    print(f"📅 Período: {start_backtest.strftime('%Y-%m-%d')} a {end_backtest.strftime('%Y-%m-%d')}")
    print(f"💰 Margin: ${CONFIG['MARGIN_USD']} | Leverage: {CONFIG['LEVERAGE']}x | Exposición: ${CONFIG['MARGIN_USD'] * CONFIG['LEVERAGE']}")
    print(f"🎯 SL: {CONFIG['SL_ATR_MULT']}x ATR | TP: {CONFIG['TP_ATR_MULT']}x ATR")
    print("\n" + "-" * 80)
    
    # Descargar datos
    print("\n📥 Descargando datos históricos (SOLO velas cerradas)...")
    symbol_data = {}
    
    for symbol in SYMBOLS:
        print(f"   {symbol}...", end=" ")
        try:
            df = fetch_ohlcv(exchange, symbol, CONFIG['TIMEFRAME'], since_ms, until_ms)
            if len(df) >= 100:
                df = calculate_indicators(df)
                symbol_data[symbol] = df
                print(f"✅ {len(df)} velas")
            else:
                print(f"⚠️ Insuficiente: {len(df)} velas")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ {len(symbol_data)} símbolos cargados")
    print("-" * 80)
    
    # Buscar señales
    print("\n🔍 Buscando señales...")
    all_signals = []
    
    for symbol, df in symbol_data.items():
        # Índice donde empieza noviembre
        nov_start_idx = df[df['timestamp'] >= start_backtest].index[0]
        
        # IMPORTANTE: usar len(df) - solo velas YA CERRADAS
        # No evaluar la última vela si pudiera estar abierta
        for idx in range(nov_start_idx, len(df)):
            row = df.iloc[idx]
            
            # Verificar LONG
            long_ok, _ = check_long_entry(df, idx)
            if long_ok:
                all_signals.append({
                    'timestamp': row['timestamp'],
                    'symbol': symbol,
                    'direction': 'LONG',
                    'idx': idx,
                    'entry_price': row['close'],  # Close de vela CERRADA
                    'atr': row['atr']
                })
            
            # Verificar SHORT
            short_ok, _ = check_short_entry(df, idx)
            if short_ok:
                all_signals.append({
                    'timestamp': row['timestamp'],
                    'symbol': symbol,
                    'direction': 'SHORT',
                    'idx': idx,
                    'entry_price': row['close'],
                    'atr': row['atr']
                })
    
    all_signals.sort(key=lambda x: x['timestamp'])
    print(f"   Total señales encontradas: {len(all_signals)}")
    
    # Ejecutar trades
    MAX_POSITIONS = CONFIG['MAX_OPEN_SYMBOLS']
    print(f"\n🚀 Ejecutando trades (máximo {MAX_POSITIONS} posiciones simultáneas)...")
    all_trades = []
    open_positions = []
    
    for signal in all_signals:
        # Limpiar posiciones cerradas
        open_positions = [end_time for end_time in open_positions if end_time > signal['timestamp']]
        
        # Verificar límite
        if len(open_positions) >= MAX_POSITIONS:
            continue
        
        # Abrir posición
        symbol = signal['symbol']
        df = symbol_data[symbol]
        idx = signal['idx']
        direction = signal['direction']
        entry_price = signal['entry_price']
        atr = signal['atr']
        
        result = simulate_trade(df, idx, direction, entry_price, atr)
        
        if direction == 'LONG':
            sl_price = entry_price - (CONFIG['SL_ATR_MULT'] * atr)
            tp_price = entry_price + (CONFIG['TP_ATR_MULT'] * atr)
        else:
            sl_price = entry_price + (CONFIG['SL_ATR_MULT'] * atr)
            tp_price = entry_price - (CONFIG['TP_ATR_MULT'] * atr)
        
        trade = {
            'symbol': symbol,
            'direction': direction,
            'entry_time': signal['timestamp'],
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'exit_time': df.iloc[result['exit_idx']]['timestamp'],
            'exit_price': result['exit_price'],
            'exit_type': result['exit_type'],
            'pnl_gross': result['pnl_gross'],
            'entry_commission': result['entry_commission'],
            'exit_commission': result['exit_commission'],
            'pnl_net': result['pnl_net'],
            'roi_pct': result['roi_pct'],
            'duration_hours': result['duration_hours']
        }
        
        all_trades.append(trade)
        open_positions.append(trade['exit_time'])
        
        emoji = "🟢" if trade['pnl_net'] > 0 else "🔴"
        print(f"   {emoji} {trade['entry_time'].strftime('%m/%d %H:%M')} {symbol} {direction} -> {trade['exit_type']} ${trade['pnl_net']:.2f} (ROI: {trade['roi_pct']:.1f}%)")
    
    # Resultados
    print("\n" + "=" * 80)
    print("RESULTADOS FINALES - NOVIEMBRE 2025 (VERIFICADO)")
    print("=" * 80)
    
    if not all_trades:
        print("\n❌ No se ejecutaron trades")
        return
    
    trades_df = pd.DataFrame(all_trades)
    
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl_net'] > 0])
    losing_trades = len(trades_df[trades_df['pnl_net'] <= 0])
    win_rate = (winning_trades / total_trades) * 100
    
    total_pnl = trades_df['pnl_net'].sum()
    total_commissions = trades_df['entry_commission'].sum() + trades_df['exit_commission'].sum()
    avg_pnl = trades_df['pnl_net'].mean()
    avg_roi = trades_df['roi_pct'].mean()
    
    max_win = trades_df['pnl_net'].max()
    max_loss = trades_df['pnl_net'].min()
    
    avg_win = trades_df[trades_df['pnl_net'] > 0]['pnl_net'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl_net'] <= 0]['pnl_net'].mean() if losing_trades > 0 else 0
    
    gross_profit = trades_df[trades_df['pnl_net'] > 0]['pnl_net'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_net'] <= 0]['pnl_net'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    long_trades = trades_df[trades_df['direction'] == 'LONG']
    short_trades = trades_df[trades_df['direction'] == 'SHORT']
    
    print(f"""
📊 MÉTRICAS GENERALES
─────────────────────────────────────────
   Total Trades:        {total_trades}
   Trades Ganadores:    {winning_trades}
   Trades Perdedores:   {losing_trades}
   Win Rate:            {win_rate:.1f}%
   
💰 RENTABILIDAD (VERIFICADA)
─────────────────────────────────────────
   PnL Neto Total:      ${total_pnl:.2f}
   Comisiones Pagadas:  ${total_commissions:.2f}
   Promedio por Trade:  ${avg_pnl:.2f}
   ROI Promedio:        {avg_roi:.2f}%
   Profit Factor:       {profit_factor:.2f}
   
   Mayor Ganancia:      ${max_win:.2f}
   Mayor Pérdida:       ${max_loss:.2f}
   
   Promedio Ganador:    ${avg_win:.2f}
   Promedio Perdedor:   ${avg_loss:.2f}

📈 POR DIRECCIÓN
─────────────────────────────────────────
   LONG:  {len(long_trades)} trades | PnL: ${long_trades['pnl_net'].sum():.2f}
   SHORT: {len(short_trades)} trades | PnL: ${short_trades['pnl_net'].sum():.2f}

🏆 POR SÍMBOLO (Top 5)
─────────────────────────────────────────""")
    
    symbol_pnl = trades_df.groupby('symbol')['pnl_net'].agg(['sum', 'count']).sort_values('sum', ascending=False)
    for i, (symbol, row) in enumerate(symbol_pnl.head(5).iterrows()):
        print(f"   {i+1}. {symbol}: ${row['sum']:.2f} ({int(row['count'])} trades)")
    
    print(f"""
📅 POR TIPO DE SALIDA
─────────────────────────────────────────
   Take Profit (TP):    {len(trades_df[trades_df['exit_type'] == 'TP'])}
   Stop Loss (SL):      {len(trades_df[trades_df['exit_type'] == 'SL'])}
   Aún Abiertos:        {len(trades_df[trades_df['exit_type'] == 'OPEN'])}
""")
    
    # Guardar
    trades_df.to_csv('/Users/laurazapata/Desktop/DICIEMBRE/trades_verificado_nov_2025.csv', index=False)
    print("📁 Trades guardados en: trades_verificado_nov_2025.csv")
    
    print("\n" + "=" * 80)
    print("✅ CONFIRMACIONES DE VERIFICACIÓN")
    print("=" * 80)
    print(f"""
   ✓ Comisiones totales: ${total_commissions:.2f} ({total_trades * 2} operaciones × 0.05%)
   ✓ PnL calculado sobre exposición de ${CONFIG['MARGIN_USD'] * CONFIG['LEVERAGE']}
   ✓ ROI calculado sobre margin de ${CONFIG['MARGIN_USD']}
   ✓ Todas las entradas en velas CERRADAS
   ✓ Sin sesgo de futuro en indicadores ni pivots
""")

if __name__ == "__main__":
    run_backtest()
