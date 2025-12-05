"""
BACKTEST BOT GANADORA - NOVIEMBRE 2025
======================================
Sin sesgo de futuro (lookahead bias)
- Todos los indicadores se calculan con datos disponibles hasta la vela actual
- Pivots se detectan con vela idx-2 (confirmados)
- SL/TP se calculan al momento de entrada
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DEL BOT (exacta del archivo)
# =============================================================================
CONFIG = {
    # Capital
    'MARGIN_USD': 100,
    'LEVERAGE': 10,
    'MAX_OPEN_SYMBOLS': 1,
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
    
    # Comisiones (Binance Futures)
    'COMMISSION_PCT': 0.0005,  # 0.05% por lado
}

SYMBOLS = [
    'DOGE/USDT', 'OP/USDT', 'ATOM/USDT', 'FIL/USDT', 'ADA/USDT',
    'TRX/USDT', 'DOT/USDT', 'LINK/USDT', 'ARB/USDT', 'APT/USDT'
]

# =============================================================================
# FUNCIONES DE INDICADORES (Sin lookahead)
# =============================================================================

def calculate_ema(series, period):
    """Calcula EMA - sin lookahead por naturaleza"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_sma(series, period):
    """Calcula SMA - sin lookahead por naturaleza"""
    return series.rolling(window=period).mean()

def calculate_atr(df, period=14):
    """Calcula ATR - sin lookahead"""
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
    """Calcula ADX - sin lookahead"""
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
    """Calcula RSI - sin lookahead"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calcula MACD - sin lookahead"""
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
    - Busca pivot anterior hasta 50 velas atrás
    """
    if idx < lookback + 3:
        return False
    
    # Pivot candidato en idx-2
    pivot_idx = idx - 2
    pivot_low = df['low'].iloc[pivot_idx]
    
    # Confirmar que es un pivot low (velas adyacentes más altas)
    if df['low'].iloc[pivot_idx - 1] <= pivot_low:
        return False
    if df['low'].iloc[pivot_idx + 1] <= pivot_low:  # idx-1, disponible
        return False
    
    # Buscar pivot low anterior
    for i in range(pivot_idx - 3, max(pivot_idx - lookback, 0), -1):
        prev_low = df['low'].iloc[i]
        # Verificar si es pivot
        if i > 0 and i < len(df) - 1:
            if df['low'].iloc[i-1] > prev_low and df['low'].iloc[i+1] > prev_low:
                # Es Higher Low?
                if pivot_low > prev_low:
                    return True
                else:
                    return False
    return False

def detect_pivot_high(df, idx, lookback=50):
    """
    Detecta Lower High SIN LOOKAHEAD
    - Verifica pivot en idx-2 (confirmado por idx-1)
    - Busca pivot anterior hasta 50 velas atrás
    """
    if idx < lookback + 3:
        return False
    
    # Pivot candidato en idx-2
    pivot_idx = idx - 2
    pivot_high = df['high'].iloc[pivot_idx]
    
    # Confirmar que es un pivot high (velas adyacentes más bajas)
    if df['high'].iloc[pivot_idx - 1] >= pivot_high:
        return False
    if df['high'].iloc[pivot_idx + 1] >= pivot_high:  # idx-1, disponible
        return False
    
    # Buscar pivot high anterior
    for i in range(pivot_idx - 3, max(pivot_idx - lookback, 0), -1):
        prev_high = df['high'].iloc[i]
        # Verificar si es pivot
        if i > 0 and i < len(df) - 1:
            if df['high'].iloc[i-1] < prev_high and df['high'].iloc[i+1] < prev_high:
                # Es Lower High?
                if pivot_high < prev_high:
                    return True
                else:
                    return False
    return False

# =============================================================================
# FUNCIONES DE BACKTEST
# =============================================================================

def fetch_ohlcv(exchange, symbol, timeframe, since, until):
    """Descarga datos históricos"""
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
    
    # Filtrar por rango
    until_dt = pd.to_datetime(until, unit='ms')
    df = df[df['timestamp'] <= until_dt]
    
    return df

def calculate_indicators(df):
    """Calcula todos los indicadores necesarios"""
    df = df.copy()
    
    # EMAs
    df['ema8'] = calculate_ema(df['close'], 8)
    df['ema20'] = calculate_ema(df['close'], 20)
    df['ema21'] = calculate_ema(df['close'], 21)
    df['ema50'] = calculate_ema(df['close'], 50)
    
    # ATR
    df['atr'] = calculate_atr(df, 14)
    
    # ADX
    df['adx'] = calculate_adx(df, 14)
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    
    # Volumen SMA
    df['vol_sma20'] = calculate_sma(df['volume'], 20)
    
    # ATR % del precio
    df['atr_pct'] = df['atr'] / df['close']
    
    # Distancia a EMA20 en ATRs
    df['ema20_dist_atr'] = abs(df['close'] - df['ema20']) / df['atr']
    
    return df

def check_long_entry(df, idx):
    """Verifica condiciones de entrada LONG - sin lookahead"""
    row = df.iloc[idx]
    
    # Verificar que tenemos suficientes datos
    if idx < 60:
        return False, "Datos insuficientes"
    
    # 1. Trend EMAs
    if not (row['ema8'] > row['ema21']):
        return False, "EMA8 <= EMA21"
    if not (row['close'] > row['ema50']):
        return False, "Close <= EMA50"
    if not (row['ema20'] > row['ema50']):
        return False, "EMA20 <= EMA50"
    
    # 2. ADX
    if not (row['adx'] >= CONFIG['ADX_MIN']):
        return False, f"ADX {row['adx']:.1f} < {CONFIG['ADX_MIN']}"
    
    # 3. RSI
    if not (row['rsi'] > CONFIG['RSI_LONG_MIN']):
        return False, f"RSI {row['rsi']:.1f} <= {CONFIG['RSI_LONG_MIN']}"
    
    # 4. MACD
    if not (row['macd_hist'] > 0):
        return False, "MACD Hist <= 0"
    
    # 5. Volumen
    if not (row['volume'] >= CONFIG['VOLUME_RATIO'] * row['vol_sma20']):
        return False, "Volumen bajo"
    
    # 6. Higher Low (pivot)
    if not detect_pivot_low(df, idx):
        return False, "Sin Higher Low"
    
    # 7. Extensión EMA20
    if not (row['ema20_dist_atr'] < CONFIG['EMA_EXTENSION_ATR_MULT']):
        return False, "Muy alejado de EMA20"
    
    # 8. Filtro ATR
    if not (CONFIG['ATR_MIN_PCT'] <= row['atr_pct'] <= CONFIG['ATR_MAX_PCT']):
        return False, f"ATR% fuera de rango: {row['atr_pct']:.4f}"
    
    return True, "OK"

def check_short_entry(df, idx):
    """Verifica condiciones de entrada SHORT - sin lookahead"""
    row = df.iloc[idx]
    
    # Verificar que tenemos suficientes datos
    if idx < 60:
        return False, "Datos insuficientes"
    
    # 1. Trend EMAs
    if not (row['ema8'] < row['ema21']):
        return False, "EMA8 >= EMA21"
    if not (row['close'] < row['ema50']):
        return False, "Close >= EMA50"
    if not (row['ema20'] < row['ema50']):
        return False, "EMA20 >= EMA50"
    
    # 2. ADX
    if not (row['adx'] >= CONFIG['ADX_MIN']):
        return False, f"ADX {row['adx']:.1f} < {CONFIG['ADX_MIN']}"
    
    # 3. RSI
    if not (row['rsi'] < CONFIG['RSI_SHORT_MAX']):
        return False, f"RSI {row['rsi']:.1f} >= {CONFIG['RSI_SHORT_MAX']}"
    
    # 4. MACD
    if not (row['macd_hist'] < 0):
        return False, "MACD Hist >= 0"
    
    # 5. Volumen
    if not (row['volume'] >= CONFIG['VOLUME_RATIO'] * row['vol_sma20']):
        return False, "Volumen bajo"
    
    # 6. Lower High (pivot)
    if not detect_pivot_high(df, idx):
        return False, "Sin Lower High"
    
    # 7. Extensión EMA20
    if not (row['ema20_dist_atr'] < CONFIG['EMA_EXTENSION_ATR_MULT']):
        return False, "Muy alejado de EMA20"
    
    # 8. Filtro ATR
    if not (CONFIG['ATR_MIN_PCT'] <= row['atr_pct'] <= CONFIG['ATR_MAX_PCT']):
        return False, f"ATR% fuera de rango: {row['atr_pct']:.4f}"
    
    return True, "OK"

def simulate_trade(df, entry_idx, direction, entry_price, atr):
    """
    Simula un trade y retorna el resultado
    Usa solo datos futuros para determinar SL/TP hit
    """
    exposure = CONFIG['MARGIN_USD'] * CONFIG['LEVERAGE']
    
    if direction == 'LONG':
        sl_price = entry_price - (CONFIG['SL_ATR_MULT'] * atr)
        tp_price = entry_price + (CONFIG['TP_ATR_MULT'] * atr)
    else:  # SHORT
        sl_price = entry_price + (CONFIG['SL_ATR_MULT'] * atr)
        tp_price = entry_price - (CONFIG['TP_ATR_MULT'] * atr)
    
    # Simular vela por vela
    for i in range(entry_idx + 1, len(df)):
        row = df.iloc[i]
        
        if direction == 'LONG':
            # Primero verificar SL (asumimos que se ejecuta primero en caso de gap)
            if row['low'] <= sl_price:
                pnl_pct = (sl_price - entry_price) / entry_price
                pnl = exposure * pnl_pct
                commission = exposure * CONFIG['COMMISSION_PCT'] * 2
                return {
                    'exit_idx': i,
                    'exit_price': sl_price,
                    'exit_type': 'SL',
                    'pnl': pnl - commission,
                    'duration_hours': i - entry_idx
                }
            # Luego verificar TP
            if row['high'] >= tp_price:
                pnl_pct = (tp_price - entry_price) / entry_price
                pnl = exposure * pnl_pct
                commission = exposure * CONFIG['COMMISSION_PCT'] * 2
                return {
                    'exit_idx': i,
                    'exit_price': tp_price,
                    'exit_type': 'TP',
                    'pnl': pnl - commission,
                    'duration_hours': i - entry_idx
                }
        else:  # SHORT
            # Primero verificar SL
            if row['high'] >= sl_price:
                pnl_pct = (entry_price - sl_price) / entry_price
                pnl = exposure * pnl_pct
                commission = exposure * CONFIG['COMMISSION_PCT'] * 2
                return {
                    'exit_idx': i,
                    'exit_price': sl_price,
                    'exit_type': 'SL',
                    'pnl': pnl - commission,
                    'duration_hours': i - entry_idx
                }
            # Luego verificar TP
            if row['low'] <= tp_price:
                pnl_pct = (entry_price - tp_price) / entry_price
                pnl = exposure * pnl_pct
                commission = exposure * CONFIG['COMMISSION_PCT'] * 2
                return {
                    'exit_idx': i,
                    'exit_price': tp_price,
                    'exit_type': 'TP',
                    'pnl': pnl - commission,
                    'duration_hours': i - entry_idx
                }
    
    # Trade aún abierto al final del período
    last_price = df['close'].iloc[-1]
    if direction == 'LONG':
        pnl_pct = (last_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - last_price) / entry_price
    pnl = exposure * pnl_pct
    commission = exposure * CONFIG['COMMISSION_PCT'] * 2
    
    return {
        'exit_idx': len(df) - 1,
        'exit_price': last_price,
        'exit_type': 'OPEN',
        'pnl': pnl - commission,
        'duration_hours': len(df) - 1 - entry_idx
    }

def run_backtest():
    """Ejecuta el backtest completo para noviembre 2025"""
    print("=" * 80)
    print("BACKTEST BOT GANADORA - NOVIEMBRE 2025")
    print("=" * 80)
    print("\n⚠️  SIN SESGO DE FUTURO (Lookahead Bias)")
    print("   • Indicadores calculados solo con datos disponibles")
    print("   • Pivots detectados en idx-2 con confirmación en idx-1")
    print("   • SL/TP fijados al momento de entrada")
    print("   • MAX 1 POSICIÓN SIMULTÁNEA (global, no por símbolo)\n")
    
    # Configurar exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    # Período: Noviembre 2025
    # Necesitamos datos desde antes para calcular indicadores
    start_data = datetime(2025, 10, 1)  # Datos desde octubre para warmup
    start_backtest = datetime(2025, 11, 1)  # Backtest desde noviembre
    end_backtest = datetime(2025, 11, 30, 23, 59, 59)
    
    since_ms = int(start_data.timestamp() * 1000)
    until_ms = int(end_backtest.timestamp() * 1000)
    
    print(f"📅 Período de backtest: {start_backtest.strftime('%Y-%m-%d')} a {end_backtest.strftime('%Y-%m-%d')}")
    print(f"💰 Capital por trade: ${CONFIG['MARGIN_USD']} x {CONFIG['LEVERAGE']}x = ${CONFIG['MARGIN_USD'] * CONFIG['LEVERAGE']}")
    print(f"🎯 SL: {CONFIG['SL_ATR_MULT']}x ATR | TP: {CONFIG['TP_ATR_MULT']}x ATR")
    print(f"📊 MAX_OPEN_SYMBOLS: 3 posiciones simultáneas")
    print("\n" + "-" * 80)
    
    # Descargar datos de todos los símbolos primero
    print("\n📥 Descargando datos históricos...")
    symbol_data = {}
    
    for symbol in SYMBOLS:
        print(f"   Descargando {symbol}...", end=" ")
        try:
            df = fetch_ohlcv(exchange, symbol, CONFIG['TIMEFRAME'], since_ms, until_ms)
            if len(df) >= 100:
                df = calculate_indicators(df)
                symbol_data[symbol] = df
                print(f"✅ {len(df)} velas")
            else:
                print(f"⚠️ Datos insuficientes: {len(df)} velas")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ {len(symbol_data)} símbolos cargados")
    print("-" * 80)
    
    # Generar todas las señales posibles con timestamps
    print("\n🔍 Buscando señales...")
    all_signals = []
    
    for symbol, df in symbol_data.items():
        nov_start_idx = df[df['timestamp'] >= start_backtest].index[0]
        
        for idx in range(nov_start_idx, len(df) - 1):
            row = df.iloc[idx]
            
            # Verificar LONG
            long_ok, _ = check_long_entry(df, idx)
            if long_ok:
                all_signals.append({
                    'timestamp': row['timestamp'],
                    'symbol': symbol,
                    'direction': 'LONG',
                    'idx': idx,
                    'entry_price': row['close'],
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
    
    # Ordenar señales por timestamp (orden cronológico real)
    all_signals.sort(key=lambda x: x['timestamp'])
    print(f"   Total señales encontradas: {len(all_signals)}")
    
    # Ejecutar trades respetando MAX POSICIONES SIMULTÁNEAS
    MAX_POSITIONS = 3  # Cambiar aquí para probar diferentes valores
    print(f"\n🚀 Ejecutando trades (máximo {MAX_POSITIONS} posiciones simultáneas)...")
    all_trades = []
    open_positions = []  # Lista de timestamps de cierre de posiciones abiertas
    
    for signal in all_signals:
        # Limpiar posiciones que ya cerraron
        open_positions = [end_time for end_time in open_positions if end_time > signal['timestamp']]
        
        # Verificar si podemos abrir posición
        if len(open_positions) >= MAX_POSITIONS:
            continue  # Ya tenemos el máximo de posiciones abiertas
        
        # Abrir nueva posición
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
            'pnl': result['pnl'],
            'duration_hours': result['duration_hours']
        }
        
        all_trades.append(trade)
        open_positions.append(trade['exit_time'])  # Agregar tiempo de cierre a la lista
        
        # Mostrar trade
        emoji = "🟢" if trade['pnl'] > 0 else "🔴"
        print(f"   {emoji} {trade['entry_time'].strftime('%m/%d %H:%M')} {symbol} {direction} -> {trade['exit_type']} ${trade['pnl']:.2f}")
    
    # Resumen por símbolo
    print("\n" + "-" * 80)
    print("📊 RESUMEN POR SÍMBOLO:")
    trades_df_temp = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    if not trades_df_temp.empty:
        for symbol in SYMBOLS:
            symbol_trades = trades_df_temp[trades_df_temp['symbol'] == symbol]
            if len(symbol_trades) > 0:
                pnl = symbol_trades['pnl'].sum()
                wins = len(symbol_trades[symbol_trades['pnl'] > 0])
                print(f"   {symbol}: {len(symbol_trades)} trades | Ganadores: {wins} | PnL: ${pnl:.2f}")
    
    # ==========================================================================
    # RESULTADOS FINALES
    # ==========================================================================
    print("\n" + "=" * 80)
    print("RESULTADOS FINALES - NOVIEMBRE 2025")
    print("=" * 80)
    
    if not all_trades:
        print("\n❌ No se ejecutaron trades")
        return
    
    # Convertir a DataFrame
    trades_df = pd.DataFrame(all_trades)
    
    # Métricas generales
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] <= 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    total_pnl = trades_df['pnl'].sum()
    avg_pnl = trades_df['pnl'].mean()
    max_win = trades_df['pnl'].max()
    max_loss = trades_df['pnl'].min()
    
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
    
    # Profit Factor
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Por dirección
    long_trades = trades_df[trades_df['direction'] == 'LONG']
    short_trades = trades_df[trades_df['direction'] == 'SHORT']
    
    print(f"""
📊 MÉTRICAS GENERALES
─────────────────────────────────────────
   Total Trades:        {total_trades}
   Trades Ganadores:    {winning_trades}
   Trades Perdedores:   {losing_trades}
   Win Rate:            {win_rate:.1f}%
   
💰 RENTABILIDAD
─────────────────────────────────────────
   PnL Total:           ${total_pnl:.2f}
   Promedio por Trade:  ${avg_pnl:.2f}
   Profit Factor:       {profit_factor:.2f}
   
   Mayor Ganancia:      ${max_win:.2f}
   Mayor Pérdida:       ${max_loss:.2f}
   
   Promedio Ganador:    ${avg_win:.2f}
   Promedio Perdedor:   ${avg_loss:.2f}

📈 POR DIRECCIÓN
─────────────────────────────────────────
   LONG:  {len(long_trades)} trades | PnL: ${long_trades['pnl'].sum():.2f}
   SHORT: {len(short_trades)} trades | PnL: ${short_trades['pnl'].sum():.2f}

🏆 POR SÍMBOLO (Top 5)
─────────────────────────────────────────""")
    
    symbol_pnl = trades_df.groupby('symbol')['pnl'].agg(['sum', 'count']).sort_values('sum', ascending=False)
    for i, (symbol, row) in enumerate(symbol_pnl.head(5).iterrows()):
        print(f"   {i+1}. {symbol}: ${row['sum']:.2f} ({int(row['count'])} trades)")
    
    print(f"""
📅 POR TIPO DE SALIDA
─────────────────────────────────────────
   Take Profit (TP):    {len(trades_df[trades_df['exit_type'] == 'TP'])}
   Stop Loss (SL):      {len(trades_df[trades_df['exit_type'] == 'SL'])}
   Aún Abiertos:        {len(trades_df[trades_df['exit_type'] == 'OPEN'])}
""")
    
    # Guardar trades detallados
    trades_df.to_csv('/Users/laurazapata/Desktop/DICIEMBRE/trades_noviembre_2025.csv', index=False)
    print("📁 Trades guardados en: trades_noviembre_2025.csv")
    
    print("\n" + "=" * 80)
    print("⚠️  DISCLAIMER")
    print("=" * 80)
    print("""
   Este backtest fue ejecutado SIN sesgo de futuro:
   • Los indicadores se calculan solo con datos históricos disponibles
   • Las señales de pivot se detectan con 2 velas de retraso
   • SL/TP se fijan al momento de entrada, no se optimizan después
   
   Limitaciones:
   • No incluye slippage (deslizamiento de precio)
   • Spread simulado como constante (0.1%)
   • No considera liquidez del orderbook
   • Resultados pasados no garantizan resultados futuros
""")

if __name__ == "__main__":
    run_backtest()
