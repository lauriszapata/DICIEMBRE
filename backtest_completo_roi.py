"""
BACKTEST COMPLETO BOT REAL - NOVIEMBRE 2025
============================================
Replica EXACTAMENTE bot_ganadora_v3.py

CARACTERÍSTICAS:
- SL/TP ROI fijo: 5% y 10% (0.5% y 1.0% precio)
- Cooldown 1 hora post-pérdida
- Filtros multi-timeframe (15min, 5min, 1min)
- Límite 2 trades/hora por símbolo
- Filtro mercado lateral (<1.5% rango)
- ADX mínimo 30
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN EXACTA DEL BOT REAL
# =============================================================================
CONFIG = {
    # Capital
    'MARGIN_USD': 100,
    'LEVERAGE': 10,
    'MAX_OPEN_POSITIONS': 3,
    'TIMEFRAME': '1h',
    
    # SL/TP ROI FIJO (diferente a backtest ATR)
    'SL_ROI': 0.05,  # 5% ROI = 0.5% movimiento precio
    'TP_ROI': 0.10,  # 10% ROI = 1.0% movimiento precio
    
    # Indicadores
    'EMA_FAST': 8,
    'EMA_SIGNAL': 21,
    'EMA_SLOW': 50,
    'ADX_MIN': 30,  # Bot real usa 30, no 28
    'RSI_LONG_MIN': 55,
    'RSI_SHORT_MAX': 70,
    'VOLUME_RATIO': 1.2,
    'EMA_EXTENSION_ATR_MULT': 3.0,
    
    # Filtros
    'ATR_MIN_PCT': 0.002,
    'ATR_MAX_PCT': 0.15,
    'PIVOT_LOOKBACK': 50,
    
    # Cooldown y límites
    'COOLDOWN_HOURS': 1,
    'MAX_TRADES_PER_HOUR': 2,
    'LATERAL_RANGE_PCT': 0.015,  # 1.5%
    
    # Comisiones
    'COMMISSION_PCT': 0.0005,
}

SYMBOLS = [
    'DOGE/USDT', 'OP/USDT', 'ATOM/USDT', 'FIL/USDT', 'ADA/USDT',
    'TRX/USDT', 'DOT/USDT', 'LINK/USDT', 'ARB/USDT', 'APT/USDT'
]

# =============================================================================
# CLASE DE ESTADO DEL BACKTEST
# =============================================================================
class BacktestState:
    """Mantiene el estado del backtest incluyendo cooldowns y límites"""
    
    def __init__(self):
        self.cooldowns = {}  # {symbol: datetime} - último trade perdedor
        self.trades_hourly = {}  # {symbol: [datetimes]} - trades última hora
        self.open_positions = []  # Lista de timestamps de cierre
        
    def is_in_cooldown(self, symbol: str, current_time: datetime) -> bool:
        """Verifica si símbolo está en cooldown"""
        if symbol not in self.cooldowns:
            return False
        
        cooldown_end = self.cooldowns[symbol] + timedelta(hours=CONFIG['COOLDOWN_HOURS'])
        if current_time >= cooldown_end:
            del self.cooldowns[symbol]
            return False
        
        return True
    
    def add_cooldown(self, symbol: str, timestamp: datetime):
        """Activa cooldown para un símbolo"""
        self.cooldowns[symbol] = timestamp
    
    def check_hourly_limit(self, symbol: str, current_time: datetime) -> bool:
        """Verifica límite de trades por hora"""
        if symbol not in self.trades_hourly:
            self.trades_hourly[symbol] = []
        
        # Limpiar trades > 1 hora
        one_hour_ago = current_time - timedelta(hours=1)
        self.trades_hourly[symbol] = [
            t for t in self.trades_hourly[symbol] if t > one_hour_ago
        ]
        
        # Verificar límite
        if len(self.trades_hourly[symbol]) >= CONFIG['MAX_TRADES_PER_HOUR']:
            return False  # Límite alcanzado
        
        return True  # OK para operar
    
    def add_trade(self, symbol: str, timestamp: datetime):
        """Registra un nuevo trade"""
        if symbol not in self.trades_hourly:
            self.trades_hourly[symbol] = []
        self.trades_hourly[symbol].append(timestamp)

# =============================================================================
# FUNCIONES DE DESCARGA MULTI-TIMEFRAME
# =============================================================================

def fetch_ohlcv_safe(exchange, symbol: str, timeframe: str, since_ms: int, until_ms: int) -> pd.DataFrame:
    """Descarga OHLCV con manejo de errores"""
    all_data = []
    current = since_ms
    
    while current < until_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            current = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"      Error fetching {symbol} {timeframe}: {e}")
            break
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Filtrar por rango
    until_dt = pd.to_datetime(until_ms, unit='ms')
    df = df[df['timestamp'] < until_dt]
    
    return df

def fetch_multi_timeframe(exchange, symbol: str, current_time_1h: datetime, 
                         since_ms: int, until_ms: int) -> Dict[str, pd.DataFrame]:
    """
    Descarga datos de múltiples timeframes para un timestamp específico de 1h.
    
    Retorna dict con keys: '1h', '15m', '5m', '1m'
    """
    dfs = {}
    
    # Para evitar lookahead, solo usamos datos ANTES del current_time_1h
    max_timestamp = current_time_1h
    
    for tf in ['1h', '15m', '5m', '1m']:
        df = fetch_ohlcv_safe(exchange, symbol, tf, since_ms, until_ms)
        if not df.empty:
            # Solo datos hasta current_time_1h (exclusive)
            df = df[df['timestamp'] < max_timestamp]
        dfs[tf] = df
    
    return dfs

# =============================================================================
# CÁLCULO DE INDICADORES
# =============================================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores en un dataframe"""
    if len(df) < 60:
        return df
    
    df = df.copy()
    
    # EMAs
    df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # ATR
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14, adjust=False).mean()
    
    # ADX
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr_adx = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_adx)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_adx)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.ewm(span=14, adjust=False).mean()
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']
    
    # Volume
    df['vol_sma'] = df['volume'].rolling(window=20).mean()
    
    # Auxiliares
    df['atr_pct'] = df['atr'] / df['close']
    df['ema20_dist_atr'] = abs(df['close'] - df['ema20']) / df['atr']
    
    return df

# =============================================================================
# DETECCIÓN DE PIVOTS
# =============================================================================

def detect_higher_low(df: pd.DataFrame, idx: int) -> Tuple[bool, str]:
    """Detecta Higher Low en idx-2"""
    if idx < CONFIG['PIVOT_LOOKBACK'] + 3:
        return False, "Datos insuficientes"
    
    pivot_idx = idx - 2
    pivot_low = df['low'].iloc[pivot_idx]
    
    # Confirmar pivot
    if df['low'].iloc[pivot_idx - 1] <= pivot_low:
        return False, "No es pivot"
    if df['low'].iloc[pivot_idx + 1] <= pivot_low:
        return False, "No es pivot"
    
    # Buscar pivot anterior
    for i in range(pivot_idx - 3, max(0, pivot_idx - CONFIG['PIVOT_LOOKBACK']), -1):
        if i <= 0 or i >= len(df) - 1:
            continue
        prev_low = df['low'].iloc[i]
        if df['low'].iloc[i-1] > prev_low and df['low'].iloc[i+1] > prev_low:
            if pivot_low > prev_low:
                return True, "Higher Low confirmado"
            else:
                return False, "Not higher low"
    
    return False, "Sin pivot anterior"

def detect_lower_high(df: pd.DataFrame, idx: int) -> Tuple[bool, str]:
    """Detecta Lower High en idx-2"""
    if idx < CONFIG['PIVOT_LOOKBACK'] + 3:
        return False, "Datos insuficientes"
    
    pivot_idx = idx - 2
    pivot_high = df['high'].iloc[pivot_idx]
    
    # Confirmar pivot
    if df['high'].iloc[pivot_idx - 1] >= pivot_high:
        return False, "No es pivot"
    if df['high'].iloc[pivot_idx + 1] >= pivot_high:
        return False, "No es pivot"
    
    # Buscar pivot anterior
    for i in range(pivot_idx - 3, max(0, pivot_idx - CONFIG['PIVOT_LOOKBACK']), -1):
        if i <= 0 or i >= len(df) - 1:
            continue
        prev_high = df['high'].iloc[i]
        if df['high'].iloc[i-1] < prev_high and df['high'].iloc[i+1] < prev_high:
            if pivot_high < prev_high:
                return True, "Lower High confirmado"
            else:
                return False, "Not lower high"
    
    return False, "Sin pivot anterior"

# =============================================================================
# FILTROS MULTI-TIMEFRAME
# =============================================================================

def check_15min_ema_filter(dfs: Dict[str, pd.DataFrame], direction: str) -> Tuple[bool, str]:
    """Filtro EMA 15 minutos - requiere 3+ velas con EMAs alineadas"""
    df_15m = dfs.get('15m')
    if df_15m is None or len(df_15m) < 50:
        return False, "Sin datos 15min"
    
    # Calcular EMAs
    df_15m['ema8'] = df_15m['close'].ewm(span=8, adjust=False).mean()
    df_15m['ema21'] = df_15m['close'].ewm(span=21, adjust=False).mean()
    
    # Verificar últimas 4 velas cerradas
    is_long = direction == 'LONG'
    consecutive = 0
    
    for i in range(-5, -1):  # -5, -4, -3, -2
        if abs(i) > len(df_15m):
            continue
        ema8 = df_15m['ema8'].iloc[i]
        ema21 = df_15m['ema21'].iloc[i]
        
        if is_long:
            if ema8 > ema21:
                consecutive += 1
            else:
                consecutive = 0
        else:
            if ema8 < ema21:
                consecutive += 1
            else:
                consecutive = 0
    
    # Verificar precio vs EMA8
    current_close = df_15m['close'].iloc[-2]
    ema8_current = df_15m['ema8'].iloc[-2]
    
    if is_long:
        price_ok = current_close > ema8_current
    else:
        price_ok = current_close < ema8_current
    
    aligned = (consecutive >= 3) and price_ok
    
    if aligned:
        return True, f"15min OK: {consecutive} velas alineadas"
    else:
        return False, f"15min FALLA: {consecutive}/3 velas, precio {'OK' if price_ok else 'FALLA'}"

def check_5min_candle_filter(dfs: Dict[str, pd.DataFrame], direction: str) -> Tuple[bool, str]:
    """Filtro velas 5min - requiere 3/5 checks"""
    df_5m = dfs.get('5m')
    if df_5m is None or len(df_5m) < 50:
        return False, "Sin datos 5min"
    
    # Calcular EMAs
    df_5m['ema8'] = df_5m['close'].ewm(span=8, adjust=False).mean()
    df_5m['ema21'] = df_5m['close'].ewm(span=21, adjust=False).mean()
    
    is_long = direction == 'LONG'
    checks_passed = 0
    
    # Check 1: Velas alineadas (2/3)
    bullish = sum(1 for i in range(-4, -1) if df_5m['close'].iloc[i] > df_5m['open'].iloc[i])
    bearish = sum(1 for i in range(-4, -1) if df_5m['close'].iloc[i] < df_5m['open'].iloc[i])
    if (is_long and bullish >= 2) or (not is_long and bearish >= 2):
        checks_passed += 1
    
    # Check 2: EMAs alineadas
    ema8 = df_5m['ema8'].iloc[-2]
    ema21 = df_5m['ema21'].iloc[-2]
    if (is_long and ema8 > ema21) or (not is_long and ema8 < ema21):
        checks_passed += 1
    
    # Check 3: Pendiente EMA8
    ema8_prev = df_5m['ema8'].iloc[-5]
    slope = ema8 - ema8_prev
    if (is_long and slope > 0) or (not is_long and slope < 0):
        checks_passed += 1
    
    # Check 4: Precio vs EMA21
    current_close = df_5m['close'].iloc[-2]
    if (is_long and current_close > ema21) or (not is_long and current_close < ema21):
        checks_passed += 1
    
    # Check 5: Gap EMAs expandiendo
    gap_current = abs(ema8 - ema21)
    gap_prev = abs(df_5m['ema8'].iloc[-5] - df_5m['ema21'].iloc[-5])
    if gap_current > gap_prev:
        checks_passed += 1
    
    if checks_passed >= 3:
        return True, f"5min OK: {checks_passed}/5 checks"
    else:
        return False, f"5min FALLA: {checks_passed}/5 checks"

def check_1min_candle_filter(dfs: Dict[str, pd.DataFrame], direction: str) -> Tuple[bool, str]:
    """Filtro velas 1min - requiere 4/5 checks"""
    df_1m = dfs.get('1m')
    if df_1m is None or len(df_1m) < 30:
        return False, "Sin datos 1min"
    
    # Calcular EMAs
    df_1m['ema8'] = df_1m['close'].ewm(span=8, adjust=False).mean()
    df_1m['ema21'] = df_1m['close'].ewm(span=21, adjust=False).mean()
    
    is_long = direction == 'LONG'
    checks_passed = 0
    
    # Check 1: Velas alineadas (2/3)
    bullish = sum(1 for i in range(-4, -1) if df_1m['close'].iloc[i] > df_1m['open'].iloc[i])
    bearish = sum(1 for i in range(-4, -1) if df_1m['close'].iloc[i] < df_1m['open'].iloc[i])
    if (is_long and bullish >= 2) or (not is_long and bearish >= 2):
        checks_passed += 1
    
    # Check 2: EMAs alineadas
    ema8 = df_1m['ema8'].iloc[-2]
    ema21 = df_1m['ema21'].iloc[-2]
    if (is_long and ema8 > ema21) or (not is_long and ema8 < ema21):
        checks_passed += 1
    
    # Check 3: Pendiente EMA8
    ema8_prev = df_1m['ema8'].iloc[-5]
    slope = ema8 - ema8_prev
    if (is_long and slope > 0) or (not is_long and slope < 0):
        checks_passed += 1
    
    # Check 4: Precio vs EMA8
    current_close = df_1m['close'].iloc[-2]
    if (is_long and current_close > ema8) or (not is_long and current_close < ema8):
        checks_passed += 1
    
    # Check 5: Momentum
    close_prev = df_1m['close'].iloc[-7]
    momentum = current_close - close_prev
    if (is_long and momentum > 0) or (not is_long and momentum < 0):
        checks_passed += 1
    
    if checks_passed >= 4:
        return True, f"1min OK: {checks_passed}/5 checks"
    else:
        return False, f"1min FALLA: {checks_passed}/5 checks"

def check_lateral_filter(df_1h: pd.DataFrame, idx: int) -> Tuple[bool, str]:
    """Filtro mercado lateral - rechaza si rango < 1.5%"""
    if idx < 4:
        return True, "Datos insuficientes para filtro lateral"
    
    # Últimas 4 velas
    recent = df_1h.iloc[idx-4:idx]
    max_high = recent['high'].max()
    min_low = recent['low'].min()
    avg_price = recent['close'].mean()
    
    range_pct = (max_high - min_low) / avg_price
    
    if range_pct < CONFIG['LATERAL_RANGE_PCT']:
        return False, f"Mercado lateral: rango {range_pct*100:.2f}% < 1.5%"
    
    return True, f"Rango OK: {range_pct*100:.2f}%"

# =============================================================================
# CONDICIONES DE ENTRADA
# =============================================================================

def check_long_entry(dfs: Dict[str, pd.DataFrame], idx: int, state: BacktestState, 
                    symbol: str, current_time: datetime) -> Tuple[bool, str]:
    """Verifica TODAS las condiciones para LONG"""
    df = dfs['1h']
    row = df.iloc[idx]
    
    if idx < 60:
        return False, "Datos insuficientes"
    
    # Cooldown
    if state.is_in_cooldown(symbol, current_time):
        return False, "En cooldown"
    
    # Límite trades/hora
    if not state.check_hourly_limit(symbol, current_time):
        return False, "Límite trades/hora alcanzado"
    
    # Filtro lateral
    lateral_ok, lateral_msg = check_lateral_filter(df, idx)
    if not lateral_ok:
        return False, lateral_msg
    
    # EMAs
    if not (row['ema8'] > row['ema21']):
        return False, "EMA8 <= EMA21"
    if not (row['close'] > row['ema50']):
        return False, "Close <= EMA50"
    if not (row['ema20'] > row['ema50']):
        return False, "EMA20 <= EMA50"
    
    # ADX
    if not (row['adx'] >= CONFIG['ADX_MIN']):
        return False, f"ADX {row['adx']:.1f} < {CONFIG['ADX_MIN']}"
    
    # RSI
    if not (row['rsi'] > CONFIG['RSI_LONG_MIN']):
        return False, f"RSI {row['rsi']:.1f} <= {CONFIG['RSI_LONG_MIN']}"
    
    # MACD
    if not (row['macd_hist'] > 0):
        return False, "MACD Hist <= 0"
    
    # Volumen
    if not (row['volume'] >= CONFIG['VOLUME_RATIO'] * row['vol_sma']):
        return False, "Volumen bajo"
    
    # Higher Low
    hl_ok, hl_msg = detect_higher_low(df, idx)
    if not hl_ok:
        return False, hl_msg
    
    # EMA Extension
    if not (row['ema20_dist_atr'] < CONFIG['EMA_EXTENSION_ATR_MULT']):
        return False, "Muy alejado de EMA20"
    
    # ATR filter
    if not (CONFIG['ATR_MIN_PCT'] <= row['atr_pct'] <= CONFIG['ATR_MAX_PCT']):
        return False, f"ATR% fuera de rango"
    
    # Filtro 15min EMA
    ema15_ok, ema15_msg = check_15min_ema_filter(dfs, 'LONG')
    if not ema15_ok:
        return False, ema15_msg
    
    # Filtro 5min
    candle5_ok, candle5_msg = check_5min_candle_filter(dfs, 'LONG')
    if not candle5_ok:
        return False, candle5_msg
    
    # Filtro 1min
    candle1_ok, candle1_msg = check_1min_candle_filter(dfs, 'LONG')
    if not candle1_ok:
        return False, candle1_msg
    
    return True, "Todas las condiciones OK"

def check_short_entry(dfs: Dict[str, pd.DataFrame], idx: int, state: BacktestState,
                     symbol: str, current_time: datetime) -> Tuple[bool, str]:
    """Verifica TODAS las condiciones para SHORT"""
    df = dfs['1h']
    row = df.iloc[idx]
    
    if idx < 60:
        return False, "Datos insuficientes"
    
    # Cooldown
    if state.is_in_cooldown(symbol, current_time):
        return False, "En cooldown"
    
    # Límite trades/hora
    if not state.check_hourly_limit(symbol, current_time):
        return False, "Límite trades/hora alcanzado"
    
    # Filtro lateral
    lateral_ok, lateral_msg = check_lateral_filter(df, idx)
    if not lateral_ok:
        return False, lateral_msg
    
    # EMAs
    if not (row['ema8'] < row['ema21']):
        return False, "EMA8 >= EMA21"
    if not (row['close'] < row['ema50']):
        return False, "Close >= EMA50"
    if not (row['ema20'] < row['ema50']):
        return False, "EMA20 >= EMA50"
    
    # ADX
    if not (row['adx'] >= CONFIG['ADX_MIN']):
        return False, f"ADX {row['adx']:.1f} < {CONFIG['ADX_MIN']}"
    
    # RSI
    if not (row['rsi'] < CONFIG['RSI_SHORT_MAX']):
        return False, f"RSI {row['rsi']:.1f} >= {CONFIG['RSI_SHORT_MAX']}"
    
    # MACD
    if not (row['macd_hist'] < 0):
        return False, "MACD Hist >= 0"
    
    # Volumen
    if not (row['volume'] >= CONFIG['VOLUME_RATIO'] * row['vol_sma']):
        return False, "Volumen bajo"
    
    # Lower High
    lh_ok, lh_msg = detect_lower_high(df, idx)
    if not lh_ok:
        return False, lh_msg
    
    # EMA Extension
    if not (row['ema20_dist_atr'] < CONFIG['EMA_EXTENSION_ATR_MULT']):
        return False, "Muy alejado de EMA20"
    
    # ATR filter
    if not (CONFIG['ATR_MIN_PCT'] <= row['atr_pct'] <= CONFIG['ATR_MAX_PCT']):
        return False, f"ATR% fuera de rango"
    
    # Filtro 15min EMA
    ema15_ok, ema15_msg = check_15min_ema_filter(dfs, 'SHORT')
    if not ema15_ok:
        return False, ema15_msg
    
    # Filtro 5min
    candle5_ok, candle5_msg = check_5min_candle_filter(dfs, 'SHORT')
    if not candle5_ok:
        return False, candle5_msg
    
    # Filtro 1min
    candle1_ok, candle1_msg = check_1min_candle_filter(dfs, 'SHORT')
    if not candle1_ok:
        return False, candle1_msg
    
    return True, "Todas las condiciones OK"

# =============================================================================
# SIMULACIÓN DE TRADE CON ROI FIJO
# =============================================================================

def simulate_trade_roi_fixed(df: pd.DataFrame, entry_idx: int, direction: str,
                             entry_price: float) -> Dict:
    """Simula trade con SL/TP fijos basados en ROI%"""
    margin = CONFIG['MARGIN_USD']
    leverage = CONFIG['LEVERAGE']
    exposure = margin * leverage  # $1000
    
    # SL/TP fijos basados en ROI
    # Con 10x leverage: 5% ROI = 0.5% movimiento, 10% ROI = 1.0% movimiento
    sl_move_pct = CONFIG['SL_ROI'] / leverage  # 0.05 / 10 = 0.005 = 0.5%
    tp_move_pct = CONFIG['TP_ROI'] / leverage  # 0.10 / 10 = 0.01 = 1.0%
    
    if direction == 'LONG':
        sl_price = entry_price * (1 - sl_move_pct)
        tp_price = entry_price * (1 + tp_move_pct)
    else:  # SHORT
        sl_price = entry_price * (1 + sl_move_pct)
        tp_price = entry_price * (1 - tp_move_pct)
    
    # Comisiones
    entry_commission = exposure * CONFIG['COMMISSION_PCT']
    
    # Simular vela por vela
    for i in range(entry_idx + 1, len(df)):
        row = df.iloc[i]
        
        if direction == 'LONG':
            # Check SL primero
            if row['low'] <= sl_price:
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
                    'roi_pct': (pnl_net / margin) * 100,
                    'duration_hours': i - entry_idx
                }
            
            # Check TP
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
            # Check SL primero
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
            
            # Check TP
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
    
    # Trade aún abierto
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

# =============================================================================
# BACKTEST PRINCIPAL
# =============================================================================

def run_backtest_complete():
    """Ejecuta backtest completo con TODAS las características del bot real"""
    print("=" * 90)
    print("BACKTEST COMPLETO BOT REAL - NOVIEMBRE 2025")
    print("=" * 90)
    print("\n✅ CARACTERÍSTICAS IMPLEMENTADAS:")
    print("   • SL/TP ROI fijo: 5% y 10% (0.5% y 1.0% precio)")
    print("   • Cooldown 1 hora post-pérdida")
    print("   • Filtros multi-timeframe (15min, 5min, 1min)")
    print("   • Límite 2 trades/hora por símbolo")
    print("   • Filtro mercado lateral (<1.5% rango)")
    print("   • ADX mínimo 30\n")
    
    # Inicializar exchange
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
    print(f"🎯 SL: {CONFIG['SL_ROI']*100}% ROI ({CONFIG['SL_ROI']/CONFIG['LEVERAGE']*100}% precio) | TP: {CONFIG['TP_ROI']*100}% ROI ({CONFIG['TP_ROI']/CONFIG['LEVERAGE']*100}% precio)")
    print("\n" + "-" * 90)
    
    # Estado del backtest
    state = BacktestState()
    
    # Descargar datos 1h primero (para saber cuándo evaluar)
    print("\n📥 Descargando datos 1h...")
    symbol_data_1h = {}
    
    for symbol in SYMBOLS:
        print(f"   {symbol}...", end=" ")
        df = fetch_ohlcv_safe(exchange, symbol, '1h', since_ms, until_ms)
        if len(df) >= 100:
            df = calculate_indicators(df)
            symbol_data_1h[symbol] = df
            print(f"✅ {len(df)} velas")
        else:
            print(f"⚠️ Insuficiente")
    
    print(f"\n✅ {len(symbol_data_1h)} símbolos cargados para 1h")
    print("-" * 90)
    
    # Buscar todas las señales
    print("\n🔍 Buscando señales con TODOS los filtros...")
    all_signals = []
    filter_stats = {
        'total_checked': 0,
        'cooldown': 0,
        'hourly_limit': 0,
        'lateral': 0,
        '15min': 0,
        '5min': 0,
        '1min': 0,
        'other': 0,
        'passed': 0
    }
    
    for symbol in symbol_data_1h.keys():
        df_1h = symbol_data_1h[symbol]
        nov_start_idx = df_1h[df_1h['timestamp'] >= start_backtest].index[0]
        
        print(f"\n   {symbol}: Evaluando desde vela {nov_start_idx}...")
        
        for idx in range(nov_start_idx, len(df_1h)):
            row = df_1h.iloc[idx]
            current_time = row['timestamp']
            
            filter_stats['total_checked'] += 2  # LONG y SHORT
            
            # Descargar datos multi-timeframe para esta vela específica
            print(f"      Descargando multi-TF para {current_time.strftime('%m-%d %H:%M')}...", end="\r")
            dfs = fetch_multi_timeframe(exchange, symbol, current_time, since_ms, until_ms)
            dfs['1h'] = df_1h  # Incluir el 1h también
            
            # Verificar LONG
            long_ok, long_msg = check_long_entry(dfs, idx, state, symbol, current_time)
            if long_ok:
                all_signals.append({
                    'timestamp': current_time,
                    'symbol': symbol,
                    'direction': 'LONG',
                    'idx': idx,
                    'entry_price': row['close']
                })
                filter_stats['passed'] += 1
            else:
                # Trackear razón de rechazo
                if 'cooldown' in long_msg.lower():
                    filter_stats['cooldown'] += 1
                elif 'trades/hora' in long_msg.lower():
                    filter_stats['hourly_limit'] += 1
                elif 'lateral' in long_msg.lower():
                    filter_stats['lateral'] += 1
                elif '15min' in long_msg:
                    filter_stats['15min'] += 1
                elif '5min' in long_msg:
                    filter_stats['5min'] += 1
                elif '1min' in long_msg:
                    filter_stats['1min'] += 1
                else:
                    filter_stats['other'] += 1
            
            # Verificar SHORT
            short_ok, short_msg = check_short_entry(dfs, idx, state, symbol, current_time)
            if short_ok:
                all_signals.append({
                    'timestamp': current_time,
                    'symbol': symbol,
                    'direction': 'SHORT',
                    'idx': idx,
                    'entry_price': row['close']
                })
                filter_stats['passed'] += 1
            else:
                if 'cooldown' in short_msg.lower():
                    filter_stats['cooldown'] += 1
                elif 'trades/hora' in short_msg.lower():
                    filter_stats['hourly_limit'] += 1
                elif 'lateral' in short_msg.lower():
                    filter_stats['lateral'] += 1
                elif '15min' in short_msg:
                    filter_stats['15min'] += 1
                elif '5min' in short_msg:
                    filter_stats['5min'] += 1
                elif '1min' in short_msg:
                    filter_stats['1min'] += 1
                else:
                    filter_stats['other'] += 1
    
    all_signals.sort(key=lambda x: x['timestamp'])
    
    print("\n\n📊 ESTADÍSTICAS DE FILTROS:")
    print(f"   Total evaluaciones: {filter_stats['total_checked']}")
    print(f"   ✅ Señales aprobadas: {filter_stats['passed']}")
    print(f"   ❌ Rechazados por cooldown: {filter_stats['cooldown']}")
    print(f"   ❌ Rechazados por límite trades/hora: {filter_stats['hourly_limit']}")
    print(f"   ❌ Rechazados por mercado lateral: {filter_stats['lateral']}")
    print(f"   ❌ Rechazados por filtro 15min: {filter_stats['15min']}")
    print(f"   ❌ Rechazados por filtro 5min: {filter_stats['5min']}")
    print(f"   ❌ Rechazados por filtro 1min: {filter_stats['1min']}")
    print(f"   ❌ Rechazados por otras razones: {filter_stats['other']}")
    
    print("\n" + "-" * 90)
    
    # Ejecutar trades
    print(f"\n🚀 Ejecutando {len(all_signals)} trades...")
    all_trades = []
    
    for signal in all_signals:
        # Verificar slots disponibles
        state.open_positions = [t for t in state.open_positions if t > signal['timestamp']]
        if len(state.open_positions) >= CONFIG['MAX_OPEN_POSITIONS']:
            continue
        
        symbol = signal['symbol']
        df = symbol_data_1h[symbol]
        idx = signal['idx']
        direction = signal['direction']
        entry_price = signal['entry_price']
        
        # Simular trade con ROI fijo
        result = simulate_trade_roi_fixed(df, idx, direction, entry_price)
        
        # Calcular SL/TP
        sl_move_pct = CONFIG['SL_ROI'] / CONFIG['LEVERAGE']
        tp_move_pct = CONFIG['TP_ROI'] / CONFIG['LEVERAGE']
        
        if direction == 'LONG':
            sl_price = entry_price * (1 - sl_move_pct)
            tp_price = entry_price * (1 + tp_move_pct)
        else:
            sl_price = entry_price * (1 + sl_move_pct)
            tp_price = entry_price * (1 - tp_move_pct)
        
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
        state.open_positions.append(trade['exit_time'])
        state.add_trade(symbol, signal['timestamp'])
        
        # Si fue pérdida, activar cooldown
        if trade['pnl_net'] < 0:
            state.add_cooldown(symbol, trade['exit_time'])
        
        emoji = "🟢" if trade['pnl_net'] > 0 else "🔴"
        print(f"   {emoji} {trade['entry_time'].strftime('%m/%d %H:%M')} {symbol} {direction} -> {trade['exit_type']} ${trade['pnl_net']:.2f} (ROI: {trade['roi_pct']:.1f}%)")
    
    # Resultados finales
    print("\n" + "=" * 90)
    print("RESULTADOS FINALES - BACKTEST COMPLETO")
    print("=" * 90)
    
    if not all_trades:
        print("\n❌ No se ejecutaron trades")
        return
    
    df_trades = pd.DataFrame(all_trades)
    
    total_trades = len(df_trades)
    winners = len(df_trades[df_trades['pnl_net'] > 0])
    losers = len(df_trades[df_trades['pnl_net'] <= 0])
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = df_trades['pnl_net'].sum()
    total_commissions = df_trades['entry_commission'].sum() + df_trades['exit_commission'].sum()
    avg_pnl = df_trades['pnl_net'].mean()
    avg_roi = df_trades['roi_pct'].mean()
    
    max_win = df_trades['pnl_net'].max()
    max_loss = df_trades['pnl_net'].min()
    
    avg_win = df_trades[df_trades['pnl_net'] > 0]['pnl_net'].mean() if winners > 0 else 0
    avg_loss = df_trades[df_trades['pnl_net'] <= 0]['pnl_net'].mean() if losers > 0 else 0
    
    gross_profit = df_trades[df_trades['pnl_net'] > 0]['pnl_net'].sum()
    gross_loss = abs(df_trades[df_trades['pnl_net'] <= 0]['pnl_net'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    longs = df_trades[df_trades['direction'] == 'LONG']
    shorts = df_trades[df_trades['direction'] == 'SHORT']
    
    print(f"""
📊 MÉTRICAS GENERALES
─────────────────────────────────────────
   Total Trades:        {total_trades}
   Trades Ganadores:    {winners}
   Trades Perdedores:   {losers}
   Win Rate:            {win_rate:.1f}%
   
💰 RENTABILIDAD (ROI FIJO)
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
   LONG:  {len(longs)} trades | PnL: ${longs['pnl_net'].sum():.2f}
   SHORT: {len(shorts)} trades | PnL: ${shorts['pnl_net'].sum():.2f}

🏆 POR SÍMBOLO (Top 5)
─────────────────────────────────────────""")
    
    symbol_pnl = df_trades.groupby('symbol')['pnl_net'].agg(['sum', 'count']).sort_values('sum', ascending=False)
    for i, (symbol, row) in enumerate(symbol_pnl.head(5).iterrows()):
        print(f"   {i+1}. {symbol}: ${row['sum']:.2f} ({int(row['count'])} trades)")
    
    print(f"""
📅 POR TIPO DE SALIDA
─────────────────────────────────────────
   Take Profit (TP):    {len(df_trades[df_trades['exit_type'] == 'TP'])}
   Stop Loss (SL):      {len(df_trades[df_trades['exit_type'] == 'SL'])}
   Aún Abiertos:        {len(df_trades[df_trades['exit_type'] == 'OPEN'])}
""")
    
    # Guardar
    df_trades.to_csv('/Users/laurazapata/Desktop/DICIEMBRE/trades_completo_roi_nov_2025.csv', index=False)
    print("📁 Trades guardados en: trades_completo_roi_nov_2025.csv")
    
    print("\n" + "=" * 90)
    print("✅ BACKTEST COMPLETO FINALIZADO")
    print("=" * 90)

if __name__ == "__main__":
    run_backtest_complete()
