"""
BOT GANADORA - BINANCE FUTURES
==============================
Configuración del 1 de Diciembre 2025
Bot profesional para trading automático en Binance Futures.

CARACTERÍSTICAS:
- Máximo 3 posiciones simultáneas
- Selección de mejor señal cuando hay múltiples candidatos
- Validación exhaustiva de indicadores (sin NaN, sin datos falsos)
- SL/TP con MARK_PRICE y closePosition
- Logs verbosos y detallados
- Manejo robusto de errores

USO:
1. Configurar API keys abajo
2. Ejecutar: python bot_ganadora.py
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import os
import sys
from typing import Optional, Dict, Tuple, List

# =============================================================================
# 🔑 CONFIGURACIÓN DE API - EDITAR AQUÍ
# =============================================================================
API_KEY = "9cjKxO08wRWK95S9rEbG0E7U3ZEZvWYjpZOH2M8ZFYkrtx1tHU1Sc86AyeU6Z0ME"
API_SECRET = "wiy9JdBSKqCDja0pFXdvSxvJ2VsApzyQmCPDolLR9Dkk2M9LitkDWBh8Hh3fWfHR"

# =============================================================================
# ⚙️ CONFIGURACIÓN DEL BOT (según archivo de configuración)
# =============================================================================
CONFIG = {
    # Capital
    'MARGIN_USD': 100,              # Margen fijo por trade
    'LEVERAGE': 10,                 # Apalancamiento 10x
    'MAX_OPEN_POSITIONS': 3,        # Máximo 3 posiciones simultáneas
    'TIMEFRAME': '1h',              # Velas de 1 hora
    
    # Risk Management
    'SL_ATR_MULT': 1.5,             # Stop Loss = 1.5 × ATR
    'TP_ATR_MULT': 3.0,             # Take Profit = 3.0 × ATR
    
    # Indicadores
    'ADX_MIN': 28,                  # ADX mínimo para operar
    'RSI_LONG_MIN': 55,             # RSI mínimo para LONG
    'RSI_SHORT_MAX': 70,            # RSI máximo para SHORT
    'VOLUME_RATIO': 1.2,            # Volumen >= 1.2x promedio
    'EMA_EXTENSION_ATR_MULT': 3.0,  # Máx distancia a EMA20
    
    # Filtros de seguridad
    'ATR_MIN_PCT': 0.002,           # ATR mínimo 0.2%
    'ATR_MAX_PCT': 0.15,            # ATR máximo 15%
    'MAX_SPREAD_PCT': 0.001,        # Spread máximo 0.1%
    
    # Intervalos
    'CHECK_INTERVAL': 60,           # Revisar cada 60 segundos
    'OHLCV_LIMIT': 100,             # Velas a descargar
}

# Símbolos a operar
SYMBOLS = [
    'DOGE/USDT', 'OP/USDT', 'ATOM/USDT', 'FIL/USDT', 'ADA/USDT',
    'TRX/USDT', 'DOT/USDT', 'LINK/USDT', 'ARB/USDT', 'APT/USDT'
]

# =============================================================================
# 📝 CONFIGURACIÓN DE LOGGING VERBOSO
# =============================================================================
class VerboseFormatter(logging.Formatter):
    """Formatter personalizado con colores y formato extendido"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)

# Crear logger
logger = logging.getLogger('BotGanadora')
logger.setLevel(logging.DEBUG)

# Handler para archivo (sin colores)
file_handler = logging.FileHandler('bot_ganadora.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

# Handler para consola (con colores)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(VerboseFormatter(
    '%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# =============================================================================
# 📊 FUNCIONES DE INDICADORES CON VALIDACIÓN
# =============================================================================

def validate_series(series: pd.Series, name: str) -> bool:
    """Valida que una serie no tenga NaN en las últimas filas críticas"""
    if series.isna().iloc[-1]:
        logger.error(f"❌ VALIDACIÓN FALLIDA: {name} tiene NaN en última fila")
        return False
    if series.isna().iloc[-5:].any():
        logger.warning(f"⚠️ ADVERTENCIA: {name} tiene NaN en últimas 5 filas")
    return True

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calcula EMA (Exponential Moving Average)"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calcula SMA (Simple Moving Average)"""
    return series.rolling(window=period).mean()

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calcula ATR (Average True Range)
    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcula ADX (Average Directional Index) y DI+/DI-
    Retorna: (ADX, +DI, -DI)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calcular +DM y -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed TR y DMs
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    
    # +DI y -DI
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # DX y ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx, plus_di, minus_di

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcula RSI (Relative Strength Index)
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Usar EMA para suavizado (método Wilder)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcula MACD (Moving Average Convergence Divergence)
    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal)
    Histogram = MACD Line - Signal Line
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def detect_higher_low(df: pd.DataFrame, lookback: int = 50) -> Tuple[bool, str]:
    """
    Detecta Higher Low (HL) para señales LONG
    Busca en las últimas velas confirmadas
    
    IMPORTANTE: Trabaja sobre velas cerradas
    idx apunta a la última vela cerrada (len-2)
    El pivot se busca en idx-2 (3 velas atrás del actual)
    """
    idx = len(df) - 2  # Última vela CERRADA
    
    if idx < lookback + 3:
        return False, "Datos insuficientes para detectar HL"
    
    # Pivot candidato en idx-2 (confirmado por idx-1)
    pivot_idx = idx - 2
    pivot_low = df['low'].iloc[pivot_idx]
    
    # Verificar que sea un pivot low local
    left_higher = df['low'].iloc[pivot_idx - 1] > pivot_low
    right_higher = df['low'].iloc[pivot_idx + 1] > pivot_low
    
    if not left_higher:
        return False, f"Vela izquierda ({df['low'].iloc[pivot_idx - 1]:.4f}) no es mayor que pivot ({pivot_low:.4f})"
    if not right_higher:
        return False, f"Vela derecha ({df['low'].iloc[pivot_idx + 1]:.4f}) no es mayor que pivot ({pivot_low:.4f})"
    
    # Buscar pivot low anterior
    for i in range(pivot_idx - 3, max(pivot_idx - lookback, 1), -1):
        prev_low = df['low'].iloc[i]
        # Verificar si es un pivot low
        if df['low'].iloc[i-1] > prev_low and df['low'].iloc[i+1] > prev_low:
            if pivot_low > prev_low:
                return True, f"HL detectado: {prev_low:.4f} -> {pivot_low:.4f} (+{((pivot_low/prev_low)-1)*100:.2f}%)"
            else:
                return False, f"Es Lower Low: {prev_low:.4f} -> {pivot_low:.4f}"
    
    return False, "No se encontró pivot low anterior para comparar"

def detect_lower_high(df: pd.DataFrame, lookback: int = 50) -> Tuple[bool, str]:
    """
    Detecta Lower High (LH) para señales SHORT
    Busca en las últimas velas confirmadas
    
    IMPORTANTE: Trabaja sobre velas cerradas
    idx apunta a la última vela cerrada (len-2)
    El pivot se busca en idx-2 (3 velas atrás del actual)
    """
    idx = len(df) - 2  # Última vela CERRADA
    
    if idx < lookback + 3:
        return False, "Datos insuficientes para detectar LH"
    
    # Pivot candidato en idx-2 (confirmado por idx-1)
    pivot_idx = idx - 2
    pivot_high = df['high'].iloc[pivot_idx]
    
    # Verificar que sea un pivot high local
    left_lower = df['high'].iloc[pivot_idx - 1] < pivot_high
    right_lower = df['high'].iloc[pivot_idx + 1] < pivot_high
    
    if not left_lower:
        return False, f"Vela izquierda ({df['high'].iloc[pivot_idx - 1]:.4f}) no es menor que pivot ({pivot_high:.4f})"
    if not right_lower:
        return False, f"Vela derecha ({df['high'].iloc[pivot_idx + 1]:.4f}) no es menor que pivot ({pivot_high:.4f})"
    
    # Buscar pivot high anterior
    for i in range(pivot_idx - 3, max(pivot_idx - lookback, 1), -1):
        prev_high = df['high'].iloc[i]
        # Verificar si es un pivot high
        if df['high'].iloc[i-1] < prev_high and df['high'].iloc[i+1] < prev_high:
            if pivot_high < prev_high:
                return True, f"LH detectado: {prev_high:.4f} -> {pivot_high:.4f} ({((pivot_high/prev_high)-1)*100:.2f}%)"
            else:
                return False, f"Es Higher High: {prev_high:.4f} -> {pivot_high:.4f}"
    
    return False, "No se encontró pivot high anterior para comparar"

def calculate_all_indicators(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool, List[str]]:
    """
    Calcula todos los indicadores y valida que no haya NaN
    Retorna: (DataFrame con indicadores, éxito, lista de errores)
    """
    errors = []
    df = df.copy()
    
    try:
        # EMAs
        df['ema8'] = calculate_ema(df['close'], 8)
        df['ema20'] = calculate_ema(df['close'], 20)
        df['ema21'] = calculate_ema(df['close'], 21)
        df['ema50'] = calculate_ema(df['close'], 50)
        
        # ATR
        df['atr'] = calculate_atr(df, 14)
        
        # ADX
        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df, 14)
        
        # RSI
        df['rsi'] = calculate_rsi(df['close'], 14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
        
        # Volumen SMA
        df['vol_sma20'] = calculate_sma(df['volume'], 20)
        
        # Métricas derivadas
        df['atr_pct'] = df['atr'] / df['close']
        df['ema20_dist_atr'] = abs(df['close'] - df['ema20']) / df['atr']
        
        # Validar última fila
        # Validar la última vela CERRADA (iloc[-2])
        last_row = df.iloc[-2]
        indicators_to_check = ['ema8', 'ema20', 'ema21', 'ema50', 'atr', 'adx', 'rsi', 
                               'macd_hist', 'vol_sma20', 'atr_pct', 'ema20_dist_atr']
        
        for ind in indicators_to_check:
            if pd.isna(last_row[ind]):
                errors.append(f"{ind} es NaN")
            elif ind in ['atr', 'vol_sma20'] and last_row[ind] <= 0:
                errors.append(f"{ind} es <= 0: {last_row[ind]}")
        
        if errors:
            return df, False, errors
        
        return df, True, []
        
    except Exception as e:
        errors.append(f"Error calculando indicadores: {str(e)}")
        return df, False, errors

# =============================================================================
# 🎯 FUNCIONES DE SEÑALES CON SCORING
# =============================================================================

def analyze_signal(df: pd.DataFrame, spread_pct: float, symbol: str) -> Optional[Dict]:
    """
    Analiza un símbolo y retorna información de señal con score
    Score más alto = mejor señal
    
    IMPORTANTE: Usa iloc[-2] para evaluar la ÚLTIMA VELA CERRADA
    La vela en iloc[-1] puede estar aún en progreso
    """
    # Usar la penúltima vela (última CERRADA)
    # iloc[-1] = vela actual (puede estar abierta)
    # iloc[-2] = última vela cerrada (confirmada)
    idx = len(df) - 2  # Índice de la última vela cerrada
    row = df.iloc[-2]  # Última vela CERRADA
    
    candle_time = df['timestamp'].iloc[-2]
    logger.debug(f"   🕐 Evaluando vela CERRADA: {candle_time}")
    
    # Log de análisis
    logger.debug(f"{'─'*60}")
    logger.debug(f"📊 ANALIZANDO {symbol}")
    logger.debug(f"   Precio: ${row['close']:.4f}")
    logger.debug(f"   Spread: {spread_pct*100:.4f}%")
    
    # Verificar datos suficientes
    if idx < 60:
        logger.debug(f"   ❌ Datos insuficientes: {idx} velas (necesita 60)")
        return None
    
    # Verificar spread
    if spread_pct > CONFIG['MAX_SPREAD_PCT']:
        logger.debug(f"   ❌ Spread muy alto: {spread_pct*100:.4f}% > {CONFIG['MAX_SPREAD_PCT']*100}%")
        return None
    
    # Verificar filtro ATR
    atr_pct = row['atr_pct']
    if not (CONFIG['ATR_MIN_PCT'] <= atr_pct <= CONFIG['ATR_MAX_PCT']):
        logger.debug(f"   ❌ ATR% fuera de rango: {atr_pct*100:.4f}% (rango: {CONFIG['ATR_MIN_PCT']*100}%-{CONFIG['ATR_MAX_PCT']*100}%)")
        return None
    
    # Verificar ADX (común para LONG y SHORT)
    if row['adx'] < CONFIG['ADX_MIN']:
        logger.debug(f"   ❌ ADX bajo: {row['adx']:.2f} < {CONFIG['ADX_MIN']}")
        return None
    
    logger.debug(f"   ✓ Spread OK: {spread_pct*100:.4f}%")
    logger.debug(f"   ✓ ATR%: {atr_pct*100:.4f}%")
    logger.debug(f"   ✓ ADX: {row['adx']:.2f}")
    
    # =========================================================================
    # VERIFICAR SEÑAL LONG
    # =========================================================================
    long_checks = {
        'ema8_gt_ema21': row['ema8'] > row['ema21'],
        'close_gt_ema50': row['close'] > row['ema50'],
        'ema20_gt_ema50': row['ema20'] > row['ema50'],
        'rsi_gt_55': row['rsi'] > CONFIG['RSI_LONG_MIN'],
        'macd_hist_positive': row['macd_hist'] > 0,
        'volume_high': row['volume'] >= CONFIG['VOLUME_RATIO'] * row['vol_sma20'],
        'ema_extension_ok': row['ema20_dist_atr'] < CONFIG['EMA_EXTENSION_ATR_MULT'],
    }
    
    hl_detected, hl_reason = detect_higher_low(df)
    long_checks['higher_low'] = hl_detected
    
    long_passed = all(long_checks.values())
    
    logger.debug(f"   📈 LONG checks:")
    for check, passed in long_checks.items():
        status = "✓" if passed else "✗"
        logger.debug(f"      {status} {check}: {passed}")
    if not hl_detected:
        logger.debug(f"         └─ Razón HL: {hl_reason}")
    
    # =========================================================================
    # VERIFICAR SEÑAL SHORT
    # =========================================================================
    short_checks = {
        'ema8_lt_ema21': row['ema8'] < row['ema21'],
        'close_lt_ema50': row['close'] < row['ema50'],
        'ema20_lt_ema50': row['ema20'] < row['ema50'],
        'rsi_lt_70': row['rsi'] < CONFIG['RSI_SHORT_MAX'],
        'macd_hist_negative': row['macd_hist'] < 0,
        'volume_high': row['volume'] >= CONFIG['VOLUME_RATIO'] * row['vol_sma20'],
        'ema_extension_ok': row['ema20_dist_atr'] < CONFIG['EMA_EXTENSION_ATR_MULT'],
    }
    
    lh_detected, lh_reason = detect_lower_high(df)
    short_checks['lower_high'] = lh_detected
    
    short_passed = all(short_checks.values())
    
    logger.debug(f"   📉 SHORT checks:")
    for check, passed in short_checks.items():
        status = "✓" if passed else "✗"
        logger.debug(f"      {status} {check}: {passed}")
    if not lh_detected:
        logger.debug(f"         └─ Razón LH: {lh_reason}")
    
    # =========================================================================
    # CALCULAR SCORE (para elegir la mejor señal)
    # =========================================================================
    if long_passed:
        # Score basado en fuerza de la señal
        score = 0
        score += (row['adx'] - CONFIG['ADX_MIN']) * 2  # ADX más alto = mejor
        score += (row['rsi'] - 50) * 0.5  # RSI más alto = momentum más fuerte
        score += abs(row['macd_hist']) * 100  # MACD histogram más grande = mejor
        score += (row['volume'] / row['vol_sma20'] - 1) * 10  # Más volumen = mejor
        
        logger.info(f"   🟢 SEÑAL LONG VÁLIDA | Score: {score:.2f}")
        
        return {
            'symbol': symbol,
            'side': 'buy',
            'direction': 'LONG',
            'price': row['close'],
            'atr': row['atr'],
            'score': score,
            'indicators': {
                'adx': row['adx'],
                'rsi': row['rsi'],
                'macd_hist': row['macd_hist'],
                'volume_ratio': row['volume'] / row['vol_sma20'],
                'atr_pct': atr_pct,
            },
            'pivot_info': hl_reason
        }
    
    if short_passed:
        # Score basado en fuerza de la señal
        score = 0
        score += (row['adx'] - CONFIG['ADX_MIN']) * 2
        score += (50 - row['rsi']) * 0.5  # RSI más bajo = momentum más fuerte
        score += abs(row['macd_hist']) * 100
        score += (row['volume'] / row['vol_sma20'] - 1) * 10
        
        logger.info(f"   🔴 SEÑAL SHORT VÁLIDA | Score: {score:.2f}")
        
        return {
            'symbol': symbol,
            'side': 'sell',
            'direction': 'SHORT',
            'price': row['close'],
            'atr': row['atr'],
            'score': score,
            'indicators': {
                'adx': row['adx'],
                'rsi': row['rsi'],
                'macd_hist': row['macd_hist'],
                'volume_ratio': row['volume'] / row['vol_sma20'],
                'atr_pct': atr_pct,
            },
            'pivot_info': lh_reason
        }
    
    logger.debug(f"   ⚪ Sin señal válida")
    return None

# =============================================================================
# 💹 CLASE PRINCIPAL DEL BOT
# =============================================================================

class BotGanadora:
    def __init__(self):
        """Inicializa el bot con validaciones"""
        logger.info("=" * 70)
        logger.info("🚀 INICIANDO BOT GANADORA - BINANCE FUTURES")
        logger.info("=" * 70)
        
        # Validar API keys
        if not API_KEY or not API_SECRET or API_KEY == "" or API_SECRET == "":
            raise ValueError("❌ API_KEY y API_SECRET deben estar configurados")
        
        logger.info("🔑 Configurando conexión con Binance Futures...")
        
        self.exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        
        # Cargar mercados
        logger.info("📡 Cargando información de mercados...")
        self.exchange.load_markets()
        
        # Verificar conexión
        logger.info("🔄 Verificando conexión y balance...")
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = float(balance['USDT']['free'])
            usdt_total = float(balance['USDT']['total'])
            logger.info(f"💰 Balance USDT: ${usdt_balance:.2f} disponible / ${usdt_total:.2f} total")
        except Exception as e:
            raise ConnectionError(f"❌ No se pudo conectar a Binance: {e}")
        
        # Configurar modo de posición (One-Way Mode)
        logger.info("⚙️ Configurando modo de posición...")
        try:
            self.exchange.set_position_mode(hedged=False)
            logger.info("   ✓ Modo One-Way activado")
        except Exception as e:
            logger.warning(f"   ⚠️ No se pudo cambiar modo de posición (puede que ya esté configurado): {e}")
        
        # Mostrar configuración
        logger.info("")
        logger.info("📋 CONFIGURACIÓN ACTIVA:")
        logger.info(f"   💵 Margen por trade: ${CONFIG['MARGIN_USD']}")
        logger.info(f"   📊 Apalancamiento: {CONFIG['LEVERAGE']}x")
        logger.info(f"   📈 Exposición por trade: ${CONFIG['MARGIN_USD'] * CONFIG['LEVERAGE']}")
        logger.info(f"   🎯 Máx posiciones simultáneas: {CONFIG['MAX_OPEN_POSITIONS']}")
        logger.info(f"   ⏱️ Timeframe: {CONFIG['TIMEFRAME']}")
        logger.info(f"   🛑 Stop Loss: {CONFIG['SL_ATR_MULT']}x ATR")
        logger.info(f"   🎯 Take Profit: {CONFIG['TP_ATR_MULT']}x ATR")
        logger.info(f"   📊 Símbolos: {', '.join(SYMBOLS)}")
        logger.info("=" * 70)
    
    def get_open_positions(self) -> Dict[str, Dict]:
        """Obtiene posiciones abiertas con detalles"""
        logger.debug("📊 Consultando posiciones abiertas...")
        
        try:
            positions = self.exchange.fetch_positions()
            open_positions = {}
            
            for pos in positions:
                contracts = abs(float(pos['contracts']))
                if contracts > 0:
                    symbol = pos['symbol']
                    side = pos['side']  # 'long' o 'short'
                    entry_price = float(pos['entryPrice'])
                    unrealized_pnl = float(pos['unrealizedPnl'])
                    
                    open_positions[symbol] = {
                        'side': side,
                        'contracts': contracts,
                        'entry_price': entry_price,
                        'unrealized_pnl': unrealized_pnl,
                        'raw': pos
                    }
                    
                    logger.debug(f"   📌 {symbol}: {side.upper()} | Entrada: ${entry_price:.4f} | PnL: ${unrealized_pnl:.2f}")
            
            if open_positions:
                logger.info(f"📊 Posiciones abiertas: {len(open_positions)}/{CONFIG['MAX_OPEN_POSITIONS']}")
            else:
                logger.debug("   Sin posiciones abiertas")
            
            return open_positions
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo posiciones: {e}")
            return {}
    
    def get_balance(self) -> float:
        """Obtiene balance USDT disponible"""
        try:
            balance = self.exchange.fetch_balance()
            available = float(balance['USDT']['free'])
            logger.debug(f"💰 Balance disponible: ${available:.2f}")
            return available
        except Exception as e:
            logger.error(f"❌ Error obteniendo balance: {e}")
            return 0
    
    def set_leverage(self, symbol: str) -> bool:
        """Configura el apalancamiento para un símbolo"""
        try:
            market_symbol = symbol.replace('/', '')
            self.exchange.set_leverage(CONFIG['LEVERAGE'], market_symbol)
            logger.debug(f"   ✓ Leverage configurado: {CONFIG['LEVERAGE']}x para {symbol}")
            return True
        except Exception as e:
            # Puede fallar si ya está configurado con el mismo valor
            logger.debug(f"   ⚠️ Leverage ya configurado o error: {e}")
            return True
    
    def get_spread(self, symbol: str) -> Tuple[float, float, float]:
        """Obtiene bid, ask y spread actual"""
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit=5)
            bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
            ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
            spread_pct = (ask - bid) / bid if bid > 0 else 1
            return bid, ask, spread_pct
        except Exception as e:
            logger.error(f"❌ Error obteniendo orderbook {symbol}: {e}")
            return 0, 0, 1
    
    def fetch_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Descarga datos OHLCV de Binance
        
        IMPORTANTE: Binance retorna la vela actual (en progreso) como última.
        Por eso evaluamos iloc[-2] que es la última vela CERRADA.
        """
        try:
            logger.debug(f"📥 Descargando {CONFIG['OHLCV_LIMIT']} velas {CONFIG['TIMEFRAME']} de {symbol}...")
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, 
                CONFIG['TIMEFRAME'], 
                limit=CONFIG['OHLCV_LIMIT']
            )
            
            if not ohlcv or len(ohlcv) < 62:  # Necesitamos 60 + 2 (vela actual + margen)
                logger.warning(f"   ⚠️ Datos insuficientes para {symbol}: {len(ohlcv) if ohlcv else 0} velas")
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Verificar datos válidos
            if df['close'].isna().any() or df['volume'].isna().any():
                logger.warning(f"   ⚠️ Datos con NaN detectados en {symbol}")
                return None
            
            # Mostrar info de velas
            current_candle = df['timestamp'].iloc[-1]
            closed_candle = df['timestamp'].iloc[-2]
            logger.debug(f"   ✓ {len(df)} velas descargadas")
            logger.debug(f"      Vela EN PROGRESO: {current_candle} (se ignora)")
            logger.debug(f"      Última CERRADA:   {closed_candle} (se evalúa)")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error descargando OHLCV {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calcula el tamaño de la posición según la precisión del mercado"""
        try:
            market = self.exchange.market(symbol)
            notional = CONFIG['MARGIN_USD'] * CONFIG['LEVERAGE']
            quantity = notional / price
            
            # Redondear según precisión del mercado
            quantity = float(self.exchange.amount_to_precision(symbol, quantity))
            
            # Verificar mínimos
            min_qty = market['limits']['amount']['min'] if market['limits']['amount']['min'] else 0
            if quantity < min_qty:
                logger.warning(f"   ⚠️ Cantidad {quantity} menor que mínimo {min_qty}")
                return 0
            
            logger.debug(f"   📐 Tamaño posición: {quantity} {symbol.split('/')[0]}")
            return quantity
            
        except Exception as e:
            logger.error(f"❌ Error calculando tamaño de posición: {e}")
            return 0
    
    def open_position(self, signal: Dict) -> bool:
        """
        Abre una posición con SL y TP usando MARK_PRICE y closePosition
        """
        symbol = signal['symbol']
        side = signal['side']
        direction = signal['direction']
        price = signal['price']
        atr = signal['atr']
        
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"🎯 ABRIENDO POSICIÓN {direction}")
        logger.info("=" * 70)
        logger.info(f"   Símbolo: {symbol}")
        logger.info(f"   Dirección: {direction} ({side})")
        logger.info(f"   Precio actual: ${price:.4f}")
        logger.info(f"   ATR: ${atr:.6f} ({(atr/price)*100:.4f}%)")
        logger.info(f"   Score: {signal['score']:.2f}")
        logger.info(f"   Pivot: {signal['pivot_info']}")
        logger.info("")
        logger.info(f"   📊 Indicadores:")
        for key, value in signal['indicators'].items():
            logger.info(f"      {key}: {value:.4f}")
        logger.info("")
        
        try:
            # Calcular SL y TP
            if side == 'buy':  # LONG
                sl_price = price - (CONFIG['SL_ATR_MULT'] * atr)
                tp_price = price + (CONFIG['TP_ATR_MULT'] * atr)
                reduce_side = 'sell'
            else:  # SHORT
                sl_price = price + (CONFIG['SL_ATR_MULT'] * atr)
                tp_price = price - (CONFIG['TP_ATR_MULT'] * atr)
                reduce_side = 'buy'
            
            # Aplicar precisión de precio
            sl_price = float(self.exchange.price_to_precision(symbol, sl_price))
            tp_price = float(self.exchange.price_to_precision(symbol, tp_price))
            
            # Calcular riesgo/beneficio esperado
            risk_pct = abs(price - sl_price) / price * 100
            reward_pct = abs(tp_price - price) / price * 100
            exposure = CONFIG['MARGIN_USD'] * CONFIG['LEVERAGE']
            expected_loss = exposure * (risk_pct / 100)
            expected_profit = exposure * (reward_pct / 100)
            
            logger.info(f"   🎯 NIVELES DE SALIDA:")
            logger.info(f"      Stop Loss: ${sl_price:.4f} (riesgo: -{risk_pct:.2f}% = -${expected_loss:.2f})")
            logger.info(f"      Take Profit: ${tp_price:.4f} (beneficio: +{reward_pct:.2f}% = +${expected_profit:.2f})")
            logger.info(f"      Ratio R:B = 1:{reward_pct/risk_pct:.1f}")
            logger.info("")
            
            # Configurar apalancamiento
            logger.info("   ⚙️ Configurando apalancamiento...")
            self.set_leverage(symbol)
            
            # Calcular cantidad
            quantity = self.calculate_position_size(symbol, price)
            if quantity <= 0:
                logger.error("   ❌ No se pudo calcular cantidad válida")
                return False
            
            market_symbol = symbol.replace('/', '')
            
            # ================================================================
            # ORDEN DE ENTRADA (MARKET)
            # ================================================================
            logger.info(f"   📤 Ejecutando orden de ENTRADA...")
            logger.info(f"      Tipo: MARKET")
            logger.info(f"      Side: {side.upper()}")
            logger.info(f"      Cantidad: {quantity}")
            
            entry_order = self.exchange.create_order(
                symbol=symbol,
                type='MARKET',
                side=side,
                amount=quantity
            )
            
            entry_id = entry_order.get('id', 'N/A')
            fill_price = float(entry_order.get('average', price))
            
            logger.info(f"      ✅ ORDEN EJECUTADA")
            logger.info(f"      Order ID: {entry_id}")
            logger.info(f"      Precio de llenado: ${fill_price:.4f}")
            logger.info("")
            
            # Recalcular SL/TP con precio de llenado real
            if side == 'buy':
                sl_price = fill_price - (CONFIG['SL_ATR_MULT'] * atr)
                tp_price = fill_price + (CONFIG['TP_ATR_MULT'] * atr)
            else:
                sl_price = fill_price + (CONFIG['SL_ATR_MULT'] * atr)
                tp_price = fill_price - (CONFIG['TP_ATR_MULT'] * atr)
            
            sl_price = float(self.exchange.price_to_precision(symbol, sl_price))
            tp_price = float(self.exchange.price_to_precision(symbol, tp_price))
            
            # ================================================================
            # ORDEN TAKE PROFIT (TAKE_PROFIT_MARKET con closePosition)
            # ================================================================
            logger.info(f"   🎯 Colocando TAKE PROFIT...")
            logger.info(f"      Tipo: TAKE_PROFIT_MARKET")
            logger.info(f"      Stop Price: ${tp_price:.4f}")
            logger.info(f"      Working Type: MARK_PRICE")
            logger.info(f"      Close Position: True")
            
            try:
                tp_order = self.exchange.create_order(
                    symbol=symbol,
                    type='TAKE_PROFIT_MARKET',
                    side=reduce_side,
                    amount=None,  # closePosition no necesita amount
                    params={
                        'workingType': 'MARK_PRICE',
                        'closePosition': True,
                        'stopPrice': tp_price
                    }
                )
                
                tp_id = tp_order.get('id', 'N/A')
                logger.info(f"      ✅ TP COLOCADO - Order ID: {tp_id}")
                
            except Exception as e:
                logger.error(f"      ❌ Error colocando TP: {e}")
                logger.warning("      ⚠️ Intentando método alternativo para TP...")
                
                # Método alternativo
                try:
                    tp_order = self.exchange.create_order(
                        symbol=symbol,
                        type='TAKE_PROFIT_MARKET',
                        side=reduce_side,
                        amount=quantity,
                        params={
                            'workingType': 'MARK_PRICE',
                            'reduceOnly': True,
                            'stopPrice': tp_price
                        }
                    )
                    logger.info(f"      ✅ TP COLOCADO (alternativo) - Order ID: {tp_order.get('id', 'N/A')}")
                except Exception as e2:
                    logger.error(f"      ❌ También falló método alternativo: {e2}")
            
            logger.info("")
            
            # ================================================================
            # ORDEN STOP LOSS (STOP_MARKET con closePosition)
            # ================================================================
            logger.info(f"   🛑 Colocando STOP LOSS...")
            logger.info(f"      Tipo: STOP_MARKET")
            logger.info(f"      Stop Price: ${sl_price:.4f}")
            logger.info(f"      Working Type: MARK_PRICE")
            logger.info(f"      Close Position: True")
            logger.info(f"      Price Protect: True")
            
            try:
                sl_order = self.exchange.create_order(
                    symbol=symbol,
                    type='STOP_MARKET',
                    side=reduce_side,
                    amount=None,  # closePosition no necesita amount
                    params={
                        'workingType': 'MARK_PRICE',
                        'closePosition': True,
                        'stopPrice': sl_price,
                        'priceProtect': True
                    }
                )
                
                sl_id = sl_order.get('id', 'N/A')
                logger.info(f"      ✅ SL COLOCADO - Order ID: {sl_id}")
                
            except Exception as e:
                logger.error(f"      ❌ Error colocando SL: {e}")
                logger.warning("      ⚠️ Intentando método alternativo para SL...")
                
                # Método alternativo
                try:
                    sl_order = self.exchange.create_order(
                        symbol=symbol,
                        type='STOP_MARKET',
                        side=reduce_side,
                        amount=quantity,
                        params={
                            'workingType': 'MARK_PRICE',
                            'reduceOnly': True,
                            'stopPrice': sl_price,
                            'priceProtect': True
                        }
                    )
                    logger.info(f"      ✅ SL COLOCADO (alternativo) - Order ID: {sl_order.get('id', 'N/A')}")
                except Exception as e2:
                    logger.error(f"      ❌ También falló método alternativo: {e2}")
                    logger.critical("      🚨 POSICIÓN ABIERTA SIN STOP LOSS - REVISAR MANUALMENTE")
            
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"✅ POSICIÓN {direction} ABIERTA EXITOSAMENTE")
            logger.info("=" * 70)
            logger.info("")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error abriendo posición: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def scan_for_signals(self, open_positions: Dict) -> List[Dict]:
        """
        Escanea todos los símbolos y retorna lista de señales válidas ordenadas por score
        """
        logger.info("")
        logger.info(f"🔍 ESCANEANDO {len(SYMBOLS)} SÍMBOLOS...")
        logger.info("")
        
        valid_signals = []
        
        for symbol in SYMBOLS:
            # Saltar si ya tenemos posición en este símbolo
            symbol_normalized = symbol.replace('/', '')
            if any(symbol_normalized in pos for pos in open_positions.keys()):
                logger.debug(f"   ⏭️ {symbol}: Ya tiene posición abierta")
                continue
            
            # Descargar datos
            df = self.fetch_ohlcv(symbol)
            if df is None:
                continue
            
            # Calcular indicadores
            df, indicators_ok, errors = calculate_all_indicators(df)
            if not indicators_ok:
                logger.warning(f"   ⚠️ {symbol}: Indicadores inválidos: {', '.join(errors)}")
                continue
            
            # Obtener spread
            bid, ask, spread_pct = self.get_spread(symbol)
            if bid == 0 or ask == 0:
                logger.warning(f"   ⚠️ {symbol}: No se pudo obtener orderbook")
                continue
            
            # Analizar señal
            signal = analyze_signal(df, spread_pct, symbol)
            
            if signal:
                valid_signals.append(signal)
            
            # Rate limit
            time.sleep(0.2)
        
        # Ordenar por score (mayor primero)
        valid_signals.sort(key=lambda x: x['score'], reverse=True)
        
        if valid_signals:
            logger.info("")
            logger.info(f"📊 SEÑALES ENCONTRADAS: {len(valid_signals)}")
            for i, sig in enumerate(valid_signals):
                logger.info(f"   {i+1}. {sig['symbol']} {sig['direction']} | Score: {sig['score']:.2f}")
        else:
            logger.info("   ⚪ No se encontraron señales válidas")
        
        return valid_signals
    
    def wait_for_candle_close(self) -> None:
        """
        Espera al cierre de la próxima vela de 1 hora.
        
        Esto asegura que evaluamos datos CONFIRMADOS (velas cerradas)
        y no datos en progreso que pueden cambiar.
        
        La estrategia requiere evaluar velas cerradas para evitar
        señales falsas que podrían generarse con velas aún abiertas.
        """
        now = datetime.now()
        
        # Calcular segundos hasta el próximo cierre de hora
        minutes_to_next_hour = 60 - now.minute
        seconds_to_next_hour = 60 - now.second
        
        # Total de segundos hasta el próximo cierre de vela 1h
        seconds_until_close = (minutes_to_next_hour - 1) * 60 + seconds_to_next_hour
        
        # Si estamos muy cerca del cierre (< 30s), esperar a la siguiente hora
        if seconds_until_close < 30:
            seconds_until_close += 3600
        
        # Agregar 10 segundos extra para asegurar que la vela esté completamente cerrada
        # y que Binance haya procesado los datos
        seconds_until_close += 10
        
        next_candle_time = now + timedelta(seconds=seconds_until_close)
        
        logger.info(f"")
        logger.info(f"⏰ SINCRONIZACIÓN DE VELAS")
        logger.info(f"   Hora actual: {now.strftime('%H:%M:%S')}")
        logger.info(f"   Próxima vela cerrada disponible: {next_candle_time.strftime('%H:%M:%S')}")
        logger.info(f"   Esperando: {seconds_until_close // 60}m {seconds_until_close % 60}s")
        
        # Esperar en chunks de 60 segundos para mostrar progreso
        remaining = seconds_until_close
        
        while remaining > 0:
            wait_time = min(60, remaining)
            time.sleep(wait_time)
            remaining -= wait_time
            
            if remaining > 0 and remaining % 300 < 60:  # Cada 5 minutos
                logger.info(f"   ⏳ {remaining // 60}m restantes para próxima vela cerrada...")
        
        logger.info(f"   ✓ Vela de 1h cerrada. Procediendo con análisis...")
    
    def run(self):
        """
        Loop principal del bot con sincronización de velas cerradas.
        
        El bot espera al cierre de cada vela de 1h para evaluar señales,
        asegurando que todos los indicadores se calculen sobre datos
        confirmados (velas cerradas) y no sobre datos en progreso.
        """
        logger.info("")
        logger.info("🟢 BOT INICIADO - Comenzando operaciones...")
        logger.info(f"   Timeframe: 1h (velas cerradas)")
        logger.info(f"   El bot se sincronizará con el cierre de cada vela horaria")
        logger.info("")
        
        cycle = 0
        
        # Sincronizar con el próximo cierre de vela antes del primer ciclo
        logger.info("🕐 Sincronización inicial con cierre de vela...")
        self.wait_for_candle_close()
        
        while True:
            try:
                cycle += 1
                logger.info("")
                logger.info("═" * 70)
                logger.info(f"🔄 CICLO #{cycle} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"   ✓ Evaluando vela {(datetime.now() - timedelta(hours=1)).strftime('%H:00')} - {datetime.now().strftime('%H:00')} (CERRADA)")
                logger.info("═" * 70)
                
                # Obtener posiciones abiertas
                open_positions = self.get_open_positions()
                num_positions = len(open_positions)
                
                # Mostrar estado de posiciones
                if open_positions:
                    logger.info("")
                    logger.info("📌 POSICIONES ACTUALES:")
                    total_pnl = 0
                    for symbol, pos in open_positions.items():
                        total_pnl += pos['unrealized_pnl']
                        emoji = "🟢" if pos['unrealized_pnl'] >= 0 else "🔴"
                        logger.info(f"   {emoji} {symbol}: {pos['side'].upper()} @ ${pos['entry_price']:.4f} | PnL: ${pos['unrealized_pnl']:.2f}")
                    logger.info(f"   📊 PnL Total No Realizado: ${total_pnl:.2f}")
                
                # Verificar si podemos abrir más posiciones
                slots_available = CONFIG['MAX_OPEN_POSITIONS'] - num_positions
                
                if slots_available <= 0:
                    logger.info("")
                    logger.info(f"⏳ Máximo de posiciones alcanzado ({num_positions}/{CONFIG['MAX_OPEN_POSITIONS']})")
                    logger.info(f"   Esperando próximo cierre de vela para verificar...")
                    self.wait_for_candle_close()
                    continue
                
                logger.info("")
                logger.info(f"🎰 Slots disponibles: {slots_available}")
                
                # Verificar balance
                balance = self.get_balance()
                required_margin = CONFIG['MARGIN_USD'] * slots_available
                
                if balance < CONFIG['MARGIN_USD']:
                    logger.warning(f"⚠️ Balance insuficiente: ${balance:.2f} < ${CONFIG['MARGIN_USD']}")
                    self.wait_for_candle_close()
                    continue
                
                # Escanear señales
                signals = self.scan_for_signals(open_positions)
                
                # Abrir posiciones (hasta llenar slots disponibles)
                positions_opened = 0
                
                for signal in signals:
                    if positions_opened >= slots_available:
                        logger.info(f"   ✋ Slots llenos, guardando señales restantes para próximo ciclo")
                        break
                    
                    # Verificar balance para cada posición
                    current_balance = self.get_balance()
                    if current_balance < CONFIG['MARGIN_USD']:
                        logger.warning(f"   ⚠️ Balance insuficiente para más posiciones: ${current_balance:.2f}")
                        break
                    
                    # Abrir posición
                    success = self.open_position(signal)
                    
                    if success:
                        positions_opened += 1
                        time.sleep(1)  # Pequeña pausa entre órdenes
                
                if positions_opened > 0:
                    logger.info(f"")
                    logger.info(f"📈 Posiciones abiertas este ciclo: {positions_opened}")
                
                # Esperar próximo cierre de vela 1h para siguiente análisis
                self.wait_for_candle_close()
                
            except KeyboardInterrupt:
                logger.info("")
                logger.info("🔴 Bot detenido por el usuario (Ctrl+C)")
                logger.info("   Posiciones abiertas permanecen activas en Binance")
                break
                
            except Exception as e:
                logger.error(f"❌ Error en ciclo principal: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.info("   Reintentando en 30 segundos...")
                time.sleep(30)

# =============================================================================
# 🚀 INICIO
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║                     🏆 BOT GANADORA v2.1 🏆                           ║
    ║                    Binance Futures Trading                            ║
    ║                                                                       ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║  CONFIGURACIÓN:                                                       ║
    ║  • Máximo 3 posiciones simultáneas                                    ║
    ║  • Selección automática de mejor señal (por score)                    ║
    ║  • SL/TP con MARK_PRICE y closePosition                              ║
    ║  • Validación exhaustiva de indicadores                               ║
    ║  • ✓ EVALUACIÓN SOLO CON VELAS CERRADAS (1h)                         ║
    ║  • ✓ Sincronización automática con cierre de vela                    ║
    ║                                                                       ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║  ⚠️  ADVERTENCIAS:                                                    ║
    ║  • Este bot opera con DINERO REAL                                     ║
    ║  • Los resultados pasados no garantizan ganancias futuras            ║
    ║  • Asegúrate de entender los riesgos del trading con apalancamiento  ║
    ║  • Monitorea el bot regularmente                                      ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        bot = BotGanadora()
        bot.run()
        
    except ValueError as e:
        print(f"\n❌ Error de configuración: {e}")
        print("\n📝 Verifica que API_KEY y API_SECRET estén configurados correctamente")
        
    except ConnectionError as e:
        print(f"\n❌ Error de conexión: {e}")
        print("\n📝 Verifica tu conexión a internet y las API keys")
        
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
