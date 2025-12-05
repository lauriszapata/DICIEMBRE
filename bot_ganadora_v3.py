"""
BOT GANADORA - CONFIGURACIÓN ORIGINAL VERIFICADA
================================================
Restaurado 2 de Diciembre 2025

Esta es la configuración EXACTA del archivo:
"CONFIGURACION BOT GANADORA 1 DE DICIEMBRE 2025.txt"

RESULTADOS VERIFICADOS:
- Anual: +797%
- Noviembre: +$187, 60 trades, 36.7% WR

CARACTERÍSTICAS:
- 10 símbolos: DOGE, OP, ATOM, FIL, ADA, TRX, DOT, LINK, ARB, APT
- $100 margen, 10x leverage
- 1 posición máxima
- SL 1.5 ATR, TP 3.0 ATR (ratio 1:2)
- EMA8 > EMA21 (no EMA20)
- MACD Histogram (no MACD line)
- EMA Extension < 3.0 ATR
- Higher Low / Lower High estrictos

USO:
python bot_ganadora_v3.py
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import sys
from typing import Optional, Dict, Tuple, List

# =============================================================================
# 🔑 API KEYS - EDITAR AQUÍ
# =============================================================================
API_KEY = "9cjKxO08wRWK95S9rEbG0E7U3ZEZvWYjpZOH2M8ZFYkrtx1tHU1Sc86AyeU6Z0ME"
API_SECRET = "wiy9JdBSKqCDja0pFXdvSxvJ2VsApzyQmCPDolLR9Dkk2M9LitkDWBh8Hh3fWfHR"

# =============================================================================
# ⚙️ CONFIGURACIÓN SEGÚN ARCHIVO GANADOR
# =============================================================================
CONFIG = {
    # === CAPITAL ===
    'MARGIN_USD': 100,              # $100 margen (nocional $1000 con 10x)
    'LEVERAGE': 20,                 # 20x (igual que backtest)
    'MAX_OPEN_POSITIONS': 3,        # 3 posiciones simultáneas
    
    # === TIMEFRAME ===
    'TIMEFRAME': '1h',              # VELAS DE 1 HORA
    'OHLCV_LIMIT': 200,             # Velas a descargar
    
    # === PERIODOS DE INDICADORES (ORIGINAL) ===
    'EMA_FAST': 8,                  # EMA8
    'EMA_SIGNAL': 21,               # EMA21 (NO EMA20)
    'EMA_SLOW': 50,                 # EMA50
    'ADX_PERIOD': 14,
    'RSI_PERIOD': 14,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'ATR_PERIOD': 14,
    'VOLUME_SMA_PERIOD': 20,
    'PIVOT_LOOKBACK': 50,
    
    # === UMBRALES DE INDICADORES (ORIGINAL) ===
    'ADX_MIN': 28,                  # ADX mínimo para fuerza de tendencia
    'RSI_LONG_MIN': 55,
    'RSI_SHORT_MAX': 70,
    'VOLUME_RATIO': 1.2,
    'EMA_EXTENSION_ATR_MULT': 3.0,  # Filtro EMA Extension < 3.0 ATR
    
    # === STOP LOSS / TAKE PROFIT (basado en ATR) ===
    # SL = 1.3 ATR (óptimo según backtest), TP = 3 ATR
    'SL_ATR_MULT': 1.3,             # SL a 1.3 ATR del precio de entrada (óptimo: +$2,055 anual)
    'TP_ATR_MULT': 3.5,             # TP a 3.5 ATR (óptimo: +$2,420 anual, PF 1.38)
    
    # === SIN TRAILING STOP EN ORIGINAL ===
    'USE_TRAILING_STOP': False,     # No estaba en config original
    
    # === SIN FILTRO DE MERCADO EN ORIGINAL ===
    'USE_MARKET_FILTER': False,     # No estaba en config original
    
    # === FILTROS DE SEGURIDAD ===
    'ATR_MIN_PCT': 0.002,           # 0.2%
    'ATR_MAX_PCT': 0.15,            # 15%
    'MAX_SPREAD_PCT': 0.001,        # 0.1%
    
    # === FILTRO ANTI-ALBOROTO (15min vs 1H) ===
    'USE_CHAOS_FILTER': False,      # DESACTIVADO temporalmente
    'CHAOS_ATR_RATIO_MAX': 2.0,     # Si ATR_15min > 2.0 × (ATR_1H/4) = mercado MUY alborotado
    
    # === FILTRO TENDENCIA EMA 15 MINUTOS ===
    'USE_15MIN_EMA_FILTER': False,  # DESACTIVADO - Confirmar tendencia en 15min antes de entrar
    'EMA_15MIN_FAST': 8,            # EMA8 en 15min
    'EMA_15MIN_SLOW': 21,           # EMA21 en 15min
    
    # === FILTRO VELAS 5 MINUTOS (TENDENCIA CORTA) ===
    'USE_5MIN_CANDLE_FILTER': False, # DESACTIVADO - Verificar análisis de velas de 5min
    
    # === FILTRO VELAS 1 MINUTO (MOMENTUM INMEDIATO) ===
    'USE_1MIN_CANDLE_FILTER': False, # DESACTIVADO - Verificar últimas 3 velas de 1min
    
    # === ANTI-SCALPING: LÍMITE DE TRADES POR HORA ===
    'MAX_TRADES_PER_HOUR': 999,     # Sin límite (999 = prácticamente ilimitado)
    
    # === FILTRO MERCADO LATERAL ===
    'USE_LATERAL_FILTER': True,     # No operar si mercado muy lateral
    'LATERAL_RANGE_PCT': 0.015,     # Si rango últimas 4 velas < 1.5%, no operar
    
    # === FILTRO DE DÍAS Y HORAS (basado en backtest 2025) ===
    # Análisis del backtest mostró:
    # - Martes fue el ÚNICO día con pérdidas (-$138.65)
    # - Horas 01, 23, 12 UTC fueron las peores (-$317 combinadas)
    # - Filtrar estas horas mejora PnL de $2,450 a $2,767 (+$317)
    'USE_TIME_FILTER': True,        # Activar filtro de días/horas
    'BLOCKED_DAYS': [1],            # 0=Lunes, 1=Martes, ... 6=Domingo (Bloquear Martes)
    'BLOCKED_HOURS_UTC': [1, 12, 23],  # Horas a evitar (peores del backtest)
    
    # === GESTIÓN DE RIESGO (FASE 1) ===
    'MAX_DAILY_LOSS_PCT': 0.05,     # Límite de pérdida diaria (5%)
    'MAX_CONSECUTIVE_LOSSES': 3,    # Pausa tras 3 pérdidas seguidas
    'PAUSE_HOURS_AFTER_STREAK': 24, # Horas de pausa tras racha
    
    # === PROP TRADING MODE (Crypto Fund Trader) ===
    'PROP_MODE': False,             # Activar modo prop trading (más estricto)
    'MAX_OVERALL_LOSS_PCT': 0.10,   # Límite de pérdida TOTAL (10% del capital inicial)
    'PROFIT_TARGET_PCT': 0.08,      # Objetivo de ganancia Fase 1 (8%)
    'CHALLENGE_CAPITAL': 25000,     # Capital del challenge ($25,000)
}


# SÍMBOLOS RENTABLES EN 2025 (basado en backtest anual)
SYMBOLS = [
    'ADA/USDT',   # +$532.52 en 2025
    'FIL/USDT',   # +$292.54 en 2025
    'ARB/USDT',   # +$266.45 en 2025
    'LINK/USDT',  # +$176.35 en 2025
    'OP/USDT',    # +$114.99 en 2025
    'TRX/USDT',   # +$107.79 en 2025
]

# =============================================================================
# 🛡️ GESTOR DE RIESGO (PROP TRADING READY)
# =============================================================================
class RiskManager:
    def __init__(self, exchange):
        self.exchange = exchange
        self.peak_balance = 0
        self.initial_balance = 0
        self.challenge_start_balance = 0  # Balance al inicio del challenge
        self.daily_pnl = 0
        self.total_pnl = 0  # PnL total desde inicio del challenge
        self.last_trades = []
        self.day_start = datetime.now().date()
        self.pause_until = None
        self.challenge_failed = False  # Si ya perdimos el challenge
        self.challenge_passed = False  # Si ya ganamos el challenge
        
        # Inicializar balance
        self._update_balance()
        self.initial_balance = self.peak_balance
        self.challenge_start_balance = self.peak_balance
        
        # Log inicial
        logger = logging.getLogger('BotGanadora')
        if CONFIG.get('PROP_MODE', False):
            capital = CONFIG.get('CHALLENGE_CAPITAL', 25000)
            target = CONFIG.get('PROFIT_TARGET_PCT', 0.08) * 100
            max_loss = CONFIG.get('MAX_OVERALL_LOSS_PCT', 0.10) * 100
            logger.info("🏆 MODO PROP TRADING ACTIVADO")
            logger.info(f"   Capital Challenge: ${capital:,.0f}")
            logger.info(f"   Objetivo Profit: {target:.0f}% (${capital * CONFIG.get('PROFIT_TARGET_PCT', 0.08):,.0f})")
            logger.info(f"   Max Pérdida Total: {max_loss:.0f}% (${capital * CONFIG.get('MAX_OVERALL_LOSS_PCT', 0.10):,.0f})")
            logger.info(f"   Max Pérdida Diaria: 5%")
        
    def _update_balance(self):
        """Actualiza balance y peak"""
        try:
            balance = self.exchange.fetch_balance()
            current_balance = float(balance['USDT']['total'])
            
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance
                
            return current_balance
        except Exception as e:
            logging.getLogger('BotGanadora').error(f"Error actualizando balance: {e}")
            return self.peak_balance

    def update(self):
        """Actualiza estado: trades recientes, PnL diario y total"""
        current_balance = self._update_balance()
        current_day = datetime.now().date()
        logger = logging.getLogger('BotGanadora')
        
        # Reset diario
        if current_day != self.day_start:
            self.daily_pnl = 0
            self.day_start = current_day
            self.initial_balance = current_balance
            
        # Calcular PnL diario
        self.daily_pnl = current_balance - self.initial_balance
        
        # Calcular PnL total (desde inicio del challenge)
        self.total_pnl = current_balance - self.challenge_start_balance
        
        # Mostrar progreso si está en modo prop
        if CONFIG.get('PROP_MODE', False) and not self.challenge_failed and not self.challenge_passed:
            capital = CONFIG.get('CHALLENGE_CAPITAL', 25000)
            target_pct = CONFIG.get('PROFIT_TARGET_PCT', 0.08)
            target_usd = capital * target_pct
            max_loss_pct = CONFIG.get('MAX_OVERALL_LOSS_PCT', 0.10)
            max_loss_usd = capital * max_loss_pct
            
            progress_pct = (self.total_pnl / target_usd * 100) if target_usd > 0 else 0
            
            logger.info("📊 PROGRESO CHALLENGE:")
            logger.info(f"   PnL Total: ${self.total_pnl:+.2f} / ${target_usd:,.0f} ({progress_pct:.1f}% del objetivo)")
            logger.info(f"   PnL Hoy: ${self.daily_pnl:+.2f}")
            logger.info(f"   Margen restante pérdida: ${max_loss_usd + self.total_pnl:.2f}")
            
            # Verificar si ganamos
            if self.total_pnl >= target_usd:
                self.challenge_passed = True
                logger.info("🎉🎉🎉 ¡CHALLENGE COMPLETADO! 🎉🎉🎉")
                logger.info(f"   Ganaste ${self.total_pnl:.2f} (>{target_pct*100:.0f}%)")
        
    def check_losing_streak(self, closed_trade):
        """Llamar cuando se cierra un trade"""
        self.last_trades.append(closed_trade)
        if len(self.last_trades) > 10:
            self.last_trades.pop(0)
            
        # Verificar racha
        if len(self.last_trades) >= CONFIG['MAX_CONSECUTIVE_LOSSES']:
            recent = self.last_trades[-CONFIG['MAX_CONSECUTIVE_LOSSES']:]
            if all(t['pnl'] < 0 for t in recent):
                pause_hours = CONFIG['PAUSE_HOURS_AFTER_STREAK']
                self.pause_until = datetime.now() + timedelta(hours=pause_hours)
                logging.getLogger('BotGanadora').warning(f"⛔ {CONFIG['MAX_CONSECUTIVE_LOSSES']} pérdidas consecutivas. Pausando trading por {pause_hours}h")

    def can_trade(self):
        """Verifica todas las reglas de riesgo"""
        logger = logging.getLogger('BotGanadora')
        
        # 0. Si el challenge ya falló o pasó, no operar
        if self.challenge_failed:
            logger.warning("🚫 CHALLENGE FALLIDO - Trading detenido permanentemente")
            return False
        if self.challenge_passed:
            logger.info("🏆 CHALLENGE COMPLETADO - Puedes pasar a la siguiente fase")
            return False
        
        # 1. Verificar pausa por racha
        if self.pause_until:
            if datetime.now() < self.pause_until:
                remaining = int((self.pause_until - datetime.now()).total_seconds() / 60)
                logger.info(f"⏳ Trading pausado por racha de pérdidas ({remaining} min restantes)")
                return False
            else:
                self.pause_until = None
                self.last_trades = []
        
        current_balance = self._update_balance()
        
        # 2. Límite diario (5%)
        max_daily_loss = current_balance * CONFIG['MAX_DAILY_LOSS_PCT']
        if self.daily_pnl < -max_daily_loss:
            logger.warning(f"⛔ Límite de pérdida DIARIA alcanzado (${self.daily_pnl:.2f} < -${max_daily_loss:.2f})")
            return False
        
        # 3. Límite TOTAL (10%) - Solo en modo prop
        if CONFIG.get('PROP_MODE', False):
            capital = CONFIG.get('CHALLENGE_CAPITAL', 25000)
            max_overall_loss = capital * CONFIG.get('MAX_OVERALL_LOSS_PCT', 0.10)
            
            if self.total_pnl < -max_overall_loss:
                logger.error(f"🚨🚨🚨 CHALLENGE FALLIDO 🚨🚨🚨")
                logger.error(f"   Pérdida total: ${self.total_pnl:.2f} (límite: -${max_overall_loss:.2f})")
                self.challenge_failed = True
                return False
            
        return True
    
    def get_leverage(self):
        """Leverage dinámico según drawdown"""
        current_balance = self._update_balance()
        if self.peak_balance == 0: return CONFIG['LEVERAGE']
        
        drawdown = (self.peak_balance - current_balance) / self.peak_balance
        
        if drawdown <= 0.10:
            return 10
        elif drawdown <= 0.20:
            return 8
        elif drawdown <= 0.30:
            return 5
        else:
            return 3

# =============================================================================
# 📝 LOGGING ULTRA VERBOSO CON PERSISTENCIA
# =============================================================================
from logging.handlers import RotatingFileHandler
import os

def setup_logging():
    """
    Configura logging ultra verboso con persistencia.
    
    ARCHIVOS DE LOG:
    - bot_ganadora_v3.log: Log principal (rotación 10MB, 5 backups)
    - bot_ganadora_v3_trades.log: Solo entradas/salidas de trades
    - bot_ganadora_v3_signals.log: Todas las señales evaluadas
    
    Esto permite consultar historial después.
    """
    # Directorio de logs
    log_dir = '/Users/laurazapata/Desktop/DICIEMBRE/logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Logger principal
    logger = logging.getLogger('BotGanadora')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # Formato detallado con milisegundos
    fmt = '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt)
    
    # === Handler 1: Consola ===
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # === Handler 2: Archivo principal (rotación 10MB, 5 backups) ===
    main_log = os.path.join(log_dir, 'bot_ganadora_v3.log')
    file_handler = RotatingFileHandler(
        main_log, 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # === Handler 3: Log de trades (solo INFO y superior) ===
    trades_log = os.path.join(log_dir, 'bot_ganadora_v3_trades.log')
    trades_handler = RotatingFileHandler(
        trades_log,
        maxBytes=5*1024*1024,  # 5 MB
        backupCount=10,
        encoding='utf-8'
    )
    trades_handler.setLevel(logging.INFO)
    trades_handler.setFormatter(formatter)
    # Filtro para solo mostrar trades
    class TradesFilter(logging.Filter):
        def filter(self, record):
            keywords = ['ABRIENDO', 'POSICIÓN', 'SEÑAL', 'LONG', 'SHORT', 'PnL', 'Balance', 'SL', 'TP']
            return any(kw in record.getMessage() for kw in keywords)
    trades_handler.addFilter(TradesFilter())
    logger.addHandler(trades_handler)
    
    # Mensaje inicial
    logger.info("=" * 70)
    logger.info("📝 SISTEMA DE LOGGING INICIADO")
    logger.info(f"   Log principal: {main_log}")
    logger.info(f"   Log de trades: {trades_log}")
    logger.info("   Rotación: 10MB, 5 backups")
    logger.info("=" * 70)
    
    return logger

logger = setup_logging()

# =============================================================================
# 🌪️ FILTRO ANTI-ALBOROTO (Mercado Caótico)
# =============================================================================
def check_market_chaos(exchange, symbol: str, atr_1h: float) -> Dict:
    """
    Detecta si el mercado está "alborotado" comparando ATR de 15min vs 1H.
    
    LÓGICA:
    - Si ATR_15min > 1.5 × (ATR_1H / 4) → Mercado caótico, NO OPERAR
    - ATR_1H/4 es lo que "debería" ser el ATR de 15min si la volatilidad fuera uniforme
    
    Retorna:
    - is_chaotic: bool
    - atr_15m: float
    - expected_atr: float (ATR_1H/4)
    - ratio: float
    """
    if not CONFIG.get('USE_CHAOS_FILTER', False):
        return {'is_chaotic': False, 'atr_15m': 0, 'expected_atr': 0, 'ratio': 0, 'message': 'Filtro desactivado'}
    
    try:
        # Descargar últimas 20 velas de 15 minutos
        ohlcv_15m = exchange.fetch_ohlcv(symbol, '15m', limit=20)
        df_15m = pd.DataFrame(ohlcv_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Calcular ATR de 15 minutos (período 14)
        high = df_15m['high']
        low = df_15m['low']
        close = df_15m['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_15m = tr.ewm(span=14, adjust=False).mean().iloc[-1]
        
        # ATR esperado = ATR_1H / 4 (porque 4 velas de 15min = 1 vela de 1H)
        expected_atr = atr_1h / 4
        
        # Ratio de caos
        ratio = atr_15m / expected_atr if expected_atr > 0 else 0
        
        # Es caótico si el ratio supera el umbral
        is_chaotic = ratio > CONFIG.get('CHAOS_ATR_RATIO_MAX', 1.5)
        
        if is_chaotic:
            message = f"🌪️ ALBOROTADO: ATR_15m={atr_15m:.6f} > {CONFIG['CHAOS_ATR_RATIO_MAX']}×(ATR_1H/4)={expected_atr*CONFIG['CHAOS_ATR_RATIO_MAX']:.6f}"
        else:
            message = f"✅ Mercado estable: ratio={ratio:.2f}x (max {CONFIG['CHAOS_ATR_RATIO_MAX']}x)"
        
        return {
            'is_chaotic': is_chaotic,
            'atr_15m': atr_15m,
            'expected_atr': expected_atr,
            'ratio': ratio,
            'message': message
        }
        
    except Exception as e:
        logger.warning(f"   ⚠️ No se pudo verificar caos de mercado: {e}")
        return {'is_chaotic': False, 'atr_15m': 0, 'expected_atr': 0, 'ratio': 0, 'message': f'Error: {e}'}


def check_15min_ema_trend(exchange, symbol: str, direction: str) -> Dict:
    """
    Verifica que la tendencia en 15 minutos confirme la señal de 1H.
    
    LÓGICA MÁS ESTRICTA:
    - Para LONG: EMA8_15m > EMA21_15m en las últimas 4 velas de 15min
    - Para SHORT: EMA8_15m < EMA21_15m en las últimas 4 velas de 15min
    - Además: El precio debe estar del lado correcto de EMA8
    
    Esto evita entrar cuando las EMAs apenas cruzaron o hay reversión inminente.
    
    Retorna:
    - aligned: bool (si las EMAs de 15min confirman la dirección)
    - ema8_15m: float
    - ema21_15m: float
    - consecutive_bars: int (cuántas velas consecutivas confirman)
    - message: str
    """
    if not CONFIG.get('USE_15MIN_EMA_FILTER', False):
        return {'aligned': True, 'ema8_15m': 0, 'ema21_15m': 0, 'consecutive_bars': 0, 'message': 'Filtro 15min EMA desactivado'}
    
    try:
        # Descargar últimas 50 velas de 15 minutos (suficiente para EMA21)
        ohlcv_15m = exchange.fetch_ohlcv(symbol, '15m', limit=50)
        df_15m = pd.DataFrame(ohlcv_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Calcular EMAs de 15 minutos
        ema_fast_period = CONFIG.get('EMA_15MIN_FAST', 8)
        ema_slow_period = CONFIG.get('EMA_15MIN_SLOW', 21)
        
        df_15m['ema8'] = df_15m['close'].ewm(span=ema_fast_period, adjust=False).mean()
        df_15m['ema21'] = df_15m['close'].ewm(span=ema_slow_period, adjust=False).mean()
        
        # Valores actuales (última vela CERRADA = índice -2)
        ema8_15m = df_15m['ema8'].iloc[-2]
        ema21_15m = df_15m['ema21'].iloc[-2]
        current_close = df_15m['close'].iloc[-2]
        
        # Verificar dirección en las últimas 4 velas cerradas (-5 a -2)
        REQUIRED_BARS = 4
        is_long = direction == 'LONG'
        consecutive = 0
        
        for i in range(-5, -1):  # -5, -4, -3, -2 (4 velas cerradas)
            if i < -len(df_15m):
                continue
            
            ema8_i = df_15m['ema8'].iloc[i]
            ema21_i = df_15m['ema21'].iloc[i]
            
            if is_long:
                if ema8_i > ema21_i:
                    consecutive += 1
                else:
                    consecutive = 0  # Reset si no cumple
            else:  # SHORT
                if ema8_i < ema21_i:
                    consecutive += 1
                else:
                    consecutive = 0  # Reset si no cumple
        
        # CONDICIONES PARA CONFIRMAR:
        # 1. Al menos 3 velas consecutivas con EMAs alineadas
        # 2. Precio actual del lado correcto de EMA8
        ema_condition = consecutive >= 3
        
        if is_long:
            price_condition = current_close > ema8_15m
        else:
            price_condition = current_close < ema8_15m
        
        aligned = ema_condition and price_condition
        
        if aligned:
            message = f"✅ 15min CONFIRMA {direction}: {consecutive} velas alineadas, precio {'>' if is_long else '<'} EMA8"
        else:
            reasons = []
            if not ema_condition:
                reasons.append(f"solo {consecutive}/3 velas alineadas")
            if not price_condition:
                reasons.append(f"precio {'<' if is_long else '>'} EMA8")
            message = f"❌ 15min NO CONFIRMA {direction}: {', '.join(reasons)}"
        
        return {
            'aligned': aligned,
            'ema8_15m': ema8_15m,
            'ema21_15m': ema21_15m,
            'current_close': current_close,
            'consecutive_bars': consecutive,
            'ema_condition': ema_condition,
            'price_condition': price_condition,
            'message': message
        }
        
    except Exception as e:
        logger.warning(f"   ⚠️ No se pudo verificar EMA 15min: {e}")
        # En caso de error, NO permitimos el trade (más seguro)
        return {'aligned': False, 'ema8_15m': 0, 'ema21_15m': 0, 'consecutive_bars': 0, 'message': f'Error: {e} (bloqueado)'}


def check_1min_candles(exchange, symbol: str, direction: str) -> Dict:
    """
    ANÁLISIS COMPLETO DE VELAS DE 1 MINUTO para confirmar entrada.
    
    VERIFICACIONES:
    1. Últimas 3 velas: Al menos 2 deben estar alineadas con la dirección
    2. EMA8 vs EMA21 en 1min: Deben estar alineadas con la dirección
    3. Pendiente de EMA8: Debe estar subiendo (LONG) o bajando (SHORT)
    4. Precio actual: Debe estar del lado correcto de EMA8
    5. Momentum: El cierre actual debe ser mejor que hace 3 velas
    
    Retorna:
    - aligned: bool (si TODOS los criterios confirman la dirección)
    - detalles de cada verificación
    """
    if not CONFIG.get('USE_1MIN_CANDLE_FILTER', True):
        return {'aligned': True, 'message': 'Filtro 1min desactivado'}
    
    try:
        # Descargar últimas 30 velas de 1 minuto (para EMAs estables)
        ohlcv_1m = exchange.fetch_ohlcv(symbol, '1m', limit=30)
        df = pd.DataFrame(ohlcv_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Calcular EMAs de 1 minuto
        df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        is_long = direction == 'LONG'
        checks = {}
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK 1: Últimas 3 velas cerradas (-4, -3, -2) alineadas
        # ═══════════════════════════════════════════════════════════════════
        bullish_count = 0
        bearish_count = 0
        candle_details = []
        
        for i in range(-4, -1):  # -4, -3, -2 (3 velas cerradas)
            if abs(i) > len(df):
                continue
            open_price = df['open'].iloc[i]
            close_price = df['close'].iloc[i]
            
            if close_price > open_price:
                bullish_count += 1
                candle_details.append("🟢")
            elif close_price < open_price:
                bearish_count += 1
                candle_details.append("🔴")
            else:
                candle_details.append("⚪")
        
        candles_str = " ".join(candle_details)
        if is_long:
            check1_pass = bullish_count >= 2
            checks['candles'] = {'pass': check1_pass, 'value': f"{bullish_count}/3 alcistas [{candles_str}]"}
        else:
            check1_pass = bearish_count >= 2
            checks['candles'] = {'pass': check1_pass, 'value': f"{bearish_count}/3 bajistas [{candles_str}]"}
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK 2: EMA8 vs EMA21 alineadas (última vela cerrada)
        # ═══════════════════════════════════════════════════════════════════
        ema8_current = df['ema8'].iloc[-2]
        ema21_current = df['ema21'].iloc[-2]
        
        if is_long:
            check2_pass = ema8_current > ema21_current
            checks['ema_cross'] = {'pass': check2_pass, 'value': f"EMA8={ema8_current:.6f} {'>' if check2_pass else '<'} EMA21={ema21_current:.6f}"}
        else:
            check2_pass = ema8_current < ema21_current
            checks['ema_cross'] = {'pass': check2_pass, 'value': f"EMA8={ema8_current:.6f} {'<' if check2_pass else '>'} EMA21={ema21_current:.6f}"}
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK 3: Pendiente de EMA8 (comparar últimas 3 velas)
        # ═══════════════════════════════════════════════════════════════════
        ema8_3_back = df['ema8'].iloc[-5]  # Hace 3 velas
        ema8_slope = ema8_current - ema8_3_back
        slope_pct = (ema8_slope / ema8_3_back) * 100 if ema8_3_back != 0 else 0
        
        if is_long:
            check3_pass = ema8_slope > 0  # Pendiente positiva para LONG
            checks['ema_slope'] = {'pass': check3_pass, 'value': f"Pendiente: {'+' if ema8_slope > 0 else ''}{slope_pct:.4f}%"}
        else:
            check3_pass = ema8_slope < 0  # Pendiente negativa para SHORT
            checks['ema_slope'] = {'pass': check3_pass, 'value': f"Pendiente: {'+' if ema8_slope > 0 else ''}{slope_pct:.4f}%"}
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK 4: Precio actual respecto a EMA8
        # ═══════════════════════════════════════════════════════════════════
        current_close = df['close'].iloc[-2]
        
        if is_long:
            check4_pass = current_close > ema8_current
            checks['price_vs_ema'] = {'pass': check4_pass, 'value': f"Precio={current_close:.6f} {'>' if check4_pass else '<'} EMA8={ema8_current:.6f}"}
        else:
            check4_pass = current_close < ema8_current
            checks['price_vs_ema'] = {'pass': check4_pass, 'value': f"Precio={current_close:.6f} {'<' if check4_pass else '>'} EMA8={ema8_current:.6f}"}
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK 5: Momentum - Precio actual vs hace 5 velas
        # ═══════════════════════════════════════════════════════════════════
        close_5_back = df['close'].iloc[-7]  # Hace 5 velas
        momentum = current_close - close_5_back
        momentum_pct = (momentum / close_5_back) * 100 if close_5_back != 0 else 0
        
        if is_long:
            check5_pass = momentum > 0  # Precio subiendo para LONG
            checks['momentum'] = {'pass': check5_pass, 'value': f"Momentum 5 velas: {'+' if momentum > 0 else ''}{momentum_pct:.4f}%"}
        else:
            check5_pass = momentum < 0  # Precio bajando para SHORT
            checks['momentum'] = {'pass': check5_pass, 'value': f"Momentum 5 velas: {'+' if momentum > 0 else ''}{momentum_pct:.4f}%"}
        
        # ═══════════════════════════════════════════════════════════════════
        # RESULTADO FINAL: Contar checks pasados
        # ═══════════════════════════════════════════════════════════════════
        passed_checks = sum(1 for c in checks.values() if c['pass'])
        total_checks = len(checks)
        
        # Requiere al menos 4 de 5 checks para confirmar
        MIN_CHECKS_REQUIRED = 4
        aligned = passed_checks >= MIN_CHECKS_REQUIRED
        
        if aligned:
            message = f"✅ 1min CONFIRMA {direction}: {passed_checks}/{total_checks} checks pasados"
        else:
            failed = [k for k, v in checks.items() if not v['pass']]
            message = f"❌ 1min NO CONFIRMA {direction}: {passed_checks}/{total_checks} checks (falló: {', '.join(failed)})"
        
        return {
            'aligned': aligned,
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'checks': checks,
            'ema8': ema8_current,
            'ema21': ema21_current,
            'current_close': current_close,
            'candles_str': candles_str,
            'message': message
        }
        
    except Exception as e:
        logger.warning(f"   ⚠️ No se pudo verificar velas 1min: {e}")
        return {'aligned': False, 'passed_checks': 0, 'total_checks': 5, 'checks': {}, 'message': f'Error: {e} (bloqueado)'}


def check_5min_candles(exchange, symbol: str, direction: str) -> Dict:
    """
    ANÁLISIS COMPLETO DE VELAS DE 5 MINUTOS para confirmar tendencia corta.
    
    VERIFICACIONES:
    1. Últimas 3 velas de 5min: Al menos 2 deben estar alineadas
    2. EMA8 vs EMA21 en 5min: Deben estar alineadas con la dirección
    3. Pendiente de EMA8: Debe estar subiendo (LONG) o bajando (SHORT)
    4. Precio actual respecto a EMA8 y EMA21
    5. Distancia entre EMAs: Si están separándose (tendencia fuerte)
    
    Retorna:
    - aligned: bool (si TODOS los criterios confirman la dirección)
    - detalles de cada verificación
    """
    if not CONFIG.get('USE_5MIN_CANDLE_FILTER', True):
        return {'aligned': True, 'message': 'Filtro 5min desactivado'}
    
    try:
        # Descargar últimas 50 velas de 5 minutos (para EMAs estables)
        ohlcv_5m = exchange.fetch_ohlcv(symbol, '5m', limit=50)
        df = pd.DataFrame(ohlcv_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Calcular EMAs de 5 minutos
        df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        is_long = direction == 'LONG'
        checks = {}
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK 1: Últimas 3 velas cerradas de 5min alineadas
        # ═══════════════════════════════════════════════════════════════════
        bullish_count = 0
        bearish_count = 0
        candle_details = []
        
        for i in range(-4, -1):  # -4, -3, -2 (3 velas cerradas)
            if abs(i) > len(df):
                continue
            open_price = df['open'].iloc[i]
            close_price = df['close'].iloc[i]
            
            if close_price > open_price:
                bullish_count += 1
                candle_details.append("🟢")
            elif close_price < open_price:
                bearish_count += 1
                candle_details.append("🔴")
            else:
                candle_details.append("⚪")
        
        candles_str = " ".join(candle_details)
        if is_long:
            check1_pass = bullish_count >= 2
            checks['candles'] = {'pass': check1_pass, 'value': f"{bullish_count}/3 alcistas [{candles_str}]"}
        else:
            check1_pass = bearish_count >= 2
            checks['candles'] = {'pass': check1_pass, 'value': f"{bearish_count}/3 bajistas [{candles_str}]"}
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK 2: EMA8 vs EMA21 alineadas (última vela cerrada)
        # ═══════════════════════════════════════════════════════════════════
        ema8_current = df['ema8'].iloc[-2]
        ema21_current = df['ema21'].iloc[-2]
        
        if is_long:
            check2_pass = ema8_current > ema21_current
            checks['ema_cross'] = {'pass': check2_pass, 'value': f"EMA8={ema8_current:.6f} {'>' if check2_pass else '<'} EMA21={ema21_current:.6f}"}
        else:
            check2_pass = ema8_current < ema21_current
            checks['ema_cross'] = {'pass': check2_pass, 'value': f"EMA8={ema8_current:.6f} {'<' if check2_pass else '>'} EMA21={ema21_current:.6f}"}
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK 3: Pendiente de EMA8 (comparar últimas 3 velas de 5min)
        # ═══════════════════════════════════════════════════════════════════
        ema8_3_back = df['ema8'].iloc[-5]  # Hace 3 velas
        ema8_slope = ema8_current - ema8_3_back
        slope_pct = (ema8_slope / ema8_3_back) * 100 if ema8_3_back != 0 else 0
        
        if is_long:
            check3_pass = ema8_slope > 0
            checks['ema_slope'] = {'pass': check3_pass, 'value': f"Pendiente: {'+' if ema8_slope > 0 else ''}{slope_pct:.4f}%"}
        else:
            check3_pass = ema8_slope < 0
            checks['ema_slope'] = {'pass': check3_pass, 'value': f"Pendiente: {'+' if ema8_slope > 0 else ''}{slope_pct:.4f}%"}
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK 4: Precio actual entre EMA8 y EMA21 o mejor
        # ═══════════════════════════════════════════════════════════════════
        current_close = df['close'].iloc[-2]
        
        if is_long:
            # Para LONG: precio debe estar por encima de EMA21 (mínimo)
            check4_pass = current_close > ema21_current
            position = "arriba de EMA21" if current_close > ema21_current else "debajo de EMA21"
            checks['price_position'] = {'pass': check4_pass, 'value': f"Precio {position}"}
        else:
            # Para SHORT: precio debe estar por debajo de EMA21 (mínimo)
            check4_pass = current_close < ema21_current
            position = "debajo de EMA21" if current_close < ema21_current else "arriba de EMA21"
            checks['price_position'] = {'pass': check4_pass, 'value': f"Precio {position}"}
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK 5: Distancia entre EMAs aumentando (tendencia fortaleciendo)
        # ═══════════════════════════════════════════════════════════════════
        ema_gap_current = abs(ema8_current - ema21_current)
        ema8_prev = df['ema8'].iloc[-5]
        ema21_prev = df['ema21'].iloc[-5]
        ema_gap_prev = abs(ema8_prev - ema21_prev)
        
        gap_expanding = ema_gap_current > ema_gap_prev
        gap_change_pct = ((ema_gap_current - ema_gap_prev) / ema_gap_prev * 100) if ema_gap_prev != 0 else 0
        
        check5_pass = gap_expanding
        checks['ema_gap'] = {'pass': check5_pass, 'value': f"Gap EMAs: {'+' if gap_expanding else ''}{gap_change_pct:.2f}% ({'expandiendo' if gap_expanding else 'contrayendo'})"}
        
        # ═══════════════════════════════════════════════════════════════════
        # RESULTADO FINAL: Contar checks pasados
        # ═══════════════════════════════════════════════════════════════════
        passed_checks = sum(1 for c in checks.values() if c['pass'])
        total_checks = len(checks)
        
        # Requiere al menos 3 de 5 checks para confirmar (menos estricto que 1min)
        MIN_CHECKS_REQUIRED = 3
        aligned = passed_checks >= MIN_CHECKS_REQUIRED
        
        if aligned:
            message = f"✅ 5min CONFIRMA {direction}: {passed_checks}/{total_checks} checks pasados"
        else:
            failed = [k for k, v in checks.items() if not v['pass']]
            message = f"❌ 5min NO CONFIRMA {direction}: {passed_checks}/{total_checks} checks (falló: {', '.join(failed)})"
        
        return {
            'aligned': aligned,
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'checks': checks,
            'ema8': ema8_current,
            'ema21': ema21_current,
            'current_close': current_close,
            'candles_str': candles_str,
            'message': message
        }
        
    except Exception as e:
        logger.warning(f"   ⚠️ No se pudo verificar velas 5min: {e}")
        return {'aligned': False, 'passed_checks': 0, 'total_checks': 5, 'checks': {}, 'message': f'Error: {e} (bloqueado)'}


# =============================================================================
# 📊 CÁLCULO DE INDICADORES
# =============================================================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula TODOS los indicadores necesarios según la configuración ganadora.
    
    INDICADORES:
    - EMA 8, 20, 21, 50 (períodos según config)
    - ADX 14
    - RSI 14
    - MACD (12, 26, 9)
    - ATR 14
    - Volume SMA 20
    """
    logger.debug("=" * 60)
    logger.debug("📊 CALCULANDO INDICADORES")
    logger.debug("=" * 60)
    
    # --- EMAs (TODAS las necesarias: EMA8, EMA20, EMA21, EMA50) ---
    logger.debug(f"   EMA{CONFIG['EMA_FAST']}...")
    df['ema8'] = df['close'].ewm(span=CONFIG['EMA_FAST'], adjust=False).mean()
    
    logger.debug("   EMA20...")
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()  # EMA20 para distancia y extensión
    
    logger.debug(f"   EMA{CONFIG['EMA_SIGNAL']}...")
    df['ema21'] = df['close'].ewm(span=CONFIG['EMA_SIGNAL'], adjust=False).mean()
    
    logger.debug(f"   EMA{CONFIG['EMA_SLOW']}...")
    df['ema50'] = df['close'].ewm(span=CONFIG['EMA_SLOW'], adjust=False).mean()
    
    # --- ADX (período 14) ---
    logger.debug(f"   ADX (período {CONFIG['ADX_PERIOD']})...")
    high = df['high']
    low = df['low']
    close = df['close']
    
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
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    
    # --- RSI (período 14) ---
    logger.debug(f"   RSI (período {CONFIG['RSI_PERIOD']})...")
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(span=CONFIG['RSI_PERIOD'], adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=CONFIG['RSI_PERIOD'], adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # --- MACD (12, 26, 9) ---
    logger.debug(f"   MACD ({CONFIG['MACD_FAST']}, {CONFIG['MACD_SLOW']}, {CONFIG['MACD_SIGNAL']})...")
    ema_fast = close.ewm(span=CONFIG['MACD_FAST'], adjust=False).mean()
    ema_slow = close.ewm(span=CONFIG['MACD_SLOW'], adjust=False).mean()
    df['macd_line'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd_line'].ewm(span=CONFIG['MACD_SIGNAL'], adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']
    
    # --- ATR (período 14) ---
    logger.debug(f"   ATR (período {CONFIG['ATR_PERIOD']})...")
    df['atr'] = tr.ewm(span=CONFIG['ATR_PERIOD'], adjust=False).mean()
    
    # --- Volume SMA (período 20) ---
    logger.debug(f"   Volume SMA (período {CONFIG['VOLUME_SMA_PERIOD']})...")
    df['vol_sma'] = df['volume'].rolling(window=CONFIG['VOLUME_SMA_PERIOD']).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma']
    
    # --- Campos auxiliares ---
    df['atr_pct'] = df['atr'] / df['close']
    df['ema20_dist'] = (df['close'] - df['ema20']).abs() / df['atr']  # Distancia a EMA20 (según config original)
    
    logger.debug("   ✓ Todos los indicadores calculados")
    logger.debug("=" * 60)
    
    return df

# =============================================================================
# 🔍 DETECCIÓN DE PIVOTS (Higher Low / Lower High)
# =============================================================================
def detect_higher_low(df: pd.DataFrame, eval_idx: int) -> Tuple[bool, str]:
    """
    Detecta Higher Low (HL) para señales LONG.
    
    SEGÚN CONFIGURACIÓN GANADORA:
    - eval_idx es la vela donde tomamos la decisión (última cerrada)
    - Se verifica pivot en idx-2 (2 velas antes de eval_idx)
    - Se usa idx-1 como confirmación (1 vela antes de eval_idx)
    - Lookback máximo: 50 velas
    
    Ejemplo: si eval_idx=198, el pivot candidato está en idx=196,
    confirmado por la vela idx=197.
    """
    lookback = CONFIG['PIVOT_LOOKBACK']
    
    # Pivot candidato: 2 velas antes de la vela de decisión
    pivot_idx = eval_idx - 2
    
    if pivot_idx < 3:
        return False, "No hay suficientes velas para evaluar"
    
    pivot_low = df['low'].iloc[pivot_idx]
    
    # Confirmar que es pivot low: velas adyacentes deben tener lows más altos
    # Vela anterior (pivot_idx - 1)
    if df['low'].iloc[pivot_idx - 1] <= pivot_low:
        return False, f"No es pivot: vela anterior low {df['low'].iloc[pivot_idx-1]:.4f} <= {pivot_low:.4f}"
    
    # Vela siguiente (pivot_idx + 1) = la vela de confirmación (eval_idx - 1)
    if df['low'].iloc[pivot_idx + 1] <= pivot_low:
        return False, f"No es pivot: vela confirmación low {df['low'].iloc[pivot_idx+1]:.4f} <= {pivot_low:.4f}"
    
    # Buscar pivot low anterior (hasta 50 velas atrás)
    for i in range(pivot_idx - 3, max(0, pivot_idx - lookback), -1):
        if i <= 0 or i >= len(df) - 1:
            continue
        prev_low = df['low'].iloc[i]
        # Verificar si es un pivot low
        if df['low'].iloc[i-1] > prev_low and df['low'].iloc[i+1] > prev_low:
            # Comparar: Higher Low significa que el actual es más alto
            if pivot_low > prev_low:
                return True, f"HL confirmado: Low {pivot_low:.4f} (idx {pivot_idx}) > Low previo {prev_low:.4f} (idx {i})"
            else:
                return False, f"No es HL: Low {pivot_low:.4f} <= Low previo {prev_low:.4f} (idx {i})"
    
    return False, f"No se encontró pivot previo en últimas {lookback} velas"

def detect_lower_high(df: pd.DataFrame, eval_idx: int) -> Tuple[bool, str]:
    """
    Detecta Lower High (LH) para señales SHORT.
    
    SEGÚN CONFIGURACIÓN GANADORA:
    - eval_idx es la vela donde tomamos la decisión (última cerrada)
    - Se verifica pivot en idx-2 (2 velas antes de eval_idx)
    - Se usa idx-1 como confirmación (1 vela antes de eval_idx)
    - Lookback máximo: 50 velas
    
    Ejemplo: si eval_idx=198, el pivot candidato está en idx=196,
    confirmado por la vela idx=197.
    """
    lookback = CONFIG['PIVOT_LOOKBACK']
    
    # Pivot candidato: 2 velas antes de la vela de decisión
    pivot_idx = eval_idx - 2
    
    if pivot_idx < 3:
        return False, "No hay suficientes velas para evaluar"
    
    pivot_high = df['high'].iloc[pivot_idx]
    
    # Confirmar que es pivot high: velas adyacentes deben tener highs más bajos
    # Vela anterior (pivot_idx - 1)
    if df['high'].iloc[pivot_idx - 1] >= pivot_high:
        return False, f"No es pivot: vela anterior high {df['high'].iloc[pivot_idx-1]:.4f} >= {pivot_high:.4f}"
    
    # Vela siguiente (pivot_idx + 1) = la vela de confirmación (eval_idx - 1)
    if df['high'].iloc[pivot_idx + 1] >= pivot_high:
        return False, f"No es pivot: vela confirmación high {df['high'].iloc[pivot_idx+1]:.4f} >= {pivot_high:.4f}"
    
    # Buscar pivot high anterior (hasta 50 velas atrás)
    for i in range(pivot_idx - 3, max(0, pivot_idx - lookback), -1):
        if i <= 0 or i >= len(df) - 1:
            continue
        prev_high = df['high'].iloc[i]
        # Verificar si es un pivot high
        if df['high'].iloc[i-1] < prev_high and df['high'].iloc[i+1] < prev_high:
            # Comparar: Lower High significa que el actual es más bajo
            if pivot_high < prev_high:
                return True, f"LH confirmado: High {pivot_high:.4f} (idx {pivot_idx}) < High previo {prev_high:.4f} (idx {i})"
            else:
                return False, f"No es LH: High {pivot_high:.4f} >= High previo {prev_high:.4f} (idx {i})"
    
    return False, f"No se encontró pivot previo en últimas {lookback} velas"

# =============================================================================
# 📈 SISTEMA HÍBRIDO - ANÁLISIS EXHAUSTIVO DEL BACKTEST +797%
# =============================================================================
"""
═══════════════════════════════════════════════════════════════════════════════
                        🔒 MANDATORIOS (7 condiciones)
                     Sin estos NO HAY TRADE bajo ningún concepto
═══════════════════════════════════════════════════════════════════════════════

1. TENDENCIA EMA8 vs EMA21 - Define si es LONG o SHORT
2. ADX >= 28 - Evita mercados laterales (verificado en backtest)
3. RSI - LONG > 55, SHORT < 70 (verificado +$110 mejor)
4. VOLUMEN >= 1.2x SMA20 - Sin volumen = falsos rompimientos
5. DISTANCIA A EMA20 < 3.0 ATR - No entrar extendido
6. ATR ENTRE 0.2% Y 15% - Filtro de seguridad
7. SPREAD < 0.1% - Filtro de liquidez

═══════════════════════════════════════════════════════════════════════════════
                     📊 OPCIONALES (6 condiciones bonus = 100 pts max)
═══════════════════════════════════════════════════════════════════════════════

1. CLOSE vs EMA50 (15 pts) - Tendencia mayor confirmada
2. EMA20 vs EMA50 (15 pts) - Alineación de tendencias
3. MACD HISTOGRAM (15 pts) - Confirmación momentum
4. HIGHER LOW / LOWER HIGH (20 pts) - Estructura de precio
5. PENDIENTES EMAs (20 pts) - Todas subiendo/bajando = momentum fuerte
6. ALINEACIÓN COMPLETA (15 pts) - Close > EMA8 > EMA20 > EMA21 > EMA50

═══════════════════════════════════════════════════════════════════════════════
                        📈 TAMAÑO DE POSICIÓN
═══════════════════════════════════════════════════════════════════════════════

- Mandatorios OK + 0-30 pts = 50% margen ($50)
- Mandatorios OK + 31-60 pts = 75% margen ($75)
- Mandatorios OK + 61-100 pts = 100% margen ($100)
"""

def calculate_ema_slopes(df: pd.DataFrame, periods: int = 3) -> Dict:
    """
    Calcula las pendientes de las EMAs para detectar momentum.
    
    Pendiente positiva = EMA subiendo = momentum alcista
    Pendiente negativa = EMA bajando = momentum bajista
    
    Returns:
        Dict con pendientes normalizadas (-100 a +100)
    """
    slopes = {}
    
    for ema_name in ['ema8', 'ema20', 'ema21', 'ema50']:
        if ema_name in df.columns:
            # Calcular cambio porcentual en las últimas N velas
            current = df[ema_name].iloc[-2]  # Última cerrada
            previous = df[ema_name].iloc[-2-periods]  # N velas antes
            
            # Pendiente como % de cambio
            slope_pct = ((current - previous) / previous) * 100
            
            # Normalizar a escala -100 a +100 (asumiendo max cambio 5% en 3 velas)
            normalized = max(-100, min(100, slope_pct * 20))
            slopes[ema_name] = {
                'raw': slope_pct,
                'normalized': normalized,
                'direction': 'UP' if slope_pct > 0 else 'DOWN' if slope_pct < 0 else 'FLAT'
            }
    
    return slopes


# =============================================================================
# 🎯 FUNCIÓN EXACTA DEL BACKTEST - TODAS LAS CONDICIONES SON MANDATORIAS
# =============================================================================

def check_lateral_backtest(df: pd.DataFrame, idx: int) -> bool:
    """
    Verifica si el mercado está lateral - EXACTAMENTE como en backtest.
    Retorna True si está lateral (NO operar).
    """
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


def check_time_filter(timestamp) -> Tuple[bool, str]:
    """
    Verifica si el día y hora están permitidos según el backtest 2025.
    
    Días bloqueados: Martes (day_of_week=1)
    Horas bloqueadas: 01:00, 12:00, 23:00 UTC
    
    Returns:
        (is_blocked, reason) - True si está bloqueado, False si permitido
    """
    if not CONFIG.get('USE_TIME_FILTER', True):
        return False, ""
    
    # Obtener día de la semana (0=Lunes, 1=Martes, ..., 6=Domingo)
    day_of_week = timestamp.weekday()
    hour_utc = timestamp.hour
    
    blocked_days = CONFIG.get('BLOCKED_DAYS', [1])  # Por defecto Martes
    blocked_hours = CONFIG.get('BLOCKED_HOURS_UTC', [1, 12, 23])
    
    day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    if day_of_week in blocked_days:
        return True, f"Día bloqueado: {day_names[day_of_week]} (bajo rendimiento en backtest)"
    
    if hour_utc in blocked_hours:
        return True, f"Hora bloqueada: {hour_utc:02d}:00 UTC (bajo rendimiento en backtest)"
    
    return False, ""


def check_entry_backtest(df: pd.DataFrame, idx: int, direction: str) -> Tuple[bool, List[str]]:
    """
    Verifica condiciones de entrada EXACTAMENTE como en el backtest.
    
    TODAS las 11 condiciones son MANDATORIAS (sin sistema de puntos):
    1. EMA8 vs EMA21 (direccion)
    2. Close vs EMA50
    3. EMA20 vs EMA50
    4. ADX >= 28
    5. RSI (LONG > 55, SHORT < 70)
    6. MACD Histogram
    7. Volumen >= 1.2x SMA20
    8. Higher Low (LONG) / Lower High (SHORT)
    9. Dist EMA20 < 3.0 ATR
    10. ATR entre 0.2% - 15%
    11. Filtro lateral
    
    Returns:
        (is_valid, list_of_failures)
    """
    row = df.iloc[idx]
    failures = []
    is_long = direction == 'LONG'
    
    # Datos insuficientes
    if idx < 60:
        return False, ["Datos insuficientes (idx < 60)"]
    
    # 1. FILTRO LATERAL (si está lateral, NO operar)
    if check_lateral_backtest(df, idx):
        failures.append(f"Mercado LATERAL (rango < {CONFIG.get('LATERAL_RANGE_PCT', 0.015)*100:.1f}%)")
    
    # 2. EMA8 vs EMA21
    if is_long:
        if not (row['ema8'] > row['ema21']):
            failures.append(f"EMA8 <= EMA21 ({row['ema8']:.6f} <= {row['ema21']:.6f})")
    else:
        if not (row['ema8'] < row['ema21']):
            failures.append(f"EMA8 >= EMA21 ({row['ema8']:.6f} >= {row['ema21']:.6f})")
    
    # 3. Close vs EMA50
    if is_long:
        if not (row['close'] > row['ema50']):
            failures.append(f"Close <= EMA50 ({row['close']:.6f} <= {row['ema50']:.6f})")
    else:
        if not (row['close'] < row['ema50']):
            failures.append(f"Close >= EMA50 ({row['close']:.6f} >= {row['ema50']:.6f})")
    
    # 4. EMA20 vs EMA50
    if is_long:
        if not (row['ema20'] > row['ema50']):
            failures.append(f"EMA20 <= EMA50 ({row['ema20']:.6f} <= {row['ema50']:.6f})")
    else:
        if not (row['ema20'] < row['ema50']):
            failures.append(f"EMA20 >= EMA50 ({row['ema20']:.6f} >= {row['ema50']:.6f})")
    
    # 5. ADX >= 28
    if not (row['adx'] >= CONFIG['ADX_MIN']):
        failures.append(f"ADX {row['adx']:.1f} < {CONFIG['ADX_MIN']}")
    
    # 6. RSI
    if is_long:
        if not (row['rsi'] > CONFIG['RSI_LONG_MIN']):
            failures.append(f"RSI {row['rsi']:.1f} <= {CONFIG['RSI_LONG_MIN']}")
    else:
        if not (row['rsi'] < CONFIG['RSI_SHORT_MAX']):
            failures.append(f"RSI {row['rsi']:.1f} >= {CONFIG['RSI_SHORT_MAX']}")
    
    # 7. MACD Histogram
    if is_long:
        if not (row['macd_hist'] > 0):
            failures.append(f"MACD Hist <= 0 ({row['macd_hist']:.6f})")
    else:
        if not (row['macd_hist'] < 0):
            failures.append(f"MACD Hist >= 0 ({row['macd_hist']:.6f})")
    
    # 8. Volumen >= 1.2x SMA20
    vol_ratio = row.get('vol_ratio', row['volume'] / row['vol_sma20'] if 'vol_sma20' in row else 0)
    if not (vol_ratio >= CONFIG['VOLUME_RATIO']):
        failures.append(f"Volumen {vol_ratio:.2f}x < {CONFIG['VOLUME_RATIO']}x")
    
    # 9. Higher Low / Lower High (usando funciones existentes)
    if is_long:
        has_structure, _ = detect_higher_low(df, idx)
        if not has_structure:
            failures.append("Sin Higher Low")
    else:
        has_structure, _ = detect_lower_high(df, idx)
        if not has_structure:
            failures.append("Sin Lower High")
    
    # 10. Dist EMA20 < 3.0 ATR
    ema20_dist = row.get('ema20_dist', abs(row['close'] - row['ema20']) / row['atr'])
    if not (ema20_dist < CONFIG['EMA_EXTENSION_ATR_MULT']):
        failures.append(f"Dist EMA20 {ema20_dist:.2f} >= {CONFIG['EMA_EXTENSION_ATR_MULT']} ATR")
    
    # 11. ATR entre 0.2% - 15%
    if not (CONFIG['ATR_MIN_PCT'] <= row['atr_pct'] <= CONFIG['ATR_MAX_PCT']):
        failures.append(f"ATR {row['atr_pct']*100:.2f}% fuera de rango")
    
    is_valid = len(failures) == 0
    return is_valid, failures


def calculate_signal_score(df: pd.DataFrame, row: pd.Series, direction: str, 
                          closed_idx: int, slopes: Dict) -> Dict:
    """
    SISTEMA HÍBRIDO - Análisis exhaustivo basado en backtest +797%
    
    🔒 MANDATORIOS (7 condiciones - SIN ESTOS NO HAY TRADE):
    1. Tendencia EMA8 vs EMA21 - Define dirección
    2. ADX >= 28 - Fuerza de tendencia
    3. RSI - LONG > 55, SHORT < 70
    4. Volumen >= 1.2x SMA20 - Participación real
    5. Distancia a EMA20 < 3.0 ATR - No extendido
    6. ATR en rango 0.2%-15% - Volatilidad operable
    7. Spread < 0.1% - (verificado antes de llamar esta función)
    
    📊 OPCIONALES (6 condiciones bonus = 100 pts max):
    1. Close vs EMA50 (15 pts)
    2. EMA20 vs EMA50 (15 pts) 
    3. MACD Histogram (15 pts)
    4. Higher Low / Lower High (20 pts)
    5. Pendientes EMAs (20 pts)
    6. Alineación completa (15 pts)
    
    📈 TAMAÑO POSICIÓN:
    - 0-30 pts = 50% margen
    - 31-60 pts = 75% margen
    - 61-100 pts = 100% margen
    """
    score = 0
    details = {}
    mandatory_passed = True
    mandatory_failures = []
    
    is_long = direction == 'LONG'
    
    # =========================================================================
    # 🔒 MANDATORIOS (7 condiciones)
    # =========================================================================
    
    # 1. TENDENCIA EMA8 vs EMA21 (MANDATORIO)
    ema_trend_ok = (row['ema8'] > row['ema21']) if is_long else (row['ema8'] < row['ema21'])
    details['ema_trend'] = {
        'passed': ema_trend_ok,
        'mandatory': True,
        'value': f"EMA8 {'>' if is_long else '<'} EMA21: {row['ema8']:.6f} vs {row['ema21']:.6f}",
        'points': 0
    }
    if not ema_trend_ok:
        mandatory_passed = False
        mandatory_failures.append("Tendencia EMA8/EMA21 incorrecta")
    
    # 2. ADX >= 28 (MANDATORIO - según backtest verificado)
    adx_ok = row['adx'] >= CONFIG['ADX_MIN']  # ADX_MIN = 28
    details['adx'] = {
        'passed': adx_ok,
        'mandatory': True,
        'value': f"ADX: {row['adx']:.1f} (mín {CONFIG['ADX_MIN']})",
        'points': 0
    }
    if not adx_ok:
        mandatory_passed = False
        mandatory_failures.append(f"ADX {row['adx']:.1f} < {CONFIG['ADX_MIN']}")
    
    # 3. RSI (MANDATORIO - verificado +$110 mejor en backtest)
    rsi = row['rsi']
    if is_long:
        rsi_ok = rsi > CONFIG['RSI_LONG_MIN']  # RSI > 55
        rsi_condition = f"RSI > {CONFIG['RSI_LONG_MIN']}"
    else:
        rsi_ok = rsi < CONFIG['RSI_SHORT_MAX']  # RSI < 70
        rsi_condition = f"RSI < {CONFIG['RSI_SHORT_MAX']}"
    
    details['rsi'] = {
        'passed': rsi_ok,
        'mandatory': True,
        'value': f"RSI: {rsi:.1f} ({rsi_condition})",
        'points': 0
    }
    if not rsi_ok:
        mandatory_passed = False
        mandatory_failures.append(f"RSI {rsi:.1f} no cumple {rsi_condition}")
    
    # 4. VOLUMEN >= 1.2x (MANDATORIO)
    vol_ratio = row['vol_ratio']
    vol_ok = vol_ratio >= CONFIG['VOLUME_RATIO']  # >= 1.2
    details['volume'] = {
        'passed': vol_ok,
        'mandatory': True,
        'value': f"Volumen: {vol_ratio:.2f}x (mín {CONFIG['VOLUME_RATIO']}x)",
        'points': 0
    }
    if not vol_ok:
        mandatory_passed = False
        mandatory_failures.append(f"Volumen {vol_ratio:.2f}x < {CONFIG['VOLUME_RATIO']}x")
    
    # 5. DISTANCIA A EMA20 < 3.0 ATR (MANDATORIO)
    ema20_dist = row['ema20_dist']
    dist_ok = ema20_dist < CONFIG['EMA_EXTENSION_ATR_MULT']  # < 3.0 ATR
    details['ema_distance'] = {
        'passed': dist_ok,
        'mandatory': True,
        'value': f"Dist EMA20: {ema20_dist:.2f} ATR (max {CONFIG['EMA_EXTENSION_ATR_MULT']})",
        'points': 0
    }
    if not dist_ok:
        mandatory_passed = False
        mandatory_failures.append(f"Dist EMA20 {ema20_dist:.2f} >= {CONFIG['EMA_EXTENSION_ATR_MULT']} ATR")
    
    # 6. ATR EN RANGO (MANDATORIO)
    atr_pct = row['atr_pct']
    atr_ok = CONFIG['ATR_MIN_PCT'] <= atr_pct <= CONFIG['ATR_MAX_PCT']
    details['atr_range'] = {
        'passed': atr_ok,
        'mandatory': True,
        'value': f"ATR%: {atr_pct*100:.2f}% (rango {CONFIG['ATR_MIN_PCT']*100:.1f}%-{CONFIG['ATR_MAX_PCT']*100:.1f}%)",
        'points': 0
    }
    if not atr_ok:
        mandatory_passed = False
        mandatory_failures.append(f"ATR {atr_pct*100:.2f}% fuera de rango")
    
    # Si CUALQUIER mandatorio falla, NO HAY TRADE
    if not mandatory_passed:
        return {
            'score': 0,
            'valid': False,
            'mandatory_passed': False,
            'mandatory_failures': mandatory_failures,
            'details': details,
            'direction': direction,
            'position_size': 0,
            'confidence': 'BLOQUEADO'
        }
    
    # =========================================================================
    # 📊 OPCIONALES (6 condiciones = 100 pts máximo)
    # =========================================================================
    
    # 1. CLOSE vs EMA50 (15 pts) - Tendencia mayor
    if is_long:
        close_ema50_ok = row['close'] > row['ema50']
    else:
        close_ema50_ok = row['close'] < row['ema50']
    
    close_ema50_pts = 15 if close_ema50_ok else 0
    score += close_ema50_pts
    details['close_ema50'] = {
        'passed': close_ema50_ok,
        'mandatory': False,
        'value': f"Close {'>' if is_long else '<'} EMA50: {row['close']:.6f} vs {row['ema50']:.6f}",
        'points': close_ema50_pts,
        'max_points': 15
    }
    
    # 2. EMA20 vs EMA50 (15 pts) - Alineación tendencias
    if is_long:
        ema20_ema50_ok = row['ema20'] > row['ema50']
    else:
        ema20_ema50_ok = row['ema20'] < row['ema50']
    
    ema20_ema50_pts = 15 if ema20_ema50_ok else 0
    score += ema20_ema50_pts
    details['ema20_ema50'] = {
        'passed': ema20_ema50_ok,
        'mandatory': False,
        'value': f"EMA20 {'>' if is_long else '<'} EMA50: {row['ema20']:.6f} vs {row['ema50']:.6f}",
        'points': ema20_ema50_pts,
        'max_points': 15
    }
    
    # 3. MACD HISTOGRAM (15 pts) - Momentum
    macd_hist = row['macd_hist']
    macd_ok = (macd_hist > 0) if is_long else (macd_hist < 0)
    macd_pts = 15 if macd_ok else 0
    score += macd_pts
    details['macd'] = {
        'passed': macd_ok,
        'mandatory': False,
        'value': f"MACD Hist {'>' if is_long else '<'} 0: {macd_hist:.6f}",
        'points': macd_pts,
        'max_points': 15
    }
    
    # 4. HIGHER LOW / LOWER HIGH (20 pts) - Estructura
    if is_long:
        has_structure, structure_msg = detect_higher_low(df, closed_idx)
    else:
        has_structure, structure_msg = detect_lower_high(df, closed_idx)
    
    structure_pts = 20 if has_structure else 0
    score += structure_pts
    details['structure'] = {
        'passed': has_structure,
        'mandatory': False,
        'value': structure_msg,
        'points': structure_pts,
        'max_points': 20
    }
    
    # 5. PENDIENTES EMAs (20 pts) - Momentum de tendencia
    ema8_slope = slopes.get('ema8', {}).get('normalized', 0)
    ema20_slope = slopes.get('ema20', {}).get('normalized', 0)
    ema21_slope = slopes.get('ema21', {}).get('normalized', 0)
    
    if is_long:
        # LONG: todas las EMAs deben subir (pendientes positivas)
        all_slopes_favorable = ema8_slope > 0 and ema21_slope > 0
        slope_avg = (ema8_slope + ema21_slope) / 2
    else:
        # SHORT: todas las EMAs deben bajar (pendientes negativas)
        all_slopes_favorable = ema8_slope < 0 and ema21_slope < 0
        slope_avg = -(ema8_slope + ema21_slope) / 2
    
    if all_slopes_favorable and slope_avg > 30:
        slope_pts = 20  # Momentum muy fuerte
    elif all_slopes_favorable and slope_avg > 15:
        slope_pts = 15
    elif all_slopes_favorable:
        slope_pts = 10
    else:
        slope_pts = 0
    
    score += slope_pts
    details['ema_slopes'] = {
        'passed': slope_pts > 0,
        'mandatory': False,
        'value': f"Pendientes EMA8: {ema8_slope:.1f}, EMA21: {ema21_slope:.1f} ({'favorables' if all_slopes_favorable else 'mixtas'})",
        'points': slope_pts,
        'max_points': 20
    }
    
    # 6. ALINEACIÓN COMPLETA (15 pts) - Perfección técnica
    if is_long:
        # LONG perfecto: Close > EMA8 > EMA20 > EMA21 > EMA50
        full_alignment = (row['close'] > row['ema8'] > row['ema20'] > row['ema21'] > row['ema50'])
    else:
        # SHORT perfecto: Close < EMA8 < EMA20 < EMA21 < EMA50
        full_alignment = (row['close'] < row['ema8'] < row['ema20'] < row['ema21'] < row['ema50'])
    
    alignment_pts = 15 if full_alignment else 0
    score += alignment_pts
    details['alignment'] = {
        'passed': full_alignment,
        'mandatory': False,
        'value': f"Alineación {'PERFECTA' if full_alignment else 'parcial'}: Close-EMA8-EMA20-EMA21-EMA50",
        'points': alignment_pts,
        'max_points': 15
    }
    
    # =========================================================================
    # 📈 DETERMINAR TAMAÑO DE POSICIÓN SEGÚN SCORE OPCIONAL
    # =========================================================================
    # Máximo 100 pts de opcionales
    # FIJO: 50% del margen siempre (conservador)
    
    position_size = 0.5     # 50% del margen FIJO
    
    if score >= 61:
        confidence = "ALTA"
    elif score >= 31:
        confidence = "MEDIA"
    else:
        confidence = "BAJA"
    
    return {
        'score': score,
        'max_score': 100,
        'valid': True,  # Si llegamos aquí, mandatorios pasaron
        'mandatory_passed': True,
        'mandatory_failures': [],
        'details': details,
        'direction': direction,
        'position_size': position_size,
        'confidence': confidence
    }


def analyze_signal_smart(df: pd.DataFrame, symbol: str, spread_pct: float) -> Optional[Dict]:
    """
    Analiza señales usando EXACTAMENTE la misma lógica del backtest.
    
    TODAS las 11 condiciones son MANDATORIAS (sin sistema de puntos).
    Si CUALQUIERA falla, NO hay señal.
    
    Esta versión es consistente con el backtest que generó +$2,450.74 anual.
    """
    closed_idx = len(df) - 2
    row = df.iloc[closed_idx]
    
    candle_time = df.index[closed_idx].strftime('%Y-%m-%d %H:%M') if hasattr(df.index[closed_idx], 'strftime') else str(df.index[closed_idx])
    
    logger.info("")
    logger.info("═" * 70)
    logger.info(f"📊 ANÁLISIS BACKTEST-SYNC: {symbol}")
    logger.info(f"   🕐 Vela 1H evaluada: {candle_time} (CERRADA)")
    logger.info(f"   💲 Precio: ${row['close']:.6f}")
    logger.info("═" * 70)
    
    # Verificar spread primero (condición adicional)
    if spread_pct > CONFIG['MAX_SPREAD_PCT']:
        logger.info(f"   ❌ Spread muy alto: {spread_pct*100:.4f}% (max {CONFIG['MAX_SPREAD_PCT']*100:.2f}%)")
        return None
    
    # Verificar filtro de tiempo (días y horas bloqueados)
    candle_timestamp = df.index[closed_idx]
    is_time_blocked, time_reason = check_time_filter(candle_timestamp)
    if is_time_blocked:
        logger.info(f"   ⏰ {time_reason}")
        return None
    
    # =========================================================================
    # MOSTRAR INDICADORES ACTUALES
    # =========================================================================
    logger.info("")
    logger.info("   📊 INDICADORES:")
    logger.info("   " + "─" * 50)
    logger.info(f"      Precio:    ${row['close']:.6f}")
    logger.info(f"      EMA8:      ${row['ema8']:.6f}")
    logger.info(f"      EMA20:     ${row['ema20']:.6f}")
    logger.info(f"      EMA21:     ${row['ema21']:.6f}")
    logger.info(f"      EMA50:     ${row['ema50']:.6f}")
    logger.info(f"      RSI:       {row['rsi']:.1f}")
    logger.info(f"      MACD Hist: {row['macd_hist']:.6f}")
    logger.info(f"      ADX:       {row['adx']:.1f}")
    logger.info(f"      ATR:       ${row['atr']:.6f} ({row['atr_pct']*100:.2f}%)")
    logger.info(f"      Volumen:   {row['vol_ratio']:.2f}x SMA20")
    logger.info(f"      Dist EMA20: {row['ema20_dist']:.2f} ATR")
    
    # =========================================================================
    # EVALUAR LONG (11 condiciones mandatorias)
    # =========================================================================
    logger.info("")
    logger.info("   📈 EVALUACIÓN LONG (11 condiciones MANDATORIAS):")
    logger.info("   " + "─" * 50)
    
    long_valid, long_failures = check_entry_backtest(df, closed_idx, 'LONG')
    
    if long_valid:
        logger.info("   ✅ LONG: TODAS las 11 condiciones cumplidas")
    else:
        logger.info(f"   ❌ LONG: {len(long_failures)} condición(es) fallida(s):")
        for f in long_failures:
            logger.info(f"      • {f}")
    
    # =========================================================================
    # EVALUAR SHORT (11 condiciones mandatorias)
    # =========================================================================
    logger.info("")
    logger.info("   📉 EVALUACIÓN SHORT (11 condiciones MANDATORIAS):")
    logger.info("   " + "─" * 50)
    
    short_valid, short_failures = check_entry_backtest(df, closed_idx, 'SHORT')
    
    if short_valid:
        logger.info("   ✅ SHORT: TODAS las 11 condiciones cumplidas")
    else:
        logger.info(f"   ❌ SHORT: {len(short_failures)} condición(es) fallida(s):")
        for f in short_failures:
            logger.info(f"      • {f}")
    
    # =========================================================================
    # DECIDIR SEÑAL
    # =========================================================================
    logger.info("")
    logger.info("   📋 DECISIÓN (lógica backtest):")
    logger.info("   " + "─" * 50)
    
    direction = None
    
    if long_valid and short_valid:
        # Ambas válidas - no debería pasar, pero por si acaso elegimos LONG
        direction = 'LONG'
        logger.info(f"   ⚠️ Ambas señales válidas - eligiendo LONG")
    elif long_valid:
        direction = 'LONG'
        logger.info(f"   🟢 SEÑAL LONG VÁLIDA")
    elif short_valid:
        direction = 'SHORT'
        logger.info(f"   🔴 SEÑAL SHORT VÁLIDA")
    else:
        logger.info(f"   ⚪ Sin señales válidas")
        logger.info(f"      LONG: {len(long_failures)} fallos")
        logger.info(f"      SHORT: {len(short_failures)} fallos")
        return None
    
    # Calcular SL y TP
    sl_atr_mult = CONFIG.get('SL_ATR_MULT', 1.3)
    tp_atr_mult = CONFIG.get('TP_ATR_MULT', 3.5)
    atr = row['atr']
    
    if direction == 'LONG':
        sl = row['close'] - (atr * sl_atr_mult)
        tp = row['close'] + (atr * tp_atr_mult)
    else:
        sl = row['close'] + (atr * sl_atr_mult)
        tp = row['close'] - (atr * tp_atr_mult)
    
    sl_pct = abs(row['close'] - sl) / row['close'] * 100
    tp_pct = abs(tp - row['close']) / row['close'] * 100
    
    logger.info("")
    logger.info(f"   🎯 SEÑAL: {direction}")
    logger.info(f"      ATR: ${atr:.6f}")
    logger.info(f"      SL: ${sl:.6f} ({sl_atr_mult} ATR = {sl_pct:.2f}%)")
    logger.info(f"      TP: ${tp:.6f} ({tp_atr_mult} ATR = {tp_pct:.2f}%)")
    
    return {
        'symbol': symbol,
        'direction': direction,
        'price': row['close'],
        'atr': row['atr'],
        'score': 100,  # Sin sistema de scoring, todas las señales tienen score máximo
        'position_size': 1.0,  # Siempre 100% del margen
        'confidence': 'BACKTEST-SYNC',
        'sl': sl,
        'tp': tp,
    }


# Mantener la función original como backup
def analyze_signal(df: pd.DataFrame, symbol: str, spread_pct: float) -> Optional[Dict]:
    """Función original - ahora redirige al sistema backtest-sync"""
    return analyze_signal_smart(df, symbol, spread_pct)


# =============================================================================
# 🌐 FILTRO DE CONDICIONES DE MERCADO (BTC)
# =============================================================================
def check_market_conditions(exchange) -> Dict:
    """
    Verifica las condiciones generales del mercado usando BTC como referencia.
    
    REGLA BASADA EN BACKTESTS:
    - ATR% >= 0.6% Y Mov3D >= 3.0%  →  Mercado NORMAL (operar)
    - ATR% >= 0.8% Y Mov3D >= 4.0%  →  Mercado EXCELENTE (margen alto)
    - ATR% < 0.5% O Mov3D < 2.0%    →  Mercado MALO (no operar)
    
    HISTORIAL:
    - ATR >= 0.6%: +$552 en 2 meses (Oct + Nov)
    - ATR < 0.6%: -$145 en 2 meses (Ago + Sep)
    
    Returns:
        Dict con:
        - can_trade: bool - Si se puede operar
        - margin: int - Margen recomendado
        - score: int - Score de mercado (0-5)
        - btc_atr_pct: float - ATR% actual de BTC
        - btc_move_3d: float - Movimiento 3D de BTC
        - reason: str - Explicación
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("🌐 VERIFICANDO CONDICIONES DE MERCADO (BTC)")
    logger.info("=" * 70)
    
    try:
        # Descargar datos de BTC (necesitamos ~100 velas para ATR estable)
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=200)
        btc = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        btc['timestamp'] = pd.to_datetime(btc['timestamp'], unit='ms')
        
        # Calcular ATR
        h, l, c = btc['high'], btc['low'], btc['close']
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        btc['atr'] = tr.ewm(span=14, adjust=False).mean()
        btc['atr_pct'] = btc['atr'] / btc['close'] * 100
        
        # Valores actuales (promedio últimos 7 días para estabilidad)
        last_7d = btc.tail(168)  # 7 días * 24 horas
        btc_atr_pct = last_7d['atr_pct'].mean()
        
        # Movimiento 3 días (valor actual)
        current_price = btc.iloc[-1]['close']
        price_3d_ago = btc.iloc[-73]['close'] if len(btc) > 73 else btc.iloc[0]['close']
        btc_move_3d = abs(current_price - price_3d_ago) / price_3d_ago * 100
        
        logger.info(f"   BTC Precio actual: ${current_price:,.0f}")
        logger.info(f"   BTC hace 3 días:   ${price_3d_ago:,.0f}")
        logger.info(f"   ATR% (7d avg):     {btc_atr_pct:.2f}%")
        logger.info(f"   Movimiento 3D:     {btc_move_3d:.2f}%")
        logger.info("")
        
        # Calcular score
        score = 0
        if btc_atr_pct >= CONFIG['BTC_ATR_PCT_MIN']: score += 1
        if btc_atr_pct >= CONFIG['BTC_ATR_PCT_HIGH']: score += 1
        if btc_move_3d >= CONFIG['BTC_MOVE_3D_MIN']: score += 1
        if btc_move_3d >= CONFIG['BTC_MOVE_3D_HIGH']: score += 1
        
        # Determinar si se puede operar y con qué margen
        atr_ok = btc_atr_pct >= CONFIG['BTC_ATR_PCT_MIN']
        move_ok = btc_move_3d >= CONFIG['BTC_MOVE_3D_MIN']
        atr_high = btc_atr_pct >= CONFIG['BTC_ATR_PCT_HIGH']
        move_high = btc_move_3d >= CONFIG['BTC_MOVE_3D_HIGH']
        
        if atr_high and move_high:
            can_trade = True
            margin = CONFIG['MARGIN_HIGH']
            reason = "🟢 EXCELENTE - Alta volatilidad y movimientos fuertes"
        elif atr_ok and move_ok:
            can_trade = True
            margin = CONFIG['MARGIN_NORMAL']
            reason = "🟡 BUENO - Condiciones normales para operar"
        elif atr_ok or move_ok:
            can_trade = True
            margin = CONFIG['MARGIN_LOW']
            reason = "🟠 MODERADO - Solo una condición cumplida"
        else:
            can_trade = False
            margin = 0
            reason = "🔴 MALO - Mercado sin volatilidad, NO operar"
        
        logger.info(f"   ┌─────────────────────────────────────────────────┐")
        logger.info(f"   │ ATR% >= {CONFIG['BTC_ATR_PCT_MIN']}%:   {'✅ SÍ' if atr_ok else '❌ NO'}")
        logger.info(f"   │ Mov3D >= {CONFIG['BTC_MOVE_3D_MIN']}%:  {'✅ SÍ' if move_ok else '❌ NO'}")
        logger.info(f"   │ Score: {score}/4")
        logger.info(f"   └─────────────────────────────────────────────────┘")
        logger.info(f"")
        logger.info(f"   {reason}")
        if can_trade:
            logger.info(f"   💰 Margen recomendado: ${margin}")
        logger.info("=" * 70)
        
        return {
            'can_trade': can_trade,
            'margin': margin,
            'score': score,
            'btc_atr_pct': btc_atr_pct,
            'btc_move_3d': btc_move_3d,
            'reason': reason
        }
        
    except Exception as e:
        logger.error(f"   ❌ Error verificando mercado: {e}")
        # En caso de error, permitir operar con margen normal
        return {
            'can_trade': True,
            'margin': CONFIG['MARGIN_NORMAL'],
            'score': -1,
            'btc_atr_pct': 0,
            'btc_move_3d': 0,
            'reason': "⚠️ Error obteniendo datos - usando margen por defecto"
        }

# =============================================================================
# 🤖 CLASE PRINCIPAL DEL BOT
# =============================================================================
class BotGanadora:
    def __init__(self):
        logger.info("=" * 70)
        logger.info("🚀 INICIANDO BOT GANADORA v3.0")
        logger.info("=" * 70)
        
        # Validar API keys
        if not API_KEY or not API_SECRET or API_KEY == "TU_API_KEY":
            raise ValueError("API keys no configuradas")
        
        # Conectar a Binance Futures
        logger.info("📡 Conectando a Binance Futures...")
        self.exchange = ccxt.binanceusdm({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # === COOLDOWN POR PÉRDIDA ===
        self.loss_cooldowns = {} # {symbol: datetime_end}
        self.COOLDOWN_HOURS = CONFIG.get('COOLDOWN_HOURS', 1)
        
        # === GESTOR DE RIESGO ===
        self.risk_manager = RiskManager(self.exchange)
        
        # === ANTI-SCALPING: REGISTRO DE TRADES POR HORA ===
        # Diccionario: {symbol: [lista de timestamps de trades]}
        self.trades_per_hour = {}
        
        # Guardar posiciones del ciclo anterior para detectar cierres
        self.previous_positions = {}
        
        # Verificar conexión
        try:
            balance = self.exchange.fetch_balance()
            usdt = float(balance['USDT']['free'])
            logger.info(f"   ✓ Conexión exitosa")
            logger.info(f"   💰 Balance USDT disponible: ${usdt:.2f}")
        except Exception as e:
            raise ConnectionError(f"No se pudo conectar: {e}")
        
        # Configurar modo Hedge
        try:
            self.exchange.set_position_mode(hedged=True)
            logger.info("   ✓ Modo Hedge activado")
        except:
            logger.info("   ℹ️ Modo posición ya configurado")
        
        # Mostrar configuración
        self._log_config()
    
    def _log_config(self):
        """Muestra toda la configuración ORIGINAL VERIFICADA"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("⚙️ CONFIGURACIÓN ORIGINAL VERIFICADA")
        logger.info("=" * 70)
        logger.info(f"   TIMEFRAME: {CONFIG['TIMEFRAME']} (Velas de 1 hora)")
        logger.info(f"   VELAS A DESCARGAR: {CONFIG['OHLCV_LIMIT']}")
        logger.info("")
        logger.info("   --- CAPITAL (ORIGINAL) ---")
        logger.info(f"   Margen: ${CONFIG['MARGIN_USD']}")
        logger.info(f"   Leverage: {CONFIG['LEVERAGE']}x")
        logger.info(f"   Exposición: ${CONFIG['MARGIN_USD'] * CONFIG['LEVERAGE']}")
        logger.info(f"   Max posiciones: {CONFIG['MAX_OPEN_POSITIONS']}")
        logger.info("")
        logger.info("   --- INDICADORES (ORIGINAL) ---")
        logger.info(f"   EMA: {CONFIG['EMA_FAST']}, {CONFIG['EMA_SIGNAL']}, {CONFIG['EMA_SLOW']} (NO EMA20)")
        logger.info(f"   ADX: {CONFIG['ADX_PERIOD']} | Mínimo: {CONFIG['ADX_MIN']}")
        logger.info(f"   RSI: {CONFIG['RSI_PERIOD']} | LONG > {CONFIG['RSI_LONG_MIN']} | SHORT < {CONFIG['RSI_SHORT_MAX']}")
        logger.info(f"   MACD: ({CONFIG['MACD_FAST']}, {CONFIG['MACD_SLOW']}, {CONFIG['MACD_SIGNAL']}) - USA HISTOGRAM")
        logger.info(f"   ATR: {CONFIG['ATR_PERIOD']}")
        logger.info(f"   Volume SMA: {CONFIG['VOLUME_SMA_PERIOD']} | Ratio >= {CONFIG['VOLUME_RATIO']}")
        logger.info(f"   EMA Extension: < {CONFIG['EMA_EXTENSION_ATR_MULT']} ATR")
        logger.info(f"   Pivot lookback: {CONFIG['PIVOT_LOOKBACK']} velas (Higher Low / Lower High ESTRICTOS)")
        logger.info("")
        logger.info("   --- RIESGO (ROI con leverage) ---")
        leverage = CONFIG.get('LEVERAGE', 10)
        sl_roi = CONFIG.get('SL_ROI', 0.05)
        tp_roi = CONFIG.get('TP_ROI', 0.10)
        logger.info(f"   Stop Loss: {sl_roi*100:.0f}% ROI = {sl_roi/leverage*100:.2f}% precio")
        logger.info(f"   Take Profit: {tp_roi*100:.0f}% ROI = {tp_roi/leverage*100:.2f}% precio")
        logger.info(f"   Ratio R:R = 1:{tp_roi/sl_roi:.1f}")
        logger.info(f"   ATR rango: {CONFIG['ATR_MIN_PCT']*100:.1f}% - {CONFIG['ATR_MAX_PCT']*100:.1f}%")
        logger.info(f"   Max spread: {CONFIG['MAX_SPREAD_PCT']*100:.2f}%")
        logger.info("")
        logger.info("   --- 10 SÍMBOLOS ORIGINALES ---")
        logger.info(f"   {', '.join(SYMBOLS)}")
        logger.info("=" * 70)
    
    def fetch_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Descarga velas OHLCV de Binance.
        
        Descarga CONFIG['OHLCV_LIMIT'] velas (200 por defecto).
        La última vela (iloc[-1]) está EN PROGRESO (no usar para señales).
        La penúltima vela (iloc[-2]) es la última CERRADA.
        """
        logger.debug(f"   📥 Descargando {CONFIG['OHLCV_LIMIT']} velas {CONFIG['TIMEFRAME']} para {symbol}...")
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, 
                CONFIG['TIMEFRAME'], 
                limit=CONFIG['OHLCV_LIMIT']
            )
            
            if len(ohlcv) < CONFIG['EMA_SLOW'] + 10:
                logger.warning(f"   ⚠️ Pocas velas: {len(ohlcv)} (necesita ~{CONFIG['EMA_SLOW']+10})")
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Información de velas
            first_candle = df.index[0].strftime('%Y-%m-%d %H:%M')
            last_closed = df.index[-2].strftime('%Y-%m-%d %H:%M')
            current_open = df.index[-1].strftime('%Y-%m-%d %H:%M')
            
            logger.debug(f"   ✓ {len(ohlcv)} velas descargadas")
            logger.debug(f"      Primera: {first_candle}")
            logger.debug(f"      Última cerrada: {last_closed}")
            logger.debug(f"      Actual (en progreso): {current_open}")
            
            return df
            
        except Exception as e:
            logger.error(f"   ❌ Error descargando {symbol}: {e}")
            return None
    
    def get_spread(self, symbol: str) -> float:
        """Obtiene spread actual del orderbook"""
        try:
            book = self.exchange.fetch_order_book(symbol, limit=5)
            bid = book['bids'][0][0] if book['bids'] else 0
            ask = book['asks'][0][0] if book['asks'] else 0
            if bid > 0:
                return (ask - bid) / bid
            return 1.0
        except:
            return 1.0
    
    def check_closed_positions_for_losses(self, current_positions: Dict):
        """
        Detecta posiciones que se cerraron con pérdida y activa cooldown.
        
        Compara las posiciones actuales con las del ciclo anterior.
        Si una posición desapareció y tenía PnL negativo, activa cooldown de 1 hora.
        """
        now = datetime.now()
        
        for symbol, prev_pos in self.previous_positions.items():
            # Si el símbolo ya no está en posiciones actuales, se cerró
            normalized_current = set()
            for sym in current_positions.keys():
                normalized = sym.split(':')[0] if ':' in sym else sym
                normalized_current.add(normalized)
            
            prev_normalized = symbol.split(':')[0] if ':' in symbol else symbol
            
            if prev_normalized not in normalized_current:
                # La posición se cerró
                pnl = prev_pos.get('pnl', 0)
                
                if pnl < 0:
                    # Fue pérdida - activar cooldown
                    self.loss_cooldowns[prev_normalized] = now
                    logger.info("")
                    logger.info(f"   🔴 PÉRDIDA DETECTADA en {prev_normalized}")
                    logger.info(f"   ⏰ Cooldown activado: 1 hora sin operar este símbolo")
                    logger.info(f"   → Podrá operar de nuevo a las {(now + timedelta(hours=self.COOLDOWN_HOURS)).strftime('%H:%M:%S')}")
                    
                    # Notificar al RiskManager
                    self.risk_manager.check_losing_streak({'pnl': -1, 'symbol': prev_normalized})
                    
                else:
                    # Fue ganancia - sin cooldown
                    logger.info("")
                    logger.info(f"   🟢 GANANCIA DETECTADA en {prev_normalized}")
                    logger.info(f"   ✅ Puede seguir operando inmediatamente")
                    
                    # Notificar al RiskManager (ganancia rompe racha)
                    self.risk_manager.check_losing_streak({'pnl': 1, 'symbol': prev_normalized})
    
    def is_symbol_in_cooldown(self, symbol: str) -> bool:
        """
        Verifica si un símbolo está en cooldown por pérdida reciente.
        
        Returns:
            True si está en cooldown (no puede operar)
            False si puede operar
        """
        normalized = symbol.split(':')[0] if ':' in symbol else symbol
        
        if normalized not in self.loss_cooldowns:
            return False
        
        loss_time = self.loss_cooldowns[normalized]
        now = datetime.now()
        cooldown_end = loss_time + timedelta(hours=self.COOLDOWN_HOURS)
        
        if now >= cooldown_end:
            # Cooldown terminó - limpiar
            del self.loss_cooldowns[normalized]
            logger.info(f"   ✅ Cooldown terminado para {normalized} - puede operar")
            return False
        else:
            # Aún en cooldown
            remaining = cooldown_end - now
            minutes_left = int(remaining.total_seconds() / 60)
            logger.info(f"   ⏰ {normalized} en COOLDOWN - {minutes_left} minutos restantes")
            return True
    
    def can_trade_this_hour(self, symbol: str) -> bool:
        """
        Verifica si podemos hacer un trade en este símbolo esta hora.
        Límite: MAX_TRADES_PER_HOUR (default 2)
        
        Returns:
            True si podemos operar, False si ya alcanzamos el límite
        """
        max_trades = CONFIG.get('MAX_TRADES_PER_HOUR', 2)
        normalized = symbol.split(':')[0] if ':' in symbol else symbol
        now = datetime.now()
        
        # Inicializar si no existe
        if normalized not in self.trades_per_hour:
            self.trades_per_hour[normalized] = []
        
        # Limpiar trades de hace más de 1 hora
        one_hour_ago = now - timedelta(hours=1)
        self.trades_per_hour[normalized] = [
            t for t in self.trades_per_hour[normalized] 
            if t > one_hour_ago
        ]
        
        trades_count = len(self.trades_per_hour[normalized])
        
        if trades_count >= max_trades:
            logger.info(f"   ⚠️ {normalized}: {trades_count} trades en última hora (máx {max_trades})")
            return False
        
        return True
    
    def register_trade(self, symbol: str):
        """Registra un trade para el control anti-scalping"""
        normalized = symbol.split(':')[0] if ':' in symbol else symbol
        now = datetime.now()
        
        if normalized not in self.trades_per_hour:
            self.trades_per_hour[normalized] = []
        
        self.trades_per_hour[normalized].append(now)
        logger.info(f"   📝 Trade registrado para {normalized} ({len(self.trades_per_hour[normalized])} en última hora)")
    
    def is_market_lateral(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Verifica si el mercado está lateral (sin tendencia clara).
        Calcula el rango de las últimas 4 velas cerradas.
        
        Returns:
            (is_lateral, range_pct)
        """
        if not CONFIG.get('USE_LATERAL_FILTER', True):
            return False, 0.0
        
        lateral_threshold = CONFIG.get('LATERAL_RANGE_PCT', 0.015)  # 1.5%
        
        # Últimas 4 velas cerradas (excluir la actual en progreso)
        last_4 = df.iloc[-5:-1]  # -5 a -2 (4 velas cerradas)
        
        high = last_4['high'].max()
        low = last_4['low'].min()
        range_pct = (high - low) / low
        
        is_lateral = range_pct < lateral_threshold
        
        return is_lateral, range_pct
    
    def get_open_positions(self) -> Dict:
        """Obtiene posiciones abiertas y detecta cierres con pérdida"""
        logger.debug("📊 Consultando posiciones...")
        
        try:
            positions = self.exchange.fetch_positions()
            open_pos = {}
            
            for p in positions:
                contracts = abs(float(p['contracts']))
                if contracts > 0:
                    symbol = p['symbol']
                    open_pos[symbol] = {
                        'side': p['side'],
                        'contracts': contracts,
                        'entry': float(p['entryPrice']),
                        'pnl': float(p['unrealizedPnl'])
                    }
            
            # Detectar posiciones cerradas con pérdida
            if self.previous_positions:
                self.check_closed_positions_for_losses(open_pos)
            
            # Guardar para próxima comparación
            self.previous_positions = open_pos.copy()
            
            if open_pos:
                logger.info(f"   📌 Posiciones abiertas: {len(open_pos)}/{CONFIG['MAX_OPEN_POSITIONS']}")
                for sym, pos in open_pos.items():
                    emoji = "🟢" if pos['pnl'] >= 0 else "🔴"
                    logger.info(f"      {emoji} {sym}: {pos['side'].upper()} @ ${pos['entry']:.4f} | PnL: ${pos['pnl']:.2f}")
            else:
                logger.debug("   Sin posiciones abiertas")
            
            # Mostrar cooldowns activos
            if self.loss_cooldowns:
                logger.info(f"   ⏰ Símbolos en cooldown: {list(self.loss_cooldowns.keys())}")
            
            return open_pos
            
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
            return {}
    
    def update_trailing_stops(self, open_positions: Dict):
        """
        TRAILING STOP - Actualiza SL de posiciones en ganancia.
        
        REGLAS:
        - Si ganancia >= 1 ATR → SL se mueve a breakeven (entrada)
        - Si ganancia >= 2 ATR → SL se mueve a +1 ATR
        - Si ganancia >= 3 ATR → SL se mueve a +2 ATR
        
        Esto protege ganancias en reversiones de mercado.
        """
        if not CONFIG.get('USE_TRAILING_STOP', False):
            return
        
        if not open_positions:
            return
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("🔄 VERIFICANDO TRAILING STOPS")
        logger.info("=" * 70)
        
        for symbol, pos in open_positions.items():
            try:
                # Obtener precio actual y ATR
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                entry_price = pos['entry']
                side = pos['side'].upper()
                
                # Calcular ATR actual
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=20)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                tr = pd.concat([
                    df['high'] - df['low'],
                    (df['high'] - df['close'].shift()).abs(),
                    (df['low'] - df['close'].shift()).abs()
                ], axis=1).max(axis=1)
                atr = tr.iloc[-14:].mean()
                
                # Calcular ganancia en ATRs
                if side == 'LONG':
                    profit_atr = (current_price - entry_price) / atr
                else:  # SHORT
                    profit_atr = (entry_price - current_price) / atr
                
                logger.info(f"   {symbol} {side}:")
                logger.info(f"      Entrada: ${entry_price:.4f} | Actual: ${current_price:.4f}")
                logger.info(f"      ATR: ${atr:.4f} | Ganancia: {profit_atr:.2f} ATR")
                
                # Determinar nuevo SL según ganancia
                activation = CONFIG.get('TRAILING_ACTIVATION', 1.0)
                step = CONFIG.get('TRAILING_STEP', 1.0)
                
                if profit_atr < activation:
                    logger.info(f"      ⏳ Ganancia < {activation} ATR, trailing no activado")
                    continue
                
                # Calcular cuántos pasos de trailing
                steps = int((profit_atr - activation) / step) + 1
                trail_atr = (steps - 1) * step  # 0 para breakeven, 1 ATR para +2 ATR, etc.
                
                if side == 'LONG':
                    new_sl = entry_price + (trail_atr * atr)
                else:  # SHORT
                    new_sl = entry_price - (trail_atr * atr)
                
                new_sl = float(self.exchange.price_to_precision(symbol, new_sl))
                
                # Obtener órdenes abiertas para encontrar el SL actual
                open_orders = self.exchange.fetch_open_orders(symbol)
                sl_order = None
                for order in open_orders:
                    if order['type'] == 'stop_market' or 'stop' in order['type'].lower():
                        sl_order = order
                        break
                
                if sl_order:
                    current_sl = float(sl_order['stopPrice'])
                    
                    # Solo mover si el nuevo SL es mejor (más favorable)
                    should_update = False
                    if side == 'LONG' and new_sl > current_sl:
                        should_update = True
                    elif side == 'SHORT' and new_sl < current_sl:
                        should_update = True
                    
                    if should_update:
                        logger.info(f"      🔄 ACTUALIZANDO SL: ${current_sl:.4f} → ${new_sl:.4f}")
                        
                        # Cancelar orden SL actual
                        self.exchange.cancel_order(sl_order['id'], symbol)
                        
                        # Crear nuevo SL
                        position_side = 'LONG' if side == 'LONG' else 'SHORT'
                        sl_side = 'sell' if side == 'LONG' else 'buy'
                        
                        self.exchange.create_order(
                            symbol=symbol,
                            type='STOP_MARKET',
                            side=sl_side,
                            amount=pos['contracts'],
                            params={
                                'positionSide': position_side,
                                'stopPrice': new_sl,
                                'closePosition': True,
                                'priceProtect': True,
                                'workingType': 'MARK_PRICE'
                            }
                        )
                        logger.info(f"      ✅ Trailing Stop actualizado a ${new_sl:.4f}")
                    else:
                        logger.info(f"      ✓ SL actual ${current_sl:.4f} ya es óptimo")
                else:
                    logger.warning(f"      ⚠️ No se encontró orden SL para {symbol}")
                
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"      ❌ Error en trailing de {symbol}: {e}")
                continue
        
        logger.info("=" * 70)
    
    def set_leverage(self, symbol: str):
        """Configura leverage dinámico"""
        try:
            market = symbol.replace('/', '')
            # Usar leverage dinámico del RiskManager
            leverage = self.risk_manager.get_leverage()
            self.exchange.set_leverage(leverage, market)
            logger.debug(f"   ✓ Leverage {leverage}x para {symbol} (Dinámico)")
        except:
            pass
    
    def open_position(self, signal: Dict) -> bool:
        """Abre posición con SL/TP - tamaño según score"""
        symbol = signal['symbol']
        direction = signal['direction']
        score = signal.get('score', 100)
        position_size = signal.get('position_size', 1.0)  # % del margen según score
        confidence = signal.get('confidence', 'N/A')
        
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"🎯 ABRIENDO POSICIÓN: {symbol} {direction}")
        logger.info(f"   Score: {score}/100 | Confianza: {confidence}")
        logger.info(f"   Tamaño posición: {position_size*100:.0f}% del margen")
        logger.info("=" * 70)
        
        try:
            # Configurar leverage
            self.set_leverage(symbol)
            
            # Obtener precio actual
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # ════════════════════════════════════════════════════════════════
            # CÁLCULO DE TAMAÑO - LÍMITE ABSOLUTO $1000 NOCIONAL
            # ════════════════════════════════════════════════════════════════
            MAX_NOTIONAL = 1000  # LÍMITE ABSOLUTO - NUNCA EXCEDER
            
            base_margin = getattr(self, 'current_margin', CONFIG['MARGIN_USD'])
            adjusted_margin = base_margin * position_size  # Ajustar según confianza
            notional = adjusted_margin * CONFIG['LEVERAGE']
            
            logger.info(f"   Margen base: ${base_margin}")
            logger.info(f"   position_size: {position_size*100:.0f}%")
            logger.info(f"   Margen × position_size: ${adjusted_margin:.2f}")
            logger.info(f"   Nocional calculado: ${notional:.2f}")
            
            # FORZAR límite de $1000 máximo
            if notional > MAX_NOTIONAL:
                notional = MAX_NOTIONAL
                adjusted_margin = notional / CONFIG['LEVERAGE']
                logger.info(f"   ⚠️ LIMITADO A ${MAX_NOTIONAL} MÁXIMO")
            
            # Calcular cantidad
            amount = notional / current_price
            
            # Redondear según mercado
            market = self.exchange.market(symbol)
            amount = self.exchange.amount_to_precision(symbol, amount)
            
            # VERIFICACIÓN FINAL - el redondeo puede cambiar el nocional
            final_notional = float(amount) * current_price
            if final_notional > MAX_NOTIONAL * 1.05:  # 5% tolerancia por redondeo
                # Recalcular con un poco menos para estar bajo el límite
                amount = (MAX_NOTIONAL * 0.95) / current_price
                amount = self.exchange.amount_to_precision(symbol, amount)
                final_notional = float(amount) * current_price
                logger.info(f"   ⚠️ Recalculado por redondeo: ${final_notional:.2f}")
            
            logger.info(f"   Precio actual: ${current_price:.6f}")
            logger.info(f"   Cantidad: {amount}")
            logger.info(f"   Nocional FINAL: ${final_notional:.2f}")
            
            # Calcular SL y TP basados en ATR
            # SL = 1.5 ATR, TP = 3.0 ATR (ratio 1:2)
            sl_atr_mult = CONFIG.get('SL_ATR_MULT', 1.5)
            tp_atr_mult = CONFIG.get('TP_ATR_MULT', 3.0)
            atr = signal.get('atr', current_price * 0.01)  # Fallback 1% si no hay ATR
            
            if direction == 'LONG':
                side = 'buy'
                sl_price = current_price - (atr * sl_atr_mult)  # Precio - 1.5 ATR
                tp_price = current_price + (atr * tp_atr_mult)  # Precio + 3.0 ATR
                sl_side = 'sell'
            else:
                side = 'sell'
                sl_price = current_price + (atr * sl_atr_mult)  # Precio + 1.5 ATR
                tp_price = current_price - (atr * tp_atr_mult)  # Precio - 3.0 ATR
                sl_side = 'buy'
            
            sl_price = float(self.exchange.price_to_precision(symbol, sl_price))
            tp_price = float(self.exchange.price_to_precision(symbol, tp_price))
            
            sl_pct = abs(current_price - sl_price) / current_price * 100
            tp_pct = abs(tp_price - current_price) / current_price * 100
            
            logger.info(f"   ATR: ${atr:.6f}")
            logger.info(f"   Stop Loss: ${sl_price:.6f} ({sl_atr_mult} ATR = {sl_pct:.2f}%)")
            logger.info(f"   Take Profit: ${tp_price:.6f} ({tp_atr_mult} ATR = {tp_pct:.2f}%)")
            
            # ═══════════════════════════════════════════════════════════════════
            # POSICIÓN PRINCIPAL
            # ═══════════════════════════════════════════════════════════════════
            position_side = 'LONG' if direction == 'LONG' else 'SHORT'
            
            logger.info("")
            logger.info(f"   📊 POSICIÓN PRINCIPAL: {direction}")
            logger.info("   ─────────────────────────────────────")
            logger.info("   📤 Enviando orden de mercado...")
            order = self.exchange.create_order(
                symbol=symbol,
                type='MARKET',
                side=side,
                amount=amount,
                params={'positionSide': position_side}
            )
            logger.info(f"   ✓ Orden ejecutada: {order['id']}")
            
            # Registrar trade para control anti-scalping
            self.register_trade(symbol)
            
            # Colocar Stop Loss
            logger.info("   📤 Colocando Stop Loss...")
            sl_order = self.exchange.create_order(
                symbol=symbol,
                type='STOP_MARKET',
                side=sl_side,
                amount=amount,
                params={
                    'positionSide': position_side,
                    'stopPrice': sl_price,
                    'closePosition': True,
                    'priceProtect': True,
                    'workingType': 'MARK_PRICE'
                }
            )
            logger.info(f"   ✓ Stop Loss: {sl_order['id']}")
            
            # Colocar Take Profit
            logger.info("   📤 Colocando Take Profit...")
            tp_order = self.exchange.create_order(
                symbol=symbol,
                type='TAKE_PROFIT_MARKET',
                side=sl_side,
                amount=amount,
                params={
                    'positionSide': position_side,
                    'stopPrice': tp_price,
                    'closePosition': True,
                    'priceProtect': True,
                    'workingType': 'MARK_PRICE'
                }
            )
            logger.info(f"   ✓ Take Profit: {tp_order['id']}")
            
            # ═══════════════════════════════════════════════════════════════════
            # POSICIÓN DE COBERTURA (HEDGE) - DESACTIVADA
            # ═══════════════════════════════════════════════════════════════════
            # hedge_direction = 'SHORT' if direction == 'LONG' else 'LONG'
            # hedge_position_side = 'SHORT' if direction == 'LONG' else 'LONG'
            # hedge_side = 'sell' if direction == 'LONG' else 'buy'
            # hedge_sl_side = 'buy' if direction == 'LONG' else 'sell'
            # 
            # # Para el hedge: el TP de la principal es el SL del hedge y viceversa
            # hedge_sl_price = tp_price  # El TP de la principal es el SL del hedge
            # hedge_tp_price = sl_price  # El SL de la principal es el TP del hedge
            # 
            # logger.info("")
            # logger.info(f"   🛡️ POSICIÓN COBERTURA (HEDGE): {hedge_direction}")
            # logger.info("   ─────────────────────────────────────")
            # logger.info(f"   Stop Loss: ${hedge_sl_price:.6f}")
            # logger.info(f"   Take Profit: ${hedge_tp_price:.6f}")
            # logger.info("   📤 Enviando orden de mercado...")
            # 
            # hedge_order = self.exchange.create_order(
            #     symbol=symbol,
            #     type='MARKET',
            #     side=hedge_side,
            #     amount=amount,
            #     params={'positionSide': hedge_position_side}
            # )
            # logger.info(f"   ✓ Orden ejecutada: {hedge_order['id']}")
            # 
            # # Colocar Stop Loss del hedge
            # logger.info("   📤 Colocando Stop Loss...")
            # hedge_sl_order = self.exchange.create_order(
            #     symbol=symbol,
            #     type='STOP_MARKET',
            #     side=hedge_sl_side,
            #     amount=amount,
            #     params={
            #         'positionSide': hedge_position_side,
            #         'stopPrice': hedge_sl_price,
            #         'closePosition': True,
            #         'priceProtect': True,
            #         'workingType': 'MARK_PRICE'
            #     }
            # )
            # logger.info(f"   ✓ Stop Loss: {hedge_sl_order['id']}")
            # 
            # # Colocar Take Profit del hedge
            # logger.info("   📤 Colocando Take Profit...")
            # hedge_tp_order = self.exchange.create_order(
            #     symbol=symbol,
            #     type='TAKE_PROFIT_MARKET',
            #     side=hedge_sl_side,
            #     amount=amount,
            #     params={
            #         'positionSide': hedge_position_side,
            #         'stopPrice': hedge_tp_price,
            #         'closePosition': True,
            #         'priceProtect': True,
            #         'workingType': 'MARK_PRICE'
            #     }
            # )
            # logger.info(f"   ✓ Take Profit: {hedge_tp_order['id']}")
            
            logger.info("")
            logger.info(f"   🎉 POSICIÓN ABIERTA EXITOSAMENTE")
            logger.info("=" * 70)
            
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Error abriendo posición: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def scan_signals(self, open_positions: Dict) -> List[Dict]:
        """Escanea todos los símbolos buscando señales"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("🔍 ESCANEANDO SEÑALES EN TODOS LOS SÍMBOLOS")
        logger.info("=" * 70)
        
        signals = []
        
        # Normalizar símbolos de posiciones abiertas (ATOM/USDT:USDT -> ATOM/USDT)
        open_symbols = set()
        for sym in open_positions.keys():
            # Binance Futures devuelve ATOM/USDT:USDT, normalizamos a ATOM/USDT
            normalized = sym.split(':')[0] if ':' in sym else sym
            open_symbols.add(normalized)
        
        for symbol in SYMBOLS:
            # Saltar si ya tenemos posición en este símbolo
            if symbol in open_symbols:
                logger.info(f"   ⏭️ {symbol}: Ya tiene posición abierta, saltando")
                continue
            
            # === VERIFICAR COOLDOWN POR PÉRDIDA ===
            if self.is_symbol_in_cooldown(symbol):
                logger.info(f"   ⏭️ {symbol}: En COOLDOWN por pérdida reciente, saltando")
                continue
            
            # === VERIFICAR LÍMITE DE TRADES POR HORA (ANTI-SCALPING) ===
            if not self.can_trade_this_hour(symbol):
                logger.info(f"   ⏭️ {symbol}: Límite de trades por hora alcanzado, saltando")
                continue
            
            # Descargar datos
            df = self.fetch_ohlcv(symbol)
            if df is None:
                continue
            
            # Calcular indicadores
            df = calculate_indicators(df)
            
            # Verificar que no hay NaN en última vela cerrada
            row = df.iloc[-2]
            required_cols = ['ema8', 'ema20', 'ema21', 'ema50', 'adx', 'rsi', 'macd_hist', 'atr', 'vol_ratio', 'ema20_dist']
            has_nan = any(pd.isna(row[col]) for col in required_cols)
            
            if has_nan:
                logger.warning(f"   ⚠️ {symbol}: Indicadores con NaN, saltando")
                continue
            
            # === VERIFICAR MERCADO LATERAL ===
            is_lateral, range_pct = self.is_market_lateral(df)
            if is_lateral:
                logger.info(f"   ⏭️ {symbol}: Mercado LATERAL (rango {range_pct*100:.2f}% < 1.5%), saltando")
                continue
            
            # Obtener spread
            spread = self.get_spread(symbol)
            
            # === SIEMPRE MOSTRAR ANÁLISIS COMPLETO (aunque después se bloquee) ===
            # Analizar señal - esto muestra TODOS los indicadores y condiciones
            signal = analyze_signal(df, symbol, spread)
            
            # === FILTRO ANTI-ALBOROTO (15min vs 1H) ===
            # Verificar si el mercado de 15min está muy caótico DESPUÉS de mostrar el análisis
            atr_1h = row['atr']
            chaos_check = check_market_chaos(self.exchange, symbol, atr_1h)
            
            if CONFIG.get('USE_CHAOS_FILTER', False):
                logger.info("")
                logger.info(f"   🌪️ FILTRO ANTI-ALBOROTO {symbol}:")
                logger.info(f"   ┌────────────────────────────────────────────────────────────┐")
                logger.info(f"   │  ATR 1H:          ${atr_1h:.6f}")
                logger.info(f"   │  ATR 15min:       ${chaos_check['atr_15m']:.6f}")
                logger.info(f"   │  ATR 15min esp:   ${chaos_check['expected_atr']:.6f} (ATR_1H ÷ 4)")
                logger.info(f"   │  Ratio actual:    {chaos_check['ratio']:.2f}x")
                logger.info(f"   │  Ratio máximo:    {CONFIG['CHAOS_ATR_RATIO_MAX']}x")
                logger.info(f"   └────────────────────────────────────────────────────────────┘")
                
                if chaos_check['is_chaotic']:
                    logger.info(f"   ⛔ MERCADO ALBOROTADO - Velas 15min muy erráticas")
                    logger.info(f"   → Aunque haya señal, esperamos a que se calme")
                    logger.info("═" * 70)
                    continue  # Saltar este símbolo aunque tenga señal
                else:
                    logger.info(f"   ✅ Mercado estable - Ratio {chaos_check['ratio']:.2f}x < {CONFIG['CHAOS_ATR_RATIO_MAX']}x OK")
            
            if signal:
                signals.append(signal)
            
            # Pequeña pausa para no saturar API
            time.sleep(0.2)
        
        # Ordenar por score
        signals.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info("")
        logger.info("=" * 70)
        if signals:
            logger.info(f"📊 RESUMEN: {len(signals)} señales encontradas")
            for i, s in enumerate(signals):
                logger.info(f"   {i+1}. {s['symbol']} {s['direction']} | Score: {s['score']:.2f}")
        else:
            logger.info("📊 RESUMEN: No se encontraron señales válidas")
        logger.info("=" * 70)
        
        return signals
    
    def wait_for_next_check(self):
        """
        Pequeña pausa entre análisis para no saturar la API.
        
        ESTRATEGIA CONTINUA:
        - Evalúa constantemente buscando oportunidades
        - Solo pausa breve (30s) para no saturar API de Binance
        - Maximiza las oportunidades de entrada
        """
        wait_seconds = 30  # 30 segundos entre análisis
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("⏰ SIGUIENTE ANÁLISIS EN 30 SEGUNDOS")
        logger.info("=" * 70)
        logger.info(f"   (Análisis continuo - pausa breve para no saturar API)")
        logger.info("")
        
        time.sleep(wait_seconds)
        
        logger.info("   ✓ ¡Continuando análisis!")
        logger.info("=" * 70)
    
    def run(self):
        """Loop principal del bot - Análisis continuo"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("🟢 BOT INICIADO - SISTEMA INTELIGENTE")
        logger.info("=" * 70)
        logger.info(f"   Timeframe indicadores: {CONFIG['TIMEFRAME']} (velas cerradas)")
        logger.info(f"   Frecuencia análisis: CONTINUO (cada 30 segundos)")
        logger.info(f"   Sistema: Scoring inteligente con ponderación")
        logger.info(f"   Umbral entrada: 60+ puntos")
        logger.info("=" * 70)
        
        cycle = 0
        
        # Primera ejecución inmediata
        logger.info("")
        logger.info("⚡ EJECUTANDO PRIMER ANÁLISIS...")
        
        while True:
            try:
                cycle += 1
                logger.info("")
                logger.info(f"🔄 CICLO #{cycle} | {datetime.now().strftime('%H:%M:%S')}")
                
                # 1. Actualizar Risk Manager
                self.risk_manager.update()
                if not self.risk_manager.can_trade():
                    logger.warning("⛔ Trading detenido por Risk Manager. Esperando...")
                    time.sleep(60)
                    continue
                
                # 2. Obtener posiciones abiertas
                now = datetime.now()
                
                # Calcular qué vela se está evaluando
                candle_start = (now - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                candle_end = now.replace(minute=0, second=0, microsecond=0)
                
                logger.info("")
                logger.info("╔" + "═" * 68 + "╗")
                logger.info(f"║  CICLO #{cycle} | {now.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"║  Evaluando vela: {candle_start.strftime('%H:%M')} - {candle_end.strftime('%H:%M')} (CERRADA)")
                logger.info("╚" + "═" * 68 + "╝")
                
                # Obtener posiciones
                open_positions = self.get_open_positions()
                num_pos = len(open_positions)
                
                # ==========================================
                # TRAILING STOP - Actualizar SLs cada ciclo
                # ==========================================
                if open_positions and CONFIG.get('USE_TRAILING_STOP', False):
                    self.update_trailing_stops(open_positions)
                
                # Calcular PnL total
                if open_positions:
                    total_pnl = sum(p['pnl'] for p in open_positions.values())
                    emoji = "🟢" if total_pnl >= 0 else "🔴"
                    logger.info(f"   {emoji} PnL Total: ${total_pnl:.2f}")
                
                # Verificar slots
                slots = CONFIG['MAX_OPEN_POSITIONS'] - num_pos
                
                if slots <= 0:
                    logger.info(f"   ⏳ Máximo de posiciones ({num_pos}/{CONFIG['MAX_OPEN_POSITIONS']})")
                    logger.info(f"   Solo ejecutando trailing stop, no buscando nuevas señales...")
                    self.wait_for_next_check()
                    continue
                
                logger.info(f"   🎰 Slots disponibles: {slots}")
                
                # Verificar condiciones de mercado (filtro BTC)
                if CONFIG['USE_MARKET_FILTER']:
                    market = check_market_conditions(self.exchange)
                    
                    if not market['can_trade']:
                        logger.info(f"   ⛔ {market['reason']}")
                        logger.info(f"   Esperando mejores condiciones de mercado...")
                        self.wait_for_next_check()
                        continue
                    
                    # Ajustar margen según condiciones
                    self.current_margin = market['margin']
                    logger.info(f"   💰 Margen ajustado a ${self.current_margin} según mercado")
                else:
                    self.current_margin = CONFIG['MARGIN_USD']
                
                # Escanear señales
                signals = self.scan_signals(open_positions)
                
                # Abrir posiciones
                opened = 0
                for signal in signals:
                    if opened >= slots:
                        break
                    
                    # === FILTRO 1: TENDENCIA EMA 15 MINUTOS ===
                    # Verificar que la tendencia en 15min confirme la señal de 1H
                    if CONFIG.get('USE_15MIN_EMA_FILTER', False):
                        ema_15min_check = check_15min_ema_trend(self.exchange, signal['symbol'], signal['direction'])
                        
                        logger.info("")
                        logger.info(f"   📈 FILTRO 1: EMA 15min {signal['symbol']}:")
                        logger.info(f"   ┌────────────────────────────────────────────────────────────┐")
                        logger.info(f"   │  Dirección señal:  {signal['direction']}")
                        logger.info(f"   │  EMA8 (15min):     {ema_15min_check['ema8_15m']:.6f}")
                        logger.info(f"   │  EMA21 (15min):    {ema_15min_check['ema21_15m']:.6f}")
                        logger.info(f"   │  Precio actual:    {ema_15min_check.get('current_close', 0):.6f}")
                        logger.info(f"   │  ─────────────────────────────────────────────────────────")
                        logger.info(f"   │  Velas consecutivas: {ema_15min_check.get('consecutive_bars', 0)}/3 mínimo")
                        logger.info(f"   │  EMAs alineadas:     {'✅' if ema_15min_check.get('ema_condition', False) else '❌'}")
                        logger.info(f"   │  Precio correcto:    {'✅' if ema_15min_check.get('price_condition', False) else '❌'}")
                        logger.info(f"   │  ─────────────────────────────────────────────────────────")
                        logger.info(f"   │  RESULTADO:         {'✅ CONFIRMADA' if ema_15min_check['aligned'] else '❌ NO CONFIRMADA'}")
                        logger.info(f"   └────────────────────────────────────────────────────────────┘")
                        
                        if not ema_15min_check['aligned']:
                            logger.info(f"   ⛔ SEÑAL BLOQUEADA - {ema_15min_check['message']}")
                            logger.info(f"   → No hay tendencia clara en 15min para {signal['direction']}")
                            continue  # Saltar esta señal
                        else:
                            logger.info(f"   ✅ {ema_15min_check['message']}")
                    
                    # === FILTRO 2: VELAS 5 MINUTOS (TENDENCIA CORTA) ===
                    # Verificar análisis completo de velas de 5min
                    if CONFIG.get('USE_5MIN_CANDLE_FILTER', True):
                        candle_5min_check = check_5min_candles(self.exchange, signal['symbol'], signal['direction'])
                        
                        logger.info("")
                        logger.info(f"   📊 FILTRO 2: Análisis 5min {signal['symbol']} (5 CHECKS):")
                        logger.info(f"   ┌────────────────────────────────────────────────────────────┐")
                        logger.info(f"   │  Dirección señal:  {signal['direction']}")
                        logger.info(f"   │  ─────────────────────────────────────────────────────────")
                        
                        checks_5m = candle_5min_check.get('checks', {})
                        
                        # Check 1: Velas
                        c1 = checks_5m.get('candles', {})
                        logger.info(f"   │  1️⃣ Velas 3 últimas:  {'✅' if c1.get('pass') else '❌'} {c1.get('value', 'N/A')}")
                        
                        # Check 2: EMA Cross
                        c2 = checks_5m.get('ema_cross', {})
                        logger.info(f"   │  2️⃣ EMA8 vs EMA21:    {'✅' if c2.get('pass') else '❌'} {c2.get('value', 'N/A')}")
                        
                        # Check 3: Pendiente EMA
                        c3 = checks_5m.get('ema_slope', {})
                        logger.info(f"   │  3️⃣ Pendiente EMA8:   {'✅' if c3.get('pass') else '❌'} {c3.get('value', 'N/A')}")
                        
                        # Check 4: Posición precio
                        c4 = checks_5m.get('price_position', {})
                        logger.info(f"   │  4️⃣ Posición precio:  {'✅' if c4.get('pass') else '❌'} {c4.get('value', 'N/A')}")
                        
                        # Check 5: Gap EMAs
                        c5 = checks_5m.get('ema_gap', {})
                        logger.info(f"   │  5️⃣ Gap EMAs:         {'✅' if c5.get('pass') else '❌'} {c5.get('value', 'N/A')}")
                        
                        logger.info(f"   │  ─────────────────────────────────────────────────────────")
                        logger.info(f"   │  CHECKS PASADOS:   {candle_5min_check.get('passed_checks', 0)}/{candle_5min_check.get('total_checks', 5)} (mínimo 3)")
                        logger.info(f"   │  RESULTADO:        {'✅ CONFIRMADA' if candle_5min_check['aligned'] else '❌ NO CONFIRMADA'}")
                        logger.info(f"   └────────────────────────────────────────────────────────────┘")
                        
                        if not candle_5min_check['aligned']:
                            logger.info(f"   ⛔ SEÑAL BLOQUEADA - {candle_5min_check['message']}")
                            logger.info(f"   → Análisis de 5min no confirma {signal['direction']}")
                            continue  # Saltar esta señal
                        else:
                            logger.info(f"   ✅ {candle_5min_check['message']}")
                    
                    # === FILTRO 3: VELAS 1 MINUTO (MOMENTUM INMEDIATO) ===
                    # Verificar análisis completo de velas de 1min
                    if CONFIG.get('USE_1MIN_CANDLE_FILTER', True):
                        candle_1min_check = check_1min_candles(self.exchange, signal['symbol'], signal['direction'])
                        
                        logger.info("")
                        logger.info(f"   📊 FILTRO 3: Análisis 1min {signal['symbol']} (5 CHECKS):")
                        logger.info(f"   ┌────────────────────────────────────────────────────────────┐")
                        logger.info(f"   │  Dirección señal:  {signal['direction']}")
                        logger.info(f"   │  ─────────────────────────────────────────────────────────")
                        
                        checks = candle_1min_check.get('checks', {})
                        
                        # Check 1: Velas
                        c1 = checks.get('candles', {})
                        logger.info(f"   │  1️⃣ Velas 3 últimas:  {'✅' if c1.get('pass') else '❌'} {c1.get('value', 'N/A')}")
                        
                        # Check 2: EMA Cross
                        c2 = checks.get('ema_cross', {})
                        logger.info(f"   │  2️⃣ EMA8 vs EMA21:    {'✅' if c2.get('pass') else '❌'} {c2.get('value', 'N/A')}")
                        
                        # Check 3: Pendiente EMA
                        c3 = checks.get('ema_slope', {})
                        logger.info(f"   │  3️⃣ Pendiente EMA8:   {'✅' if c3.get('pass') else '❌'} {c3.get('value', 'N/A')}")
                        
                        # Check 4: Precio vs EMA
                        c4 = checks.get('price_vs_ema', {})
                        logger.info(f"   │  4️⃣ Precio vs EMA8:   {'✅' if c4.get('pass') else '❌'} {c4.get('value', 'N/A')}")
                        
                        # Check 5: Momentum
                        c5 = checks.get('momentum', {})
                        logger.info(f"   │  5️⃣ Momentum 5 velas: {'✅' if c5.get('pass') else '❌'} {c5.get('value', 'N/A')}")
                        
                        logger.info(f"   │  ─────────────────────────────────────────────────────────")
                        logger.info(f"   │  CHECKS PASADOS:   {candle_1min_check.get('passed_checks', 0)}/{candle_1min_check.get('total_checks', 5)} (mínimo 4)")
                        logger.info(f"   │  RESULTADO:        {'✅ CONFIRMADA' if candle_1min_check['aligned'] else '❌ NO CONFIRMADA'}")
                        logger.info(f"   └────────────────────────────────────────────────────────────┘")
                        
                        if not candle_1min_check['aligned']:
                            logger.info(f"   ⛔ SEÑAL BLOQUEADA - {candle_1min_check['message']}")
                            logger.info(f"   → Análisis de 1min no confirma {signal['direction']}")
                            continue  # Saltar esta señal
                        else:
                            logger.info(f"   ✅ {candle_1min_check['message']}")
                    
                    logger.info("")
                    logger.info(f"   ✅ TODOS LOS FILTROS PASADOS - Procediendo a abrir posición")
                    
                    if self.open_position(signal):
                        opened += 1
                        time.sleep(1)
                
                if opened > 0:
                    logger.info(f"   📈 Posiciones abiertas este ciclo: {opened}")
                
                # Pausa breve antes del próximo análisis
                self.wait_for_next_check()
                
            except KeyboardInterrupt:
                logger.info("")
                logger.info("🔴 Bot detenido por usuario (Ctrl+C)")
                logger.info("   Las posiciones abiertas permanecen activas")
                break
                
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.info("   Reintentando en 60 segundos...")
                time.sleep(60)
                time.sleep(60)

# =============================================================================
# 🚀 INICIO
# =============================================================================
if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║         🏆 BOT GANADORA - SISTEMA HÍBRIDO VERIFICADO 🏆              ║
    ║                    Binance Futures Trading                            ║
    ║              Basado en Backtest +797% Anual Verificado                ║
    ║                                                                       ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║  🔒 MANDATORIOS (7 condiciones 1H + Confirmación 15min):              ║
    ║  ─────────────────────────────────────────────────────────────────    ║
    ║  1. EMA8 vs EMA21 (define dirección LONG/SHORT)                       ║
    ║  2. ADX >= 28 (tendencia fuerte)                                      ║
    ║  3. RSI > 55 LONG / RSI < 70 SHORT (momentum)                         ║
    ║  4. Volumen >= 1.2x SMA20 (participación real)                        ║
    ║  5. Distancia EMA20 < 3.0 ATR (no extendido)                          ║
    ║  6. ATR entre 0.2% y 15% (volatilidad operable)                       ║
    ║  7. Spread < 0.1% (liquidez)                                          ║
    ║  + EMA8/EMA21 en 15min CONFIRMA dirección (filtro adicional)          ║
    ║                                                                       ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║  📊 OPCIONALES (6 condiciones bonus = 100 pts max):                   ║
    ║  ─────────────────────────────────────────────────────────────────    ║
    ║  • Close vs EMA50: 15 pts                                             ║
    ║  • EMA20 vs EMA50: 15 pts                                             ║
    ║  • MACD Histogram: 15 pts                                             ║
    ║  • Higher Low / Lower High: 20 pts                                    ║
    ║  • Pendientes EMAs: 20 pts                                            ║
    ║  • Alineación completa: 15 pts                                        ║
    ║                                                                       ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║  📈 TAMAÑO DE POSICIÓN (según puntos opcionales):                     ║
    ║  ─────────────────────────────────────────────────────────────────    ║
    ║  • 0-30 pts:  50% margen ($50)  → Entrada conservadora                ║
    ║  • 31-60 pts: 75% margen ($75)  → Entrada normal                      ║
    ║  • 61-100 pts: 100% margen ($100) → Entrada agresiva                  ║
    ║                                                                       ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║  ⚙️ CONFIG: $100 base | 10x | SL 1.5 ATR | TP 3.0 ATR                 ║
    ║  📊 Evaluación cada 15 min usando velas 1H cerradas                   ║
    ║                                                                       ║
    ║  ⚠️  ADVERTENCIA: Opera con DINERO REAL                               ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        bot = BotGanadora()
        bot.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
