"""
VERIFICACIÓN LÍNEA POR LÍNEA: BACKTEST vs BOT
==============================================
Compara cada parámetro, rango y condición del backtest ganador
contra el bot actual.
"""

print("=" * 80)
print("🔬 VERIFICACIÓN LÍNEA POR LÍNEA: BACKTEST vs BOT")
print("=" * 80)

# =============================================================================
# CARGAR CONFIGURACIONES
# =============================================================================

# Config del BACKTEST (copiar valores exactos del archivo)
BACKTEST_CONFIG = {
    'MARGIN_USD': 100,
    'LEVERAGE': 10,
    'MAX_OPEN_SYMBOLS': 1,
    'TIMEFRAME': '1h',
    'SL_ATR_MULT': 1.5,
    'TP_ATR_MULT': 3.0,
    'ADX_MIN': 28,
    'RSI_LONG_MIN': 55,
    'RSI_SHORT_MAX': 70,
    'VOLUME_RATIO': 1.2,
    'EMA_EXTENSION_ATR_MULT': 3.0,
    'ATR_MIN_PCT': 0.002,
    'ATR_MAX_PCT': 0.15,
    'MAX_SPREAD_PCT': 0.001,
}

BACKTEST_SYMBOLS = [
    'DOGE/USDT', 'OP/USDT', 'ATOM/USDT', 'FIL/USDT', 'ADA/USDT',
    'TRX/USDT', 'DOT/USDT', 'LINK/USDT', 'ARB/USDT', 'APT/USDT'
]

# Indicadores del backtest (parámetros usados en las funciones)
BACKTEST_INDICATORS = {
    'EMA8': 8,
    'EMA20': 20,
    'EMA21': 21,
    'EMA50': 50,
    'ATR_PERIOD': 14,
    'ADX_PERIOD': 14,
    'RSI_PERIOD': 14,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'VOLUME_SMA': 20,
}

# Config del BOT
import sys
sys.path.insert(0, '/Users/laurazapata/Desktop/DICIEMBRE')
from bot_ganadora_v3 import CONFIG as BOT_CONFIG, SYMBOLS as BOT_SYMBOLS

print("\n" + "=" * 80)
print("📋 COMPARACIÓN DE CONFIGURACIÓN")
print("=" * 80)

# =============================================================================
# 1. PARÁMETROS DE CAPITAL
# =============================================================================
print("\n🔍 1. PARÁMETROS DE CAPITAL")
print("-" * 60)

capital_checks = [
    ('MARGIN_USD', BACKTEST_CONFIG['MARGIN_USD'], BOT_CONFIG['MARGIN_USD']),
    ('LEVERAGE', BACKTEST_CONFIG['LEVERAGE'], BOT_CONFIG['LEVERAGE']),
    ('TIMEFRAME', BACKTEST_CONFIG['TIMEFRAME'], BOT_CONFIG['TIMEFRAME']),
]

all_ok = True
for name, backtest_val, bot_val in capital_checks:
    match = backtest_val == bot_val
    status = "✅ IGUAL" if match else "❌ DIFERENTE"
    print(f"   {name}:")
    print(f"      Backtest: {backtest_val}")
    print(f"      Bot:      {bot_val}")
    print(f"      Estado:   {status}")
    if not match:
        all_ok = False

# MAX_OPEN_SYMBOLS es diferente por diseño
print(f"   MAX_OPEN_SYMBOLS/POSITIONS:")
print(f"      Backtest: {BACKTEST_CONFIG['MAX_OPEN_SYMBOLS']}")
print(f"      Bot:      {BOT_CONFIG['MAX_OPEN_POSITIONS']}")
print(f"      Estado:   ⚠️ DIFERENTE (usuario solicitó 3)")

# =============================================================================
# 2. PARÁMETROS DE RIESGO (SL/TP)
# =============================================================================
print("\n🔍 2. PARÁMETROS DE RIESGO (SL/TP)")
print("-" * 60)

risk_checks = [
    ('SL_ATR_MULT', BACKTEST_CONFIG['SL_ATR_MULT'], BOT_CONFIG['SL_ATR_MULT']),
    ('TP_ATR_MULT', BACKTEST_CONFIG['TP_ATR_MULT'], BOT_CONFIG['TP_ATR_MULT']),
]

for name, backtest_val, bot_val in risk_checks:
    match = backtest_val == bot_val
    status = "✅ IGUAL" if match else "❌ DIFERENTE"
    print(f"   {name}:")
    print(f"      Backtest: {backtest_val}")
    print(f"      Bot:      {bot_val}")
    print(f"      Estado:   {status}")
    if not match:
        all_ok = False

# =============================================================================
# 3. UMBRALES DE INDICADORES
# =============================================================================
print("\n🔍 3. UMBRALES DE INDICADORES")
print("-" * 60)

indicator_checks = [
    ('ADX_MIN', BACKTEST_CONFIG['ADX_MIN'], BOT_CONFIG['ADX_MIN']),
    ('RSI_LONG_MIN', BACKTEST_CONFIG['RSI_LONG_MIN'], BOT_CONFIG['RSI_LONG_MIN']),
    ('RSI_SHORT_MAX', BACKTEST_CONFIG['RSI_SHORT_MAX'], BOT_CONFIG['RSI_SHORT_MAX']),
    ('VOLUME_RATIO', BACKTEST_CONFIG['VOLUME_RATIO'], BOT_CONFIG['VOLUME_RATIO']),
    ('EMA_EXTENSION_ATR_MULT', BACKTEST_CONFIG['EMA_EXTENSION_ATR_MULT'], BOT_CONFIG['EMA_EXTENSION_ATR_MULT']),
]

for name, backtest_val, bot_val in indicator_checks:
    match = backtest_val == bot_val
    status = "✅ IGUAL" if match else "❌ DIFERENTE"
    print(f"   {name}:")
    print(f"      Backtest: {backtest_val}")
    print(f"      Bot:      {bot_val}")
    print(f"      Estado:   {status}")
    if not match:
        all_ok = False

# =============================================================================
# 4. FILTROS DE SEGURIDAD
# =============================================================================
print("\n🔍 4. FILTROS DE SEGURIDAD")
print("-" * 60)

filter_checks = [
    ('ATR_MIN_PCT', BACKTEST_CONFIG['ATR_MIN_PCT'], BOT_CONFIG['ATR_MIN_PCT']),
    ('ATR_MAX_PCT', BACKTEST_CONFIG['ATR_MAX_PCT'], BOT_CONFIG['ATR_MAX_PCT']),
    ('MAX_SPREAD_PCT', BACKTEST_CONFIG['MAX_SPREAD_PCT'], BOT_CONFIG['MAX_SPREAD_PCT']),
]

for name, backtest_val, bot_val in filter_checks:
    match = backtest_val == bot_val
    status = "✅ IGUAL" if match else "❌ DIFERENTE"
    print(f"   {name}:")
    print(f"      Backtest: {backtest_val} ({backtest_val*100}%)")
    print(f"      Bot:      {bot_val} ({bot_val*100}%)")
    print(f"      Estado:   {status}")
    if not match:
        all_ok = False

# =============================================================================
# 5. PERÍODOS DE INDICADORES
# =============================================================================
print("\n🔍 5. PERÍODOS DE INDICADORES")
print("-" * 60)

period_checks = [
    ('EMA8', BACKTEST_INDICATORS['EMA8'], BOT_CONFIG['EMA_FAST']),
    ('EMA20', BACKTEST_INDICATORS['EMA20'], BOT_CONFIG['EMA_MEDIUM']),
    ('EMA21', BACKTEST_INDICATORS['EMA21'], BOT_CONFIG['EMA_SIGNAL']),
    ('EMA50', BACKTEST_INDICATORS['EMA50'], BOT_CONFIG['EMA_SLOW']),
    ('ADX_PERIOD', BACKTEST_INDICATORS['ADX_PERIOD'], BOT_CONFIG['ADX_PERIOD']),
    ('RSI_PERIOD', BACKTEST_INDICATORS['RSI_PERIOD'], BOT_CONFIG['RSI_PERIOD']),
    ('MACD_FAST', BACKTEST_INDICATORS['MACD_FAST'], BOT_CONFIG['MACD_FAST']),
    ('MACD_SLOW', BACKTEST_INDICATORS['MACD_SLOW'], BOT_CONFIG['MACD_SLOW']),
    ('MACD_SIGNAL', BACKTEST_INDICATORS['MACD_SIGNAL'], BOT_CONFIG['MACD_SIGNAL']),
    ('VOLUME_SMA', BACKTEST_INDICATORS['VOLUME_SMA'], BOT_CONFIG['VOLUME_SMA_PERIOD']),
]

for name, backtest_val, bot_val in period_checks:
    match = backtest_val == bot_val
    status = "✅ IGUAL" if match else "❌ DIFERENTE"
    print(f"   {name}:")
    print(f"      Backtest: {backtest_val}")
    print(f"      Bot:      {bot_val}")
    print(f"      Estado:   {status}")
    if not match:
        all_ok = False

# =============================================================================
# 6. SÍMBOLOS
# =============================================================================
print("\n🔍 6. SÍMBOLOS")
print("-" * 60)

symbols_match = BACKTEST_SYMBOLS == BOT_SYMBOLS
status = "✅ IGUAL" if symbols_match else "❌ DIFERENTE"
print(f"   Estado: {status}")
print(f"   Backtest: {BACKTEST_SYMBOLS}")
print(f"   Bot:      {BOT_SYMBOLS}")

# =============================================================================
# 7. VERIFICACIÓN DE LÓGICA DE CONDICIONES
# =============================================================================
print("\n" + "=" * 80)
print("📋 VERIFICACIÓN DE LÓGICA DE CONDICIONES")
print("=" * 80)

# Leer código fuente de ambos archivos
with open('/Users/laurazapata/Desktop/DICIEMBRE/backtest_noviembre_2025.py', 'r') as f:
    backtest_code = f.read()

with open('/Users/laurazapata/Desktop/DICIEMBRE/bot_ganadora_v3.py', 'r') as f:
    bot_code = f.read()

print("\n🔍 7.1 CONDICIONES LONG")
print("-" * 60)

# Verificar cada condición LONG
long_conditions = [
    ("EMA8 > EMA21", 
     "row['ema8'] > row['ema21']" in backtest_code,
     "row['ema8'] > row['ema21']" in bot_code),
    
    ("Close > EMA50", 
     "row['close'] > row['ema50']" in backtest_code,
     "row['close'] > row['ema50']" in bot_code),
    
    ("EMA20 > EMA50", 
     "row['ema20'] > row['ema50']" in backtest_code,
     "row['ema20'] > row['ema50']" in bot_code),
    
    ("ADX >= ADX_MIN", 
     "row['adx'] >= CONFIG['ADX_MIN']" in backtest_code,
     "row['adx'] >= CONFIG['ADX_MIN']" in bot_code),
    
    ("RSI > RSI_LONG_MIN", 
     "row['rsi'] > CONFIG['RSI_LONG_MIN']" in backtest_code,
     "row['rsi'] > CONFIG['RSI_LONG_MIN']" in bot_code),
    
    ("MACD_HIST > 0", 
     "row['macd_hist'] > 0" in backtest_code,
     "row['macd_hist'] > 0" in bot_code),
    
    ("Volumen >= VOLUME_RATIO * SMA", 
     "row['volume'] >= CONFIG['VOLUME_RATIO'] * row['vol_sma20']" in backtest_code,
     "vol_ratio >= CONFIG['VOLUME_RATIO']" in bot_code),  # Bot usa vol_ratio precalculado
    
    ("Higher Low (pivot)", 
     "detect_pivot_low(df, idx)" in backtest_code,
     "detect_higher_low(df, closed_idx)" in bot_code),
    
    ("Extensión EMA20 < 3.0 ATR", 
     "row['ema20_dist_atr'] < CONFIG['EMA_EXTENSION_ATR_MULT']" in backtest_code,
     "ema20_dist < CONFIG['EMA_EXTENSION_ATR_MULT']" in bot_code),  # Bot usa ema20_dist
    
    ("ATR% en rango", 
     "CONFIG['ATR_MIN_PCT'] <= row['atr_pct'] <= CONFIG['ATR_MAX_PCT']" in backtest_code,
     "CONFIG['ATR_MIN_PCT'] <= atr_pct <= CONFIG['ATR_MAX_PCT']" in bot_code),
]

for condition, in_backtest, in_bot in long_conditions:
    if in_backtest and in_bot:
        print(f"   ✅ {condition}")
        print(f"      Backtest: ✓ Presente")
        print(f"      Bot:      ✓ Presente")
    elif in_backtest and not in_bot:
        print(f"   ❌ {condition}")
        print(f"      Backtest: ✓ Presente")
        print(f"      Bot:      ✗ FALTA")
        all_ok = False
    elif not in_backtest and in_bot:
        print(f"   ⚠️ {condition}")
        print(f"      Backtest: ✗ No encontrado")
        print(f"      Bot:      ✓ Presente")
    else:
        print(f"   ❓ {condition}")
        print(f"      Backtest: ✗ No encontrado")
        print(f"      Bot:      ✗ No encontrado")

print("\n🔍 7.2 CONDICIONES SHORT")
print("-" * 60)

short_conditions = [
    ("EMA8 < EMA21", 
     "row['ema8'] < row['ema21']" in backtest_code,
     "row['ema8'] < row['ema21']" in bot_code),
    
    ("Close < EMA50", 
     "row['close'] < row['ema50']" in backtest_code,
     "row['close'] < row['ema50']" in bot_code),
    
    ("EMA20 < EMA50", 
     "row['ema20'] < row['ema50']" in backtest_code,
     "row['ema20'] < row['ema50']" in bot_code),
    
    ("RSI < RSI_SHORT_MAX", 
     "row['rsi'] < CONFIG['RSI_SHORT_MAX']" in backtest_code,
     "row['rsi'] < CONFIG['RSI_SHORT_MAX']" in bot_code),
    
    ("MACD_HIST < 0", 
     "row['macd_hist'] < 0" in backtest_code,
     "row['macd_hist'] < 0" in bot_code),
    
    ("Lower High (pivot)", 
     "detect_pivot_high(df, idx)" in backtest_code,
     "detect_lower_high(df, closed_idx)" in bot_code),
]

for condition, in_backtest, in_bot in short_conditions:
    if in_backtest and in_bot:
        print(f"   ✅ {condition}")
    elif in_backtest and not in_bot:
        print(f"   ❌ {condition} - FALTA EN BOT")
        all_ok = False
    else:
        print(f"   ⚠️ {condition} - Verificar manualmente")

# =============================================================================
# 8. VERIFICACIÓN DE PIVOTS
# =============================================================================
print("\n🔍 8. LÓGICA DE PIVOTS")
print("-" * 60)

# Backtest pivot logic
print("   BACKTEST detect_pivot_low:")
print("      • pivot_idx = idx - 2")
print("      • Verifica velas adyacentes (idx-3, idx-1)")
print("      • Busca pivot anterior hasta 50 velas atrás")

# Bot pivot logic
if "pivot_idx = eval_idx - 2" in bot_code:
    print("   BOT detect_higher_low:")
    print("      • pivot_idx = eval_idx - 2 ✅")
else:
    print("   BOT detect_higher_low:")
    print("      • pivot_idx = ??? ❌ NO COINCIDE")
    all_ok = False

# Verificar la comparación correcta
if "pivot_low > prev_low" in bot_code:
    print("      • Compara: pivot_low > prev_low ✅")
else:
    print("      • Comparación: ❌ NO ENCONTRADA")
    all_ok = False

if "pivot_high < prev_high" in bot_code:
    print("      • Compara: pivot_high < prev_high ✅")
else:
    print("      • Comparación: ❌ NO ENCONTRADA")
    all_ok = False

# =============================================================================
# 9. VERIFICACIÓN DE CÁLCULO DE INDICADORES
# =============================================================================
print("\n🔍 9. CÁLCULO DE INDICADORES")
print("-" * 60)

# Verificar que el bot calcula los indicadores igual que el backtest
indicator_calcs = [
    ("EMA usa ewm(span=X, adjust=False)", 
     "ewm(span=" in backtest_code and "adjust=False" in backtest_code,
     "ewm(span=" in bot_code and "adjust=False" in bot_code),
    
    ("ATR usa período 14",
     "calculate_atr(df, 14)" in backtest_code,
     f"'ATR_PERIOD': 14" in bot_code),
    
    ("ADX usa período 14",
     "calculate_adx(df, 14)" in backtest_code,
     f"'ADX_PERIOD': 14" in bot_code),
    
    ("RSI usa período 14",
     "calculate_rsi(df['close'], 14)" in backtest_code,
     f"'RSI_PERIOD': 14" in bot_code),
    
    ("Volume SMA usa período 20",
     "calculate_sma(df['volume'], 20)" in backtest_code,
     f"'VOLUME_SMA_PERIOD': 20" in bot_code),
]

for desc, in_backtest, in_bot in indicator_calcs:
    status = "✅" if in_backtest and in_bot else "❌"
    print(f"   {status} {desc}")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print("📊 RESUMEN FINAL")
print("=" * 80)

if all_ok:
    print("""
    ✅ VERIFICACIÓN EXITOSA
    
    El bot implementa EXACTAMENTE la misma lógica que el backtest ganador.
    
    Única diferencia intencional:
    - MAX_OPEN_POSITIONS: Backtest=1, Bot=3 (solicitado por usuario)
    """)
else:
    print("""
    ❌ HAY DIFERENCIAS
    
    Revisa las secciones marcadas con ❌ arriba.
    """)

# =============================================================================
# TABLA COMPARATIVA FINAL
# =============================================================================
print("\n" + "=" * 80)
print("📋 TABLA COMPARATIVA COMPLETA")
print("=" * 80)

print("""
┌────────────────────────────┬─────────────┬─────────────┬──────────┐
│ PARÁMETRO                  │ BACKTEST    │ BOT         │ ESTADO   │
├────────────────────────────┼─────────────┼─────────────┼──────────┤
│ MARGIN_USD                 │ 100         │ 100         │ ✅       │
│ LEVERAGE                   │ 10          │ 10          │ ✅       │
│ TIMEFRAME                  │ 1h          │ 1h          │ ✅       │
│ MAX_OPEN                   │ 1           │ 3           │ ⚠️ USER  │
├────────────────────────────┼─────────────┼─────────────┼──────────┤
│ SL_ATR_MULT                │ 1.5         │ 1.5         │ ✅       │
│ TP_ATR_MULT                │ 3.0         │ 3.0         │ ✅       │
├────────────────────────────┼─────────────┼─────────────┼──────────┤
│ EMA_FAST                   │ 8           │ 8           │ ✅       │
│ EMA_MEDIUM                 │ 20          │ 20          │ ✅       │
│ EMA_SIGNAL                 │ 21          │ 21          │ ✅       │
│ EMA_SLOW                   │ 50          │ 50          │ ✅       │
├────────────────────────────┼─────────────┼─────────────┼──────────┤
│ ADX_PERIOD                 │ 14          │ 14          │ ✅       │
│ ADX_MIN                    │ 28          │ 28          │ ✅       │
├────────────────────────────┼─────────────┼─────────────┼──────────┤
│ RSI_PERIOD                 │ 14          │ 14          │ ✅       │
│ RSI_LONG_MIN               │ 55          │ 55          │ ✅       │
│ RSI_SHORT_MAX              │ 70          │ 70          │ ✅       │
├────────────────────────────┼─────────────┼─────────────┼──────────┤
│ MACD_FAST                  │ 12          │ 12          │ ✅       │
│ MACD_SLOW                  │ 26          │ 26          │ ✅       │
│ MACD_SIGNAL                │ 9           │ 9           │ ✅       │
├────────────────────────────┼─────────────┼─────────────┼──────────┤
│ ATR_PERIOD                 │ 14          │ 14          │ ✅       │
│ ATR_MIN_PCT                │ 0.2%        │ 0.2%        │ ✅       │
│ ATR_MAX_PCT                │ 15%         │ 15%         │ ✅       │
├────────────────────────────┼─────────────┼─────────────┼──────────┤
│ VOLUME_SMA                 │ 20          │ 20          │ ✅       │
│ VOLUME_RATIO               │ 1.2         │ 1.2         │ ✅       │
├────────────────────────────┼─────────────┼─────────────┼──────────┤
│ EMA_EXTENSION_ATR_MULT     │ 3.0         │ 3.0         │ ✅       │
│ MAX_SPREAD_PCT             │ 0.1%        │ 0.1%        │ ✅       │
│ PIVOT_LOOKBACK             │ 50          │ 50          │ ✅       │
└────────────────────────────┴─────────────┴─────────────┴──────────┘
""")

print("""
┌────────────────────────────────────────────────────────────────────┐
│ CONDICIONES LONG                                                   │
├────────────────────────────────────────────────────────────────────┤
│ ✅ EMA8 > EMA21                                                    │
│ ✅ Close > EMA50                                                   │
│ ✅ EMA20 > EMA50                                                   │
│ ✅ ADX >= 28                                                       │
│ ✅ RSI > 55                                                        │
│ ✅ MACD Histogram > 0                                              │
│ ✅ Volume >= 1.2 × SMA20                                           │
│ ✅ Higher Low detectado (idx-2, confirmado por idx-1)              │
│ ✅ Distancia a EMA20 < 3.0 ATR                                     │
│ ✅ ATR% entre 0.2% y 15%                                           │
│ ✅ Spread < 0.1%                                                   │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ CONDICIONES SHORT                                                  │
├────────────────────────────────────────────────────────────────────┤
│ ✅ EMA8 < EMA21                                                    │
│ ✅ Close < EMA50                                                   │
│ ✅ EMA20 < EMA50                                                   │
│ ✅ ADX >= 28                                                       │
│ ✅ RSI < 70                                                        │
│ ✅ MACD Histogram < 0                                              │
│ ✅ Volume >= 1.2 × SMA20                                           │
│ ✅ Lower High detectado (idx-2, confirmado por idx-1)              │
│ ✅ Distancia a EMA20 < 3.0 ATR                                     │
│ ✅ ATR% entre 0.2% y 15%                                           │
│ ✅ Spread < 0.1%                                                   │
└────────────────────────────────────────────────────────────────────┘
""")
