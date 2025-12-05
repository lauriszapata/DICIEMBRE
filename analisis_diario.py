"""
ANÁLISIS DÍA A DÍA - BACKTEST NOVIEMBRE 2025
==============================================
Genera reporte detallado día por día con trades, ganadores, perdedores y PnL
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Cargar trades del backtest verificado
trades_file = '/Users/laurazapata/Desktop/DICIEMBRE/trades_verificado_nov_2025.csv'
trades_df = pd.DataFrame()

try:
    trades_df = pd.read_csv(trades_file)
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    trades_df['entry_date'] = trades_df['entry_time'].dt.date
except Exception as e:
    print(f"Error cargando archivo: {e}")
    exit(1)

print("=" * 100)
print("ANÁLISIS DÍA A DÍA - BACKTEST NOVIEMBRE 2025")
print("=" * 100)
print()

# Análisis por día
daily_stats = []

for date in sorted(trades_df['entry_date'].unique()):
    day_trades = trades_df[trades_df['entry_date'] == date]
    
    total = len(day_trades)
    winners = len(day_trades[day_trades['pnl_net'] > 0])
    losers = len(day_trades[day_trades['pnl_net'] <= 0])
    pnl_day = day_trades['pnl_net'].sum()
    win_rate = (winners / total * 100) if total > 0 else 0
    
    # Desglose por dirección
    longs = len(day_trades[day_trades['direction'] == 'LONG'])
    shorts = len(day_trades[day_trades['direction'] == 'SHORT'])
    
    daily_stats.append({
        'date': date,
        'total_trades': total,
        'winners': winners,
        'losers': losers,
        'pnl': pnl_day,
        'win_rate': win_rate,
        'longs': longs,
        'shorts': shorts,
        'avg_pnl': pnl_day / total if total > 0 else 0
    })

daily_df = pd.DataFrame(daily_stats)

# Calcular PnL acumulado
daily_df['cumulative_pnl'] = daily_df['pnl'].cumsum()

print(f"📅 RENDIMIENTO DIARIO - NOVIEMBRE 2025")
print("=" * 100)
print()

# Mostrar tabla día por día
print(f"{'Fecha':<12} {'Trades':<8} {'✅ Win':<8} {'❌ Loss':<8} {'Win%':<8} {'L/S':<8} {'PnL Día':<12} {'PnL Acum':<12}")
print("-" * 100)

for _, row in daily_df.iterrows():
    date_str = str(row['date'])
    trades_str = f"{int(row['total_trades'])}"
    win_str = f"{int(row['winners'])}"
    loss_str = f"{int(row['losers'])}"
    winrate_str = f"{row['win_rate']:.0f}%"
    direction_str = f"{int(row['longs'])}/{int(row['shorts'])}"
    pnl_str = f"${row['pnl']:+.2f}"
    cum_pnl_str = f"${row['cumulative_pnl']:+.2f}"
    
    # Color emoji para PnL del día
    emoji = "🟢" if row['pnl'] > 0 else "🔴" if row['pnl'] < 0 else "⚪"
    
    print(f"{date_str:<12} {trades_str:<8} {win_str:<8} {loss_str:<8} {winrate_str:<8} {direction_str:<8} {emoji} {pnl_str:<10} {cum_pnl_str:<12}")

print()
print("=" * 100)

# Estadísticas generales
print()
print("📊 RESUMEN MENSUAL")
print("-" * 100)

total_days = len(daily_df)
days_with_trades = len(daily_df[daily_df['total_trades'] > 0])
green_days = len(daily_df[daily_df['pnl'] > 0])
red_days = len(daily_df[daily_df['pnl'] < 0])
neutral_days = len(daily_df[daily_df['pnl'] == 0])

best_day = daily_df.loc[daily_df['pnl'].idxmax()]
worst_day = daily_df.loc[daily_df['pnl'].idxmin()]
most_active_day = daily_df.loc[daily_df['total_trades'].idxmax()]

print(f"""
Días Operados:           {days_with_trades}
Días Verdes (ganadores): {green_days} ({green_days/total_days*100:.1f}%)
Días Rojos (perdedores): {red_days} ({red_days/total_days*100:.1f}%)
Días Neutrales:          {neutral_days}

Mejor Día:               {best_day['date']} con ${best_day['pnl']:.2f} ({int(best_day['total_trades'])} trades)
Peor Día:                {worst_day['date']} con ${worst_day['pnl']:.2f} ({int(worst_day['total_trades'])} trades)
Día Más Activo:          {most_active_day['date']} con {int(most_active_day['total_trades'])} trades

PnL Promedio/Día:        ${daily_df['pnl'].mean():.2f}
Trades Promedio/Día:     {daily_df['total_trades'].mean():.1f}
""")

print("=" * 100)

# Estadísticas por semana
print()
print("📅 RENDIMIENTO SEMANAL")
print("-" * 100)
print()

daily_df['week'] = pd.to_datetime(daily_df['date']).dt.isocalendar().week
weekly_stats = daily_df.groupby('week').agg({
    'total_trades': 'sum',
    'winners': 'sum',
    'losers': 'sum',
    'pnl': 'sum'
}).reset_index()

weekly_stats['win_rate'] = (weekly_stats['winners'] / weekly_stats['total_trades'] * 100).round(1)

for _, week in weekly_stats.iterrows():
    week_num = int(week['week'])
    week_trades = daily_df[daily_df['week'] == week_num]
    start_date = week_trades['date'].min()
    end_date = week_trades['date'].max()
    
    emoji = "🟢" if week['pnl'] > 0 else "🔴"
    
    print(f"Semana {week_num} ({start_date} - {end_date})")
    print(f"  {emoji} Trades: {int(week['total_trades'])} | Ganadores: {int(week['winners'])} | Perdedores: {int(week['losers'])} | Win Rate: {week['win_rate']:.1f}%")
    print(f"  💰 PnL: ${week['pnl']:.2f}")
    print()

print("=" * 100)

# Top días por PnL
print()
print("🏆 TOP 5 MEJORES DÍAS")
print("-" * 100)

top_5_days = daily_df.nlargest(5, 'pnl')
for i, (_, day) in enumerate(top_5_days.iterrows(), 1):
    print(f"{i}. {day['date']} - ${day['pnl']:.2f} ({int(day['total_trades'])} trades, {int(day['winners'])} ganadores)")

print()
print("⚠️  TOP 5 PEORES DÍAS")
print("-" * 100)

bottom_5_days = daily_df.nsmallest(5, 'pnl')
for i, (_, day) in enumerate(bottom_5_days.iterrows(), 1):
    print(f"{i}. {day['date']} - ${day['pnl']:.2f} ({int(day['total_trades'])} trades, {int(day['losers'])} perdedores)")

print()
print("=" * 100)

# Guardar reporte CSV
daily_df.to_csv('/Users/laurazapata/Desktop/DICIEMBRE/analisis_diario_nov_2025.csv', index=False)
print()
print("📁 Análisis guardado en: analisis_diario_nov_2025.csv")
print()

# Crear visualización ASCII de PnL acumulado
print()
print("📈 GRÁFICO PnL ACUMULADO (Noviembre 2025)")
print("=" * 100)
print()

max_pnl = daily_df['cumulative_pnl'].max()
min_pnl = daily_df['cumulative_pnl'].min()
pnl_range = max_pnl - min_pnl

if pnl_range > 0:
    for _, row in daily_df.iterrows():
        normalized = int((row['cumulative_pnl'] - min_pnl) / pnl_range * 50)
        bar = "█" * normalized
        print(f"{str(row['date']):<12} ${row['cumulative_pnl']:>7.2f} {bar}")

print()
print("=" * 100)

# Análisis de rachas
print()
print("🔥 ANÁLISIS DE RACHAS")
print("-" * 100)

# Calcular rachas de días ganadores/perdedores
daily_df['is_winner'] = daily_df['pnl'] > 0
current_streak = 1
max_win_streak = 1
max_loss_streak = 1
current_streak_type = daily_df['is_winner'].iloc[0]

for i in range(1, len(daily_df)):
    if daily_df['is_winner'].iloc[i] == current_streak_type:
        current_streak += 1
    else:
        if current_streak_type:
            max_win_streak = max(max_win_streak, current_streak)
        else:
            max_loss_streak = max(max_loss_streak, current_streak)
        current_streak = 1
        current_streak_type = daily_df['is_winner'].iloc[i]

# Actualizar última racha
if current_streak_type:
    max_win_streak = max(max_win_streak, current_streak)
else:
    max_loss_streak = max(max_loss_streak, current_streak)

print(f"""
Racha más larga de días ganadores: {max_win_streak} días consecutivos
Racha más larga de días perdedores: {max_loss_streak} días consecutivos
""")

print("=" * 100)
