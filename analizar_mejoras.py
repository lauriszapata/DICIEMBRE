import pandas as pd
import numpy as np
from datetime import datetime

# Cargar trades
df = pd.read_csv('trades_anual_2025.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])
df['month'] = df['entry_time'].dt.month
df['hour'] = df['entry_time'].dt.hour
df['day_of_week'] = df['entry_time'].dt.dayofweek
df['is_winner'] = df['pnl'] > 0

print('=' * 80)
print('ANÁLISIS PROFUNDO PARA MEJORAS')
print('=' * 80)

# 1. Análisis por símbolo
print('\n📊 RENTABILIDAD POR SÍMBOLO:')
print('-' * 80)
symbol_stats = df.groupby('symbol').agg({
    'pnl': ['sum', 'mean', 'count'],
    'is_winner': 'sum'
})
symbol_stats.columns = ['PnL Total', 'PnL Promedio', 'Trades', 'Ganadores']
symbol_stats['Win Rate %'] = (symbol_stats['Ganadores'] / symbol_stats['Trades'] * 100).round(1)
symbol_stats = symbol_stats.sort_values('PnL Total', ascending=False)
print(symbol_stats.to_string())

# 2. Análisis por dirección
print('\n\n📈 LONG vs SHORT:')
print('-' * 80)
for direction in ['LONG', 'SHORT']:
    dir_df = df[df['direction'] == direction]
    total = len(dir_df)
    winners = dir_df['is_winner'].sum()
    wr = winners / total * 100 if total > 0 else 0
    pnl = dir_df['pnl'].sum()
    avg = dir_df['pnl'].mean()
    print(f'{direction:5} | Trades: {total:3} | Win Rate: {wr:5.1f}% | PnL: ${pnl:8.2f} | Avg: ${avg:6.2f}')

# 3. Análisis por hora del día
print('\n\n⏰ RENTABILIDAD POR HORA DEL DÍA (Top 10):')
print('-' * 80)
hour_stats = df.groupby('hour').agg({
    'pnl': ['sum', 'mean', 'count'],
    'is_winner': 'sum'
})
hour_stats.columns = ['PnL Total', 'PnL Avg', 'Trades', 'Ganadores']
hour_stats['Win %'] = (hour_stats['Ganadores'] / hour_stats['Trades'] * 100).round(1)
hour_stats = hour_stats[hour_stats['Trades'] >= 5]
hour_stats = hour_stats.sort_values('PnL Total', ascending=False).head(10)
print(hour_stats.to_string())

print('\n\n⏰ PEORES HORAS (Bottom 5):')
print('-' * 80)
hour_stats_all = df.groupby('hour').agg({
    'pnl': ['sum', 'count'],
    'is_winner': 'sum'
})
hour_stats_all.columns = ['PnL Total', 'Trades', 'Ganadores']
hour_stats_all['Win %'] = (hour_stats_all['Ganadores'] / hour_stats_all['Trades'] * 100).round(1)
hour_stats_all = hour_stats_all[hour_stats_all['Trades'] >= 5]
hour_stats_worst = hour_stats_all.sort_values('PnL Total').head(5)
print(hour_stats_worst.to_string())

# 4. Análisis por día de la semana
print('\n\n📅 RENTABILIDAD POR DÍA DE LA SEMANA:')
print('-' * 80)
days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
day_stats = df.groupby('day_of_week').agg({
    'pnl': ['sum', 'mean', 'count'],
    'is_winner': 'sum'
})
day_stats.index = [days[i] for i in day_stats.index]
day_stats.columns = ['PnL Total', 'PnL Avg', 'Trades', 'Ganadores']
day_stats['Win %'] = (day_stats['Ganadores'] / day_stats['Trades'] * 100).round(1)
print(day_stats.to_string())

# 5. Análisis de duración
print('\n\n⏱️ RENTABILIDAD POR DURACIÓN:')
print('-' * 80)
df['duration_category'] = pd.cut(df['duration_hours'], 
                                  bins=[0, 5, 10, 20, 50, 1000],
                                  labels=['0-5h', '5-10h', '10-20h', '20-50h', '50+h'])
dur_stats = df.groupby('duration_category').agg({
    'pnl': ['sum', 'mean', 'count'],
    'is_winner': 'sum'
})
dur_stats.columns = ['PnL Total', 'PnL Avg', 'Trades', 'Ganadores']
dur_stats['Win %'] = (dur_stats['Ganadores'] / dur_stats['Trades'] * 100).round(1)
print(dur_stats.to_string())

# 6. Meses problemáticos
print('\n\n⚠️ ANÁLISIS DE MESES:')
print('-' * 80)
month_names = {1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio',
               7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}
month_stats = df.groupby('month').agg({
    'pnl': ['sum', 'count'],
    'is_winner': 'sum'
})
month_stats.index = [month_names[i] for i in month_stats.index]
month_stats.columns = ['PnL', 'Trades', 'Ganadores']
month_stats['Win %'] = (month_stats['Ganadores'] / month_stats['Trades'] * 100).round(1)
month_stats = month_stats.sort_values('PnL')
print(month_stats.to_string())

# 7. Análisis símbolos por mes problemático
print('\n\n🔍 SÍMBOLOS EN ENERO (mes peor):')
print('-' * 80)
enero = df[df['month'] == 1]
enero_symbols = enero.groupby('symbol').agg({
    'pnl': ['sum', 'count'],
    'is_winner': 'sum'
})
enero_symbols.columns = ['PnL', 'Trades', 'Ganadores']
enero_symbols['Win %'] = (enero_symbols['Ganadores'] / enero_symbols['Trades'] * 100).round(1)
enero_symbols = enero_symbols.sort_values('PnL')
print(enero_symbols.to_string())

print('\n\n🔍 SÍMBOLOS EN OCTUBRE (segundo peor):')
print('-' * 80)
oct = df[df['month'] == 10]
oct_symbols = oct.groupby('symbol').agg({
    'pnl': ['sum', 'count'],
    'is_winner': 'sum'
})
oct_symbols.columns = ['PnL', 'Trades', 'Ganadores']
oct_symbols['Win %'] = (oct_symbols['Ganadores'] / oct_symbols['Trades'] * 100).round(1)
oct_symbols = oct_symbols.sort_values('PnL')
print(oct_symbols.to_string())

print('\n' + '=' * 80)
