"""Аудируемые суточные профили; без восстановления, интерполяции или подмены измерений."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'household_power_consumption.zip'
OUT = ROOT / 'figures'
MAIN_YEAR = 2008
COMPARE_YEARS = (2007, 2008, 2009)


def load_raw(path=DATA):
    """Загружает оригинальные строки без заполнения временной сетки."""
    df = pd.read_csv(path, sep=';', na_values=['?'], low_memory=False)
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    return df


def audit(df):
    """Аудит исходных строк, включая пропущенные значения и временные отметки."""
    ts = df['timestamp']
    counts = {}
    for year in COMPARE_YEARS:
        part = df.loc[ts.dt.year == year]
        expected = 527040 if year % 4 == 0 else 525600
        duplicates = int(part['timestamp'].dropna().duplicated().sum())
        distinct = int(part['timestamp'].nunique())
        # В выбранных годах нет пропусков timestamp и нет повторов.
        counts[str(year)] = dict(rows=int(len(part)), expected_minutes=expected,
                                 absent_timestamps=expected-distinct,
                                 duplicate_timestamps=duplicates,
                                 missing_power=int(part['Global_active_power'].isna().sum()),
                                 coverage_pct=round(100 * part['Global_active_power'].count() / expected, 5))
    return counts


def clean_year(df, year=MAIN_YEAR):
    """Отбрасывает отсутствующие и физически невозможные значения, не создает новых."""
    sub = df.loc[df['timestamp'].dt.year == year, ['timestamp','Global_active_power', 'Voltage','Global_intensity']].copy()
    before = len(sub)
    bad_time = sub['timestamp'].isna()
    sub = sub.loc[~bad_time]
    duplicate = sub['timestamp'].duplicated(keep='first')
    sub = sub.loc[~duplicate]
    missing_power = sub['Global_active_power'].isna()
    sub = sub.loc[~missing_power]
    # Мощность должна быть неотрицательной. Нулевое измерение может быть настоящим.
    impossible = ~np.isfinite(sub['Global_active_power']) | (sub['Global_active_power'] < 0)
    sub = sub.loc[~impossible].sort_values('timestamp')
    # Отметка для проверки, но НЕ автоматическое удаление настоящих пиков:
    q1, q3 = sub['Global_active_power'].quantile([.25,.75])
    statistical_high = int((sub['Global_active_power'] > q3 + 3 * (q3-q1)).sum())
    report = dict(rows_before=int(before), bad_timestamps=int(bad_time.sum()),
                  duplicate_timestamps=int(duplicate.sum()), missing_power=int(missing_power.sum()),
                  physically_invalid_power=int(impossible.sum()), retained_rows=int(len(sub)),
                  statistical_high_candidates_retained=statistical_high,
                  threshold_3iqr_kw=round(float(q3+3*(q3-q1)),3))
    return sub, report


def profiles(clean):
    """Два честных расчёта: среднее по имеющимся минутам и одинаковый вес дней.

    Дневные часы без измерений остаются отсутствующими; пустые дни не участвуют.
    """
    x=clean.copy()
    x['date']=x['timestamp'].dt.floor('D')
    x['hour']=x['timestamp'].dt.hour
    x['power']=x['Global_active_power']
    hourly_raw=x.groupby('hour')['power'].agg(['mean','count']).reindex(range(24))
    daily_hour=x.groupby(['date','hour'])['power'].agg(['mean','count']).reset_index()
    equal_day=daily_hour.groupby('hour')['mean'].mean().reindex(range(24))
    daily_counts=x.groupby('date').size()
    complete_dates=daily_counts.index[daily_counts.eq(1440)]
    complete=x.loc[x['date'].isin(complete_dates)]
    complete_hour=complete.groupby(['date','hour'])['power'].mean().groupby('hour').mean().reindex(range(24))
    table=pd.DataFrame({'all_available_kw':hourly_raw['mean'],
                        'equal_day_weight_kw':equal_day,
                        'only_complete_days_kw':complete_hour,
                        'minute_observations':hourly_raw['count'].astype(int)})
    table.index.name='hour'
    return table, int(len(complete_dates)), daily_counts


def run_analysis():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=load_raw()
    yearly_audit=audit(raw)
    clean,clean_report=clean_year(raw,MAIN_YEAR)
    table,complete_days,daily_counts=profiles(clean)
    table.to_csv(OUT/'hourly_profile.csv',float_format='%.6f')
    quality=pd.DataFrame([dict(year=int(y),**v) for y,v in yearly_audit.items()])
    quality.to_csv(OUT/'year_quality.csv',index=False)
    month_counts=clean.assign(month=clean.timestamp.dt.month).groupby('month')['Global_active_power'].agg(['size','mean']).reindex(range(1,13))
    month_counts.index.name='month'
    month_counts.to_csv(OUT/'monthly_coverage.csv')
    plt.style.use('seaborn-v0_8-whitegrid')
    fig,ax=plt.subplots(figsize=(10,5))
    for col, label in [('all_available_kw','Все доступные минуты'),('equal_day_weight_kw','Равный вес дней'),('only_complete_days_kw','Только дни без пропусков')]:
        ax.plot(table.index, table[col],marker='o',markersize=3,label=label)
    ax.set(xlabel='Час суток', ylabel='Средняя активная мощность, кВт',
           title='Суточный профиль, 2008 год: три способа отбора / усреднения',xticks=range(0,24,2))
    ax.legend();fig.tight_layout();fig.savefig(OUT/'daily_profiles.png',dpi=160);plt.close(fig)

    # Сезонные различия в одном и том же календарном году.
    ts=clean.timestamp
    month_hours=clean.groupby([ts.dt.month,ts.dt.hour])['Global_active_power'].mean().unstack().reindex(columns=range(24))
    fig,ax=plt.subplots(figsize=(10,5))
    for month,name in [(1,'Январь'),(4,'Апрель'),(7,'Июль'),(10,'Октябрь')]:
        ax.plot(range(24),month_hours.loc[month],label=name)
    ax.set(xlabel='Час суток',ylabel='Средняя активная мощность, кВт',title='Сезонные различия в 2008 году',xticks=range(0,24,2));ax.legend()
    fig.tight_layout();fig.savefig(OUT/'seasonal_profiles.png',dpi=160);plt.close(fig)

    # Проверка на другом годе без заполнения и без пересчёта исходных данных.
    fig,ax=plt.subplots(figsize=(10,5))
    years_profiles={}
    for year in COMPARE_YEARS:
        yclean,_=clean_year(raw,year)
        yp,_,_=profiles(yclean)
        years_profiles[year]=yp['all_available_kw']
        ax.plot(range(24),yp['all_available_kw'],label=f'{year} г.')
    ax.set(xlabel='Час суток',ylabel='Средняя активная мощность, кВт',title='Проверка устойчивости: 2007–2009 годы',xticks=range(0,24,2));ax.legend()
    fig.tight_layout();fig.savefig(OUT/'year_comparison.png',dpi=160);plt.close(fig)
    pd.DataFrame(years_profiles).rename_axis('hour').to_csv(OUT/'year_comparison.csv',float_format='%.6f')

    # Пропуски по месяцам: отсутствие значения не заменяется средним или нулем.
    raw_year=raw.loc[raw.timestamp.dt.year == MAIN_YEAR]
    missing_month=raw_year.groupby(raw_year.timestamp.dt.month)['Global_active_power'].agg(['size',lambda a:a.isna().sum()])
    missing_month.columns=['rows','missing_values']
    missing_month.index.name='month';missing_month.to_csv(OUT/'missing_by_month.csv')
    fig,ax=plt.subplots(figsize=(9,4))
    ax.bar(missing_month.index,missing_month['missing_values']);ax.set(xlabel='Месяц 2008 года',ylabel='Число отсутствующих измерений',title='Распределение пропусков в 2008 году',xticks=range(1,13))
    fig.tight_layout();fig.savefig(OUT/'missing_by_month.png',dpi=160);plt.close(fig)

    delta=(table['only_complete_days_kw']-table['all_available_kw']).abs()
    metrics={'main_year':MAIN_YEAR,'year_quality':yearly_audit,'cleaning':clean_report,
             'complete_days':complete_days,'days_in_year':366,
             'peak_hour':int(table['all_available_kw'].idxmax()),
             'min_hour':int(table['all_available_kw'].idxmin()),
             'peak_kw':round(float(table['all_available_kw'].max()),4),
             'minimum_kw':round(float(table['all_available_kw'].min()),4),
             'max_difference_complete_days_kw':round(float(delta.max()),6),
             'average_difference_complete_days_kw':round(float(delta.mean()),6),
             'number_of_synthetic_values':0}
    (OUT/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    return metrics,table



def analyze_low_consumption(clean, threshold_kw=0.25, min_coverage=0.95):
    """Описание низких дней без восстановления показаний и без вывода о местонахождении людей.

    Дни с недостаточным покрытием помечаются как непригодные для ранжирования;
    порог 0.25 кВт — диагностический, не нормативный.
    """
    x = clean.assign(date=clean.timestamp.dt.floor('D'))
    daily = x.groupby('date')['Global_active_power'].agg(['mean','median','max','count'])
    daily.index.name = 'date'
    daily['coverage_pct'] = daily['count'] / 1440 * 100
    daily['month'] = daily.index.month
    daily['weekday'] = daily.index.weekday
    daily['adequate_coverage'] = daily['coverage_pct'] >= min_coverage*100
    # Сезонно-недельный сравнительный ориентир, только по настоящим значениям.
    valid = daily[daily['adequate_coverage']]
    reference = valid.groupby(['month','weekday'])['mean'].median()
    daily['reference_kw'] = [reference.get((m,w), float('nan')) for m,w in zip(daily.month,daily.weekday)]
    daily['ratio_to_reference'] = daily['mean'] / daily['reference_kw']
    daily['low_absolute'] = daily['adequate_coverage'] & (daily['mean'] < threshold_kw)
    daily['low_relative'] = daily['adequate_coverage'] & (daily['ratio_to_reference'] < .65)
    # Находим все последовательности низких дней, учитывая календарную непрерывность.
    low_dates = daily.index[daily['low_absolute']]
    streaks=[]
    if len(low_dates):
        start=prev=low_dates[0]
        for day in low_dates[1:]:
            if (day-prev).days != 1:
                streaks.append((start,prev,(prev-start).days+1))
                start=day
            prev=day
        streaks.append((start,prev,(prev-start).days+1))
    streaks.sort(key=lambda t:(-t[2],t[0]))
    streak_table=pd.DataFrame([{'start':a.strftime('%Y-%m-%d'),'end':b.strftime('%Y-%m-%d'),'days':n,
        'mean_kw':round(float(daily.loc[a:b,'mean'].mean()),6),
        'minimum_coverage_pct':round(float(daily.loc[a:b,'coverage_pct'].min()),3)} for a,b,n in streaks])
    return daily,streak_table


def run_extended_analysis():
    """Все базовые рисунки + описательная диагностика дней с низкой нагрузкой."""
    metrics,profile = run_analysis()
    raw = load_raw()
    clean,_ = clean_year(raw,MAIN_YEAR)
    daily,streaks = analyze_low_consumption(clean)
    daily.to_csv(OUT/'daily_consumption.csv',float_format='%.6f')
    streaks.to_csv(OUT/'low_consumption_periods.csv',index=False)
    ranked = daily[daily.adequate_coverage].sort_values('ratio_to_reference').head(15)
    ranked.to_csv(OUT/'unusually_low_days.csv',float_format='%.6f')
    fig,ax=plt.subplots(figsize=(13,5))
    ax.plot(daily.index,daily['mean'],linewidth=1.1,label='Средняя измеренная мощность за день')
    ax.axhline(.25,color='tab:red',linestyle='--',linewidth=1,label='Диагностический порог 0,25 кВт')
    if not streaks.empty:
        top=streaks.iloc[0]
        ax.axvspan(pd.Timestamp(top['start']),pd.Timestamp(top['end']),color='tab:orange',alpha=.2,
                   label='Самый длинный период низкой нагрузки')
    ax.set(title='Среднесуточная мощность в 2008 году: возможные периоды низкой активности',
           xlabel='Дата',ylabel='Средняя активная мощность, кВт')
    ax.legend(loc='upper right');fig.tight_layout();fig.savefig(OUT/'daily_consumption_timeline.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5))
    focus=daily.loc['2008-07-15':'2008-09-15']
    ax.plot(focus.index,focus['mean'],marker='.',markersize=3)
    ax.axhline(.25,color='tab:red',linestyle='--',label='Порог 0,25 кВт')
    ax.set(title='Лето 2008: длительное снижение измеренной нагрузки',xlabel='Дата',ylabel='Средняя активная мощность, кВт')
    ax.legend();fig.tight_layout();fig.savefig(OUT/'summer_low_consumption.png',dpi=170);plt.close(fig)
    special_dates=['2008-01-01','2008-05-01','2008-07-14','2008-11-11',
                   '2008-12-24','2008-12-25','2008-12-26','2008-12-29','2008-12-31']
    observed=daily.loc[special_dates,['mean','count','coverage_pct','reference_kw','ratio_to_reference']].copy()
    observed.to_csv(OUT/'calendar_dates_comparison.csv',float_format='%.6f')
    metrics['low_consumption']={
        'daily_coverage_minimum_pct':95,
        'absolute_threshold_kw':0.25,
        'relative_threshold_ratio':0.65,
        'days_eligible':int(daily.adequate_coverage.sum()),
        'days_below_absolute_threshold':int(daily.low_absolute.sum()),
        'days_below_relative_threshold':int(daily.low_relative.sum()),
        'longest_low_period':streaks.iloc[0].to_dict() if len(streaks) else None,
        'lowest_mean_day':str(daily.loc[daily.adequate_coverage,'mean'].idxmin().date()),
        'lowest_mean_day_kw':round(float(daily.loc[daily.adequate_coverage,'mean'].min()),6),
        'example_calendar_dates':{str(day.date()):{'mean_kw':round(float(v['mean']),5),
          'ratio_to_reference':round(float(v['ratio_to_reference']),4)} for day,v in observed.iterrows()}
    }
    (OUT/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    return metrics,profile,daily,streaks

if __name__=='__main__':
    metrics,_,_,_=run_extended_analysis()
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
