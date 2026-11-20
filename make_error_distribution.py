# Розподіл похибок прогнозів продажу товарів у магазинах.
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def prepare_pairs(path):
    # %% ----------------------- перевірені прогнози + факт -----------------------
    for name in ['data_manifest.json', 'reference_manifest.json']:
        manifest = json.loads((path/name).read_text(encoding='utf-8'))
        for filename, entry in manifest.items():
            assert hashlib.sha256((path/filename).read_bytes()).hexdigest() == entry['sha256']

    prediction = pd.read_csv(path/'reference/predictions.csv.gz', float_precision='round_trip')
    truth = pd.read_csv(path/'data/evaluation.csv.gz', float_precision='round_trip')
    keys = ['window', 'origin', 'sample_id', 'product_id', 'shop_id']
    assert set(prediction['method']) == {'standard', 'project'}
    assert not prediction.duplicated(keys+['method']).any()
    pairs = prediction.pivot(index=keys, columns='method', values='forecast').reset_index()
    pairs.columns.name = None
    assert pairs[['standard', 'project']].notna().all().all()
    pairs = pairs.merge(truth, on='sample_id', validate='many_to_one')
    assert len(pairs)*2 == len(prediction)
    assert pairs['actual_qty'].notna().all()
    pairs = pairs[pairs['active_or']].copy()
    assert np.isfinite(pairs[['standard', 'project', 'actual_qty']]).all().all()
    assert pairs[['standard', 'project', 'actual_qty']].ge(0).all().all()
    actual = pairs['actual_qty'].replace(0, np.nan)
    for method in ['standard', 'project']:
        pairs['ape_'+method] = (pairs[method] - actual).abs() / actual * 100
    return pairs


def count_changes(difference):
    # Допуск виражений у тих самих одиницях, що й різниця похибок.
    tolerance = 1e-8
    better = int((difference < -tolerance).sum())
    worse = int((difference > tolerance).sum())
    equal = int((difference.abs() <= tolerance).sum())
    assert better + worse + equal == len(difference)
    return {'better': better, 'worse': worse, 'equal': equal}


def summarize(forecasts):
    # %% Інтервали APE й порівняння прогнозів для тих самих спостережень
    bins = [0, 15, 25, 50, 100, np.inf]
    labels = ['0–15%', '>15–25%', '>25–50%', '>50–100%', '>100%']
    groups, comparisons, zeros = [], [], []
    for window in ['all', '112d']:
        history = forecasts[forecasts['window'] == window]
        for period in ['both', *sorted(history['origin'].unique())]:
            current = history if period == 'both' else history[history['origin'] == period]
            positive = current[current['actual_qty'] > 0]
            zero = current[current['actual_qty'] == 0]
            for method in ['standard', 'project']:
                interval = pd.cut(positive['ape_'+method], bins, labels=labels, include_lowest=True)
                counts = interval.value_counts(sort=False)
                assert counts.sum() == len(positive)
                for label, count in counts.items():
                    groups.append(dict(window=window, period=period, method=method, error_bin=label,
                                       count=int(count), percent=count/len(positive)*100))

            changes = count_changes(positive['ape_project'] - positive['ape_standard'])
            comparisons.append(dict(window=window, period=period, positive_n=len(positive), **changes))

            # Для нульових продажів абсолютна похибка у штуках дорівнює прогнозу.
            changes = count_changes(zero['project'] - zero['standard'])
            zeros.append(dict(window=window, period=period, zero_n=len(zero), **changes,
                              standard_forecast=zero['standard'].sum(), project_forecast=zero['project'].sum()))
    return pd.DataFrame(groups), pd.DataFrame(comparisons), pd.DataFrame(zeros)


def draw_forecasts(axes, current, titles, x_ticks, y_ticks, x_limit, y_limit, max_sales):
    # Однакові межі осей і правила розміру маркерів роблять моделі зіставними.
    sizes = 7 + 180 * np.sqrt(current['actual_qty'] / max_sales)
    for ax, method, color in zip(axes, ['standard', 'project'], ['#2876bb', '#db7527']):
        ax.scatter(np.log1p(current['ape_'+method]), np.log1p(current[method]),
                   s=sizes, color=color, alpha=.35, edgecolors='none')
        for boundary in [15, 25, 50, 100]:
            ax.axvline(np.log1p(boundary), color='#758295', ls='--', lw=.8)
        ax.set_xticks(np.log1p(x_ticks), labels=[str(x) for x in x_ticks])
        ax.set_yticks(np.log1p(y_ticks), labels=[str(y) for y in y_ticks])
        ax.set_xlim(0, x_limit)
        ax.set_ylim(0, y_limit)
        ax.tick_params(axis='x', labelrotation=45, labelsize=9)
        ax.set_title(titles[method], color=color, fontsize=13, pad=14)
        ax.set_xlabel('Абсолютна відносна похибка, %', labelpad=12)
        ax.grid(axis='y', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Прогноз продажу, шт.')


def save_png(pairs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    positive = pairs[(pairs['window'] == 'all') & (pairs['actual_qty'] > 0)]
    changes = count_changes(positive['ape_project'] - positive['ape_standard'])
    titles = {'standard': 'Стандартне масштабування', 'project': 'Проєктне нормування'}
    x_ticks = [0, 15, 25, 50, 100, 300, 1000, 3000, 15000]
    y_ticks = [0, 1, 5, 10, 25, 50, 100, 200, 500]
    x_limit = np.log1p(positive[['ape_standard', 'ape_project']].max().max()) * 1.04
    y_limit = np.log1p(positive[['standard', 'project']].max().max()) * 1.05

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
    fig.patch.set_facecolor('#f6f8fb')
    draw_forecasts(axes, positive, titles, x_ticks, y_ticks, x_limit, y_limit, pairs['actual_qty'].max())
    fig.suptitle('Похибка прогнозу в окремих магазинах', fontsize=19, x=.07, ha='left', y=.97)
    fig.text(.07, .895, f'Повна історія, дві акції. Кількість прогнозів із продажами — {len(positive)}.'
             + ' Більша бульбашка означає більші фактичні продажі.', fontsize=11)
    fig.text(.07, .08, 'Обидві осі мають шкалу log(1+x). Позначки показують похибку у відсотках і прогноз у штуках.', fontsize=10, color='#465268')
    fig.text(.07, .035, f"З проєктним нормуванням похибка менша у {changes['better']} випадках, більша у {changes['worse']}, однакова у {changes['equal']}.", fontsize=11)
    fig.subplots_adjust(left=.07, right=.98, top=.8, bottom=.23, wspace=.10)
    fig.savefig(output/'error_distribution.png', dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Діагностика готових прогнозів без повторного навчання')
    parser.add_argument('--png', action='store_true', help='Додатково створити PNG; потрібен matplotlib')
    args = parser.parse_args()
    path = Path(__file__).resolve().parent
    output = path/'results'
    output.mkdir(exist_ok=True)
    pairs = prepare_pairs(path)
    groups, comparisons, zeros = summarize(pairs)
    groups.to_csv(output/'error_bins.csv', index=False)
    comparisons.to_csv(output/'error_pair_comparison.csv', index=False)
    zeros.to_csv(output/'error_zero_actual.csv', index=False)
    pairs.to_csv(output/'error_pairs.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})

    # На діаграмі APE показуємо спостереження з додатними продажами.
    columns = ['window', 'origin', 'sample_id', 'product_id', 'shop_id',
               'actual_qty', 'standard', 'project', 'ape_standard', 'ape_project']
    positive = pairs[pairs['actual_qty'].gt(0)]
    payload = dict(columns=columns, rows=positive[columns].values.tolist(),
                   bins=groups.to_dict('records'), comparisons=comparisons.to_dict('records'),
                   zeros=zeros.to_dict('records'), bubble_max_actual=float(pairs['actual_qty'].max()))
    template = (path/'assets/error_distribution_template.html').read_text(encoding='utf-8')
    html = template.replace('__DATA__', json.dumps(payload, ensure_ascii=False, allow_nan=False))
    (output/'error_distribution.html').write_text(html, encoding='utf-8')
    if args.png:
        save_png(pairs, output)
    checks = dict(positive_per_window=positive.groupby('window').size().to_dict(),
                  zero_per_window=pairs[pairs['actual_qty'].eq(0)].groupby('window').size().to_dict(),
                  paired_predictions_complete=True, bin_counts_sum_to_positive_n=True,
                  source_manifests_verified=True, equality_tolerance=1e-8)
    (output/'error_distribution_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(comparisons.to_string(index=False))
    print('Готово: results/error_distribution.html')


if __name__ == '__main__':
    main()
