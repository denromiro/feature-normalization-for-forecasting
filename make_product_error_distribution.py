# Розподіл похибок сумарного прогнозу для кожного товару й акції.
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from make_error_distribution import prepare_pairs, summarize, draw_forecasts


def prepare_products(path):
    # %% Сумуємо прогнози та продажі по магазинах, відібраних для оцінки
    pairs = prepare_pairs(path)
    keys = ['window', 'origin', 'product_id']
    products = pairs.groupby(keys)[['standard', 'project', 'actual_qty']].sum().reset_index()
    actual = products['actual_qty'].replace(0, np.nan)
    for method in ['standard', 'project']:
        products['ape_'+method] = (products[method] - actual).abs() / actual * 100

    # Звіряємо з основним звітом прогноз, продажі та APE кожного товару.
    reference = pd.read_csv(path/'results/product_forecast_fact.csv', float_precision='round_trip')
    max_difference = 0.
    for method in ['standard', 'project']:
        expected = reference[reference['method'].eq(method)]
        joined = products.merge(expected, on=keys, validate='one_to_one', suffixes=('', '_report'))
        assert len(joined) == len(products) == len(expected)
        for actual, expected_name in [(method, 'forecast'), ('actual_qty', 'actual_qty_report'),
                                      ('ape_'+method, 'ape')]:
            difference = (joined[actual]-joined[expected_name]).abs().max()
            max_difference = max(max_difference, float(difference))
            assert np.allclose(joined[actual], joined[expected_name], rtol=0, atol=1e-8, equal_nan=True)
    return products, max_difference


def check_mape(products, path):
    # %% ----------------------- MAPE і кількість великих похибок -----------------------
    scores = []
    expected = pd.read_csv(path/'results/mape.csv', float_precision='round_trip')
    report_keys = ['window', 'period', 'method']
    for report in expected.itertuples(index=False):
        current = products[products['window'] == report.window]
        if report.period != 'both':
            current = current[current['origin'] == report.period]
        error = current['ape_'+report.method].dropna()
        assert len(error) == report.positive_n
        assert (current['actual_qty'] == 0).sum() == report.zero_n
        assert np.isclose(error.mean(), report.mape, rtol=0, atol=1e-8)
        scores.append(dict(window=report.window, period=report.period, method=report.method,
                           mape=error.mean(), positive_n=len(error), over_50=int((error > 50).sum()),
                           over_100=int((error > 100).sum())))
    assert len(expected) == 12 and not expected.duplicated(report_keys).any()
    return pd.DataFrame(scores)


def save_png(products, scores, comparisons, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Спільні межі для двох моделей та обох довжин історії охоплюють усі прогнози.
    positive = products[products['actual_qty'].gt(0)]
    x_max = positive[['ape_standard', 'ape_project']].max().max()
    y_max = positive[['standard', 'project']].max().max()
    x_ticks = [0, 15, 25, 50, 100, 300, 1000, 3000, 10000, 20000]
    y_ticks = [0, 1, 10, 50, 200, 500, 1000, 2500]
    both_scores = scores[scores['period'] == 'both'].set_index(['window', 'method'])
    for window, title in [('112d', 'коротка історія · 112 днів'), ('all', 'повна історія')]:
        current = positive[positive['window'].eq(window)]
        summary = comparisons[comparisons['window'].eq(window) & comparisons['period'].eq('both')].iloc[0]
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
        fig.patch.set_facecolor('#f6f8fb')
        titles = {}
        for method, label in [('standard', 'Стандартне масштабування'), ('project', 'Проєктне нормування')]:
            score = both_scores.loc[(window, method)]
            mape = format(score.mape, '.2f').replace('.', ',')
            titles[method] = f'{label}\nMAPE {mape}% · кількість похибок понад 100% - {int(score.over_100)}'
        draw_forecasts(axes, current, titles, x_ticks, y_ticks, np.log1p(x_max)*1.035,
                       np.log1p(y_max)*1.05, positive['actual_qty'].max())
        fig.suptitle(f'Сумарні продажі товару за акцію · {title}', fontsize=19, x=.07, ha='left', y=.97)
        fig.text(.07, .895, f'Дві акції, {len(current)} товарів-періодів із продажами. Більша бульбашка означає більші фактичні продажі.', fontsize=11)
        fig.text(.07, .08, 'Обидві осі мають шкалу log(1+x). Межі осей однакові для обох моделей та обох довжин історії.', fontsize=10, color='#465268')
        fig.text(.07, .035, f'З проєктним нормуванням похибка менша у {int(summary.better)} випадках, більша у {int(summary.worse)}, однакова у {int(summary.equal)}.', fontsize=11)
        fig.subplots_adjust(left=.07, right=.98, top=.76, bottom=.23, wspace=.10)
        fig.savefig(output/f'product_error_distribution_{window}.png', dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Розподіл помилок товарних тоталів без нового навчання')
    parser.add_argument('--png', action='store_true', help='Додаткові рисунки; потрібен matplotlib')
    args = parser.parse_args()
    path = Path(__file__).resolve().parent
    output = path/'results'
    products, difference = prepare_products(path)
    groups, comparisons, zeros = summarize(products)
    scores = check_mape(products, path)
    groups.to_csv(output/'product_error_bins.csv', index=False)
    comparisons.to_csv(output/'product_error_comparison.csv', index=False)
    zeros.to_csv(output/'product_error_zero_actual.csv', index=False)
    scores.to_csv(output/'product_error_summary.csv', index=False)
    products.to_csv(output/'product_error_details.csv', index=False)

    columns = ['window', 'origin', 'product_id', 'actual_qty', 'standard', 'project', 'ape_standard', 'ape_project']
    positive = products[products['actual_qty'].gt(0)]
    payload = dict(columns=columns, rows=positive[columns].values.tolist(), bins=groups.to_dict('records'),
                   comparisons=comparisons.to_dict('records'), zeros=zeros.to_dict('records'),
                   scores=scores.to_dict('records'), bubble_max_actual=float(positive['actual_qty'].max()))
    template = (path/'assets/product_error_distribution_template.html').read_text(encoding='utf-8')
    html = template.replace('__DATA__', json.dumps(payload, ensure_ascii=False, allow_nan=False))
    (output/'product_error_distribution.html').write_text(html, encoding='utf-8')
    if args.png:
        save_png(products, scores, comparisons, output)
    checks = dict(product_totals_match_main_report=True, max_total_or_ape_difference=difference,
                  all_12_mape_values_match=True, bin_counts_sum_to_positive_n=True,
                  positive_per_window=positive.groupby('window').size().to_dict(),
                  zero_per_window=products[products['actual_qty'].eq(0)].groupby('window').size().to_dict(),
                  source_manifests_verified=True, equality_tolerance=1e-8)
    (output/'product_error_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(scores[scores['period'].eq('both')].to_string(index=False))
    print(comparisons.to_string(index=False))
    print('Готово: results/product_error_distribution.html')


if __name__ == '__main__':
    main()
