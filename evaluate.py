# Після фіксації прогнозів порівнюємо прогноз і факт кожного товару за його промо.
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from normalization import transform


def evaluate(path):
    output = path / 'outputs'
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((output / 'predictions.csv.gz').read_bytes()).hexdigest() == protocol['prediction_sha256']
    for name, sha in protocol['code_sha256'].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == sha, name
    for name, entry in protocol['data_manifest'].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == entry['sha256'], name
    prediction = pd.read_csv(output / 'predictions.csv.gz', float_precision='round_trip')
    truth = pd.read_csv(path / 'data/evaluation.csv.gz', float_precision='round_trip')
    keys = ['origin', 'window', 'method']
    assert not prediction.duplicated(keys + ['sample_id']).any()
    detail = prediction.merge(truth, on='sample_id', validate='many_to_one')
    assert len(detail) == len(prediction) and detail['actual_qty'].notna().all()
    # Для оцінки беремо магазини, де в товару були продажі або додатний залишок під час промо.
    active = detail[detail['active_or']].copy()
    products = active.groupby(keys + ['product_id'])[['forecast', 'actual_qty']].sum().reset_index()
    absolute_error = (products['forecast'] - products['actual_qty']).abs()
    products['ape'] = absolute_error / products['actual_qty'].replace(0, np.nan) * 100
    products.to_csv(output / 'product_forecast_fact.csv', index=False)

    # %% ----------------------- незважений MAPE, кожне промо і обидва разом -----------------------
    scores = []
    for period in ['2026-05-28', '2026-06-11', 'both']:
        current = products if period == 'both' else products[products['origin'] == period]
        for (window, method), group in current.groupby(['window', 'method']):
            scores.append({'period': period, 'window': window, 'method': method, 'mape': group['ape'].mean(),
                           'positive_n': int(group['actual_qty'].gt(0).sum()), 'zero_n': int(group['actual_qty'].eq(0).sum())})
    scores = pd.DataFrame(scores)
    assert scores.loc[scores['period'] == 'both', 'positive_n'].eq(797).all()
    scores.to_csv(output / 'mape.csv', index=False)
    # Незалежна перевірка товарних сум через bincount.
    for key, group in active.groupby(keys):
        codes, ids = pd.factorize(group['product_id'], sort=True)
        expected = products.set_index(keys).sort_index().loc[key].set_index('product_id').loc[ids]
        assert np.allclose(np.bincount(codes, weights=group['forecast']), expected['forecast'], atol=1e-7)
        assert np.allclose(np.bincount(codes, weights=group['actual_qty']), expected['actual_qty'], atol=1e-10, rtol=0)

    # %% ----------------------- відтворення збережених моделей і порівняння з еталоном -----------------------
    targets = pd.read_csv(path / 'data/target.csv.gz', float_precision='round_trip')
    replay_error = 0.
    for key, group in prediction.groupby(keys):
        date, window, method = key
        target = targets.set_index('sample_id').loc[group['sample_id']]
        saved = joblib.load(output / ('_'.join(key) + '.joblib'))
        state = joblib.load(output / saved['normalization_file'])
        x = transform(target, state, method)
        categories = protocol['settings']['category_features']
        x[categories] = saved['encoder'].transform(target[categories])
        with threadpool_limits(limits=protocol['settings']['threads']):
            replay = np.maximum(saved['model'].predict(x[saved['columns']]), 0) * target['horizon_days'].to_numpy()
        replay_error = max(replay_error, float(np.max(np.abs(replay - group['forecast'].to_numpy()))))
    assert replay_error < 1e-8
    reference_available = (path / 'reference/predictions.csv.gz').exists()
    reference_error = None
    if reference_available:
        manifest = json.loads((path / 'reference_manifest.json').read_text(encoding='utf-8'))
        for name, entry in manifest.items():
            assert hashlib.sha256((path / name).read_bytes()).hexdigest() == entry['sha256'], name
        reference = pd.read_csv(path / 'reference/predictions.csv.gz', float_precision='round_trip')
        joined = prediction.merge(reference[keys + ['sample_id', 'forecast']], on=keys + ['sample_id'],
                                  validate='one_to_one', suffixes=('', '_reference'))
        assert len(joined) == len(reference) == len(prediction)
        reference_error = float((joined['forecast'] - joined['forecast_reference']).abs().max())
        assert np.allclose(joined['forecast'], joined['forecast_reference'], atol=1e-6, rtol=1e-8)
        old = pd.read_csv(path / 'reference/mape.csv', float_precision='round_trip')
        index = ['period', 'window', 'method']
        actual_scores = scores.set_index(index).sort_index()
        old_scores = old.set_index(index).sort_index()
        pd.testing.assert_index_equal(actual_scores.index, old_scores.index)
        assert np.allclose(actual_scores, old_scores, atol=1e-6, rtol=1e-8)

    # %% ----------------------- інтервал різниці MAPE, групування за товаром -----------------------
    comparisons = []
    rng = np.random.default_rng(42)
    for window in ['112d', 'all']:
        current = products[products['window'].eq(window)&products['actual_qty'].gt(0)]
        pairs = current.pivot(index=['origin', 'product_id'], columns='method', values='ape').reset_index()
        assert len(pairs) == 797 and pairs[['standard', 'project']].notna().all().all()
        pairs['difference'] = pairs['project'] - pairs['standard']
        grouped = pairs.groupby('product_id')['difference'].agg(['sum', 'size'])
        sample = rng.integers(0, len(grouped), size=(3000, len(grouped)))
        # Вибираємо товари з поверненням і зберігаємо всі промо кожного вибраного товару.
        difference_sum = grouped['sum'].to_numpy()[sample].sum(axis=1)
        period_count = grouped['size'].to_numpy()[sample].sum(axis=1)
        delta = difference_sum / period_count
        low, high = np.quantile(delta, [.025, .975])
        comparisons.append({'window': window, 'mape_difference_pp': pairs['difference'].mean(),
                            'bootstrap_low': low, 'bootstrap_high': high})
    pd.DataFrame(comparisons).to_csv(output / 'mape_comparisons.csv', index=False)
    validation = {'models': 8, 'prediction_rows': len(prediction), 'saved_models_replay_error': replay_error,
                  'reference_available': reference_available, 'reference_max_difference': reference_error,
                  'independent_product_sums_match': True, 'positive_product_periods': 797,
                  'source_hashes_match': True, 'network_shops': 148}
    (output / 'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')

    # %% ----------------------- звіт для перегляду -----------------------
    table = scores.pivot(index=['window', 'period'], columns='method', values='mape').reset_index()
    table['difference_pp'] = table['project'] - table['standard']
    table = table[['window', 'period', 'standard', 'project', 'difference_pp']]
    summary = table[table['period'] == 'both']
    summary.to_csv(output / 'summary.csv', index=False)
    names = {'112d': '112 днів', 'all': 'Повна історія', 'both': 'Обидва промо',
             '2026-05-28': '28.05-10.06, 14 днів', '2026-06-11': '11-21.06, 11 днів'}
    def html_table(frame):
        labels = {'window': 'Історія', 'period': 'Період', 'standard': 'Стандартне, MAPE %',
                  'project': 'Проєктне, MAPE %', 'difference_pp': 'Різниця, в. п.'}
        table = frame.replace(names).rename(columns=labels).rename_axis(None, axis='columns')
        html = table.to_html(index=False, border=0, float_format=lambda value: f'{value:.2f}'.replace('.', ','))
        return '<div class="scroll">' + html + '</div>' 
    html = '''<!doctype html><html lang="uk"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Стандартне та проєктне нормування: MAPE товарних тоталів</title>
<style>body{font:16px Segoe UI,Arial;max-width:1150px;margin:32px auto;padding:0 20px;color:#203c34}p,li{line-height:1.6}table{border-collapse:collapse;font-size:14px}td,th{padding:10px;border-bottom:1px solid #dce5df;text-align:right;white-space:nowrap}th{background:#edf5ef}.scroll{overflow:auto;margin-bottom:26px}a{color:#146b50}</style>
<h1>Стандартне та проєктне нормування</h1>
<p>Для мережі зі 148 магазинів порівнюємо два способи нормування ознак. Градієнтний бустинг прогнозує продажі в кожному магазині. Для оцінки підсумовуємо прогноз і фактичні продажі товару за період промо.</p>
<h2>Обидва промо</h2>''' + html_table(summary) + '''
<h2>Кожне промо</h2>''' + html_table(table[table['period'] != 'both']) + '''
<p>MAPE дорівнює середній абсолютній відносній похибці для 797 товарів-періодів із додатними продажами (609 у першому промо і 188 у другому). До підсумку входять магазини, де під час акції товар продавався або мав додатний залишок. Навіть за цього відбору товар міг бути відсутнім у частині днів.</p>
<p>Нормування використовує останні 56 днів доступної історії. Модель вчиться прогнозувати середньоденні продажі за перші 11 днів історичних промо або за всю акцію, якщо вона коротша. Прогнозований темп множимо на тривалість контрольного промо (14 або 11 днів). Відстані для розрахунку ваг вимірюємо в координатах відповідного нормування.</p>
<p>Проєктний метод обирає радіальний профіль за обмеженням на інтеграл спотворення оберненого відображення. Джерелом цієї умови є означення класу перед <a href="https://arxiv.org/html/2509.02008v3#S4">теоремою 4.3 у статті Sevost’yanov, Targonskii та Ilkevych</a>. Зміна MAPE є емпіричним результатом двох промо, використаних також під час розроблення методу.</p>
<p><a href="../README.md">Опис і запуск</a> · <a href="../docs/THEORY.md">Математичне обґрунтування</a> · <a href="../results/product_error_distribution.html">Розподіл похибок за товарами</a> · <a href="mape.csv">MAPE</a> · <a href="mape_comparisons.csv">Інтервали різниці MAPE</a> · <a href="validation.json">Перевірки</a></p></html>'''
    (output / 'report.html').write_text(html, encoding='utf-8')
    print(table.round(3).to_string(index=False), flush=True)
    print(json.dumps(validation), flush=True)


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    evaluate(Path(__file__).resolve().parent)
