# %% ----------------------- дані, параметри та перевірка цілісності -----------------------
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import hashlib
import time
import numpy as np
import pandas as pd
import scipy
import sklearn
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from normalization import fit_normalization, bounded_coordinates, transform
from radial_budget import fit_radial_budget
from check_math import check_profile
from weights import proximity_weights

path = Path(__file__).resolve().parent
output = path / 'outputs'
output.mkdir(exist_ok=True)
settings = json.loads((path / 'settings.json').read_text(encoding='utf-8'))
manifest = json.loads((path / 'data_manifest.json').read_text(encoding='utf-8'))
for name, entry in manifest.items():
    assert hashlib.sha256((path / name).read_bytes()).hexdigest() == entry['sha256'], name
dates = ['date_start', 'date_end', 'eval_end']
history = pd.read_csv(path / 'data/history.csv.gz', parse_dates=dates, float_precision='round_trip')
targets = pd.read_csv(path / 'data/target.csv.gz', parse_dates=dates, float_precision='round_trip')
network = json.loads((path / 'data/network.json').read_text())['shop_ids']
assert len(network) == settings['network_shops'] == 148
assert history['shop_id'].isin(network).all() and targets['shop_id'].isin(network).all()
assert history['sample_id'].is_unique and targets['sample_id'].is_unique
assert 'actual_qty' not in targets and 'qty_per_day' not in targets
assert history['qty_per_day'].notna().all()
code = ['run_experiment.py', 'normalization.py', 'radial_budget.py', 'weights.py', 'check_math.py', 'evaluate.py']
protocol = {'settings': settings, 'data_manifest': manifest, 'python': sys.version,
            'packages': {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
                        'sklearn': sklearn.__version__, 'joblib': joblib.__version__},
            'code_sha256': {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in code}}
(output / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
predictions, audit = [], []

# %% ----------------------- послідовний прогноз кожного промо -----------------------
for origin in pd.to_datetime(settings['origins']):
    date = str(origin.date())
    available = history[history['date_end'] < origin].copy()
    target = targets[targets['date_start'] == origin].copy()
    calibration_start = origin - pd.Timedelta(days=settings['calibration_days'])
    calibration = available['date_start'].ge(calibration_start).to_numpy()
    assert available['date_end'].max() < origin and calibration.sum() > 64
    assert not set(available['sample_id']) & set(target['sample_id'])
    state = fit_normalization(available.loc[calibration], settings)
    columns = state['columns']
    bounded = bounded_coordinates(available.loc[calibration], state)
    values = bounded[columns].to_numpy() / state['divisor']
    radial, candidates = fit_radial_budget(values, settings['radial'])
    math_check = check_profile(radial)
    # Фіксуємо профіль і всі результати відбору кандидатів до прогнозування.
    state['radial'] = radial
    joblib.dump(state, output / (date + '_normalization.joblib'))
    candidates.to_csv(output / (date + '_candidates.csv'), index=False)
    (output / (date + '_math_checks.json')).write_text(json.dumps(math_check, indent=2), encoding='utf-8')
    print('Нормування', date, 'alpha', radial['alpha'], 'budget', math_check['budget_upper'], flush=True)
    for method in settings['methods']:
        x_all = transform(available, state, method)
        x_target = transform(target, state, method)
        raw_weights, _, bandwidth = proximity_weights(x_all[columns], x_target[columns], calibration, settings)
        for window, days in settings['windows'].items():
            started = time.monotonic()
            use = np.ones(len(available), dtype=bool)
            if days is not None:
                history_start = origin - pd.Timedelta(days=days)
                use = available['date_start'].ge(history_start).to_numpy()
            train = available.loc[use]
            x_train = x_all.loc[use].copy()
            x_test = x_target.copy()
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
            categories = settings['category_features']
            n_numeric = x_train.shape[1]
            x_train[categories] = encoder.fit_transform(train[categories])
            x_test[categories] = encoder.transform(target[categories])
            assert all(len(c) <= 255 for c in encoder.categories_)
            window_weights = raw_weights[use]
            weights = window_weights / window_weights.mean()
            assert np.isclose(weights.mean(), 1.)
            stem = '_'.join([date, window, method])
            print('Навчання', stem, len(train), flush=True)
            model = HistGradientBoostingRegressor(**settings['model'],
                categorical_features=list(range(n_numeric, x_train.shape[1])))
            with threadpool_limits(limits=settings['threads']):
                model.fit(x_train, train['qty_per_day'], sample_weight=weights)
                forecast = np.maximum(model.predict(x_test), 0) * target['horizon_days'].to_numpy()
            assert np.isfinite(forecast).all()
            joblib.dump({'model': model, 'encoder': encoder, 'columns': list(x_train.columns),
                         'normalization_file': date + '_normalization.joblib', 'method': method}, output / (stem + '.joblib'))
            part = target[['sample_id', 'product_id', 'shop_id', 'horizon_days']].copy()
            part['origin'] = date
            part['window'] = window
            part['method'] = method
            part['forecast'] = forecast
            predictions.append(part)
            audit.append({'origin': date, 'window': window, 'method': method, 'train_rows': len(train),
                          'calibration_rows': int(calibration.sum()), 'target_rows': len(target),
                          'train_latest_end': str(train['date_end'].max().date()), 'bandwidth': bandwidth,
                          'weight_mean': float(weights.mean()), 'effective_rows': float(weights.sum()**2 / (weights**2).sum()),
                          'alpha': radial['alpha'] if method == 'project' else np.nan, 'seconds': time.monotonic() - started})
            pd.DataFrame(audit).to_csv(output / 'fit_audit.csv', index=False)

# %% ----------------------- фіксуємо прогноз перед завантаженням факту -----------------------
assert len(audit) == 8
prediction = pd.concat(predictions, ignore_index=True)
assert not prediction.duplicated(['origin', 'window', 'method', 'sample_id']).any()
prediction.to_csv(output / 'predictions.csv.gz', index=False)
protocol['prediction_sha256'] = hashlib.sha256((output / 'predictions.csv.gz').read_bytes()).hexdigest()
(output / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
print('8 моделей готові. Оцінюємо товарні тотали.', flush=True)
from evaluate import evaluate
evaluate(path)
