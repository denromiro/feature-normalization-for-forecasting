# Два способи нормування тих самих 40 числових ознак.
import numpy as np
from sklearn.preprocessing import StandardScaler
from radial_budget import radial_transform


def fit_normalization(calibration, settings):
    blocks = settings['feature_blocks']
    columns = [c for block in blocks.values() for c in block]
    center = calibration[columns].median().fillna(0)
    # Основний масштаб - міжквартильний розмах. Коли він майже нульовий, беремо
    # стандартне відхилення, а для сталої ознаки - одиницю.
    scale = calibration[columns].quantile(.75) - calibration[columns].quantile(.25)
    tolerance = 1e-9 * center.abs().clip(lower=1)
    scale = scale.where(scale > tolerance)
    std = calibration[columns].std(ddof=0)
    scale = scale.fillna(std.where(std > tolerance)).fillna(1)
    state = {'columns': columns, 'blocks': blocks, 'numeric': settings['numeric_features'],
             'center': center, 'scale': scale, 'divisor': np.sqrt(len(blocks))}
    state['standard'] = StandardScaler().fit(calibration[columns].fillna(center))
    prepared = prepare_coordinates(calibration, state)
    state['ball_scales'] = {}
    for name, block in blocks.items():
        radius = np.linalg.norm(prepared[block].to_numpy(), axis=1)
        radius95 = np.quantile(radius, .95)
        if radius95 <= 1e-12:
            positive = radius[radius > 1e-12]
            radius95 = np.median(positive) if len(positive) else 1.
        # Після стискання цей опорний радіус блоку дорівнює 0,8.
        state['ball_scales'][name] = float(radius95 * (1 - .8) / .8)
    return state


def prepare_coordinates(data, state):
    # Прапорець зберігає інформацію про пропуск після заміни медіаною.
    result = data[state['numeric']].copy()
    for block in state['blocks'].values():
        result[[c + '_missing' for c in block]] = data[block].isna().astype(int).to_numpy()
        center = state['center'][block]
        filled = data[block].fillna(center)
        scaled = (filled - center) / state['scale'][block]
        result[block] = scaled / np.sqrt(len(block))
    return result[state['numeric'] + [c + '_missing' for c in state['columns']]]


def bounded_coordinates(data, state):
    # Стискаємо кожен блок у відкриту одиничну кулю. Об'єднані блоки
    # потрапляють в одну кулю після ділення на корінь із кількості блоків.
    result = prepare_coordinates(data, state)
    for name, block in state['blocks'].items():
        values = result[block].to_numpy()
        radius = np.linalg.norm(values, axis=1)
        ball_radius = radius / (state['ball_scales'][name] + radius)
        assert (ball_radius < 1).all()
        factor = np.divide(ball_radius, radius, out=np.zeros_like(radius), where=radius > 0)
        result[block] = values * factor[:, None]
    return result


def transform(data, state, method):
    columns = state['columns']
    if method == 'standard':
        result = data[state['numeric']].copy()
        result[[c + '_missing' for c in columns]] = data[columns].isna().astype(int).to_numpy()
        result[columns] = state['standard'].transform(data[columns].fillna(state['center']))
    else:
        assert method == 'project'
        result = bounded_coordinates(data, state)
        values = result[columns].to_numpy() / state['divisor']
        radial = state['radial']
        # Застосовуємо профіль, який пройшов інтегральний відбір у fit_radial_budget.
        result[columns] = radial_transform(values, radial['source'], radial['destination']) * state['divisor']
    assert np.isfinite(result[columns]).all().all()
    return result[state['numeric'] + [c + '_missing' for c in columns]]
