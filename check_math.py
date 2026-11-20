# Числові перевірки обраного профілю. Виведення формул наведено у docs/THEORY.md.
import numpy as np
from scipy.integrate import quad
from scipy.special import beta
from radial_budget import radial_transform, integral_bounds


def check_profile(profile):
    source, destination = profile['source'], profile['destination']
    n = profile['dimension']
    scale = profile['settings']['phi_scale']
    assert (np.diff(source) > 0).all() and (np.diff(destination) > 0).all()
    assert np.array_equal(source[[0, 1, -2, -1]], destination[[0, 1, -2, -1]])
    # Квадратурою обчислюємо той самий інтеграл у координатах образу й звіряємо з оцінками за оболонками.
    total, error = 0., 0.
    weight_integral = beta(n / 2, n / 2) / 4
    for left, right, low, high in zip(source[:-1], source[1:], destination[:-1], destination[1:]):
        a = (high - low) / (right - left)
        b = low - a * left

        def integrand(s):
            t = (s - b) / s if s else 1.
            k = max(t, 1)**n / t
            return np.expm1(k / scale) / np.expm1(1 / scale) * s**(n - 1) / (1 + s * s)**n / weight_integral

        value, err = quad(integrand, low, high, epsabs=1e-10, epsrel=1e-9, limit=100)
        total += value
        error += err
    subdivisions = profile['settings'].get('integral_subdivisions', 32)
    lower, upper = integral_bounds(source, destination, n, scale, subdivisions)
    fine_lower, fine_upper = integral_bounds(source, destination, n, scale, 256)
    assert np.exp(lower) - 1e-8 <= total <= np.exp(upper) + 1e-8
    assert np.exp(fine_lower) - 1e-8 <= total <= np.exp(fine_upper) + 1e-8
    assert np.exp(upper) <= profile['settings']['budget']
    # Обернення на точках біля вузлів, центра, межі і у випадкових напрямах.
    rng = np.random.default_rng(42)
    radii = np.r_[0, 1, source, np.maximum(source - 1e-9, 0), np.minimum(source + 1e-9, 1), rng.random(1000)]
    points = rng.normal(size=(len(radii), n))
    points = points / np.linalg.norm(points, axis=1)[:, None] * radii[:, None]
    mapped = radial_transform(points, source, destination)
    restored = radial_transform(mapped, destination, source)
    inverse_error = float(np.max(np.abs(restored - points)))
    assert inverse_error < 1e-10
    # За скінченними різницями оцінюємо матрицю похідної. Її сингулярні числа
    # дають незалежну перевірку формули спотворення для T та його оберненого.
    jacobian_error = 0.
    for j in range(1, len(source) - 2):
        r = (source[j] + source[j + 1]) / 2
        direction = rng.normal(size=n)
        direction /= np.linalg.norm(direction)
        point = direction * r
        step = min(1e-6, (source[j + 1] - source[j]) / 1000)
        plus = radial_transform(point + np.eye(n) * step, source, destination)
        minus = radial_transform(point - np.eye(n) * step, source, destination)
        jacobian = ((plus - minus) / (2 * step)).T
        singular = np.linalg.svd(jacobian, compute_uv=False)
        log_ko = n * np.log(singular.max()) - np.log(singular).sum()
        inverse_singular = np.linalg.svd(np.linalg.inv(jacobian), compute_uv=False)
        log_ki_inverse = np.log(inverse_singular).sum() - n * np.log(inverse_singular.min())
        slope = (destination[j + 1] - destination[j]) / (source[j + 1] - source[j])
        t = r * slope / np.interp(r, source, destination)
        expected = n * max(np.log(t), 0) - np.log(t)
        jacobian_error = max(jacobian_error, abs(log_ko - expected), abs(log_ki_inverse - expected))
    assert jacobian_error < 1e-5
    return {'alpha': profile['alpha'], 'budget_lower': float(np.exp(lower)),
            'budget_upper': float(np.exp(upper)), 'budget_quad': total, 'quad_error_estimate': error,
            'fine_lower': float(np.exp(fine_lower)), 'fine_upper': float(np.exp(fine_upper)),
            'max_inverse_error': inverse_error, 'max_log_distortion_error': jacobian_error,
            'positive_slopes': True, 'anchors_fixed': True}


if __name__ == '__main__':
    # Перевіряємо тотожний профіль і мале відхилення від нього.
    settings = {'phi_scale': 100., 'budget': 2.}
    for destination in [np.array([0, .15, .4, .65, .9, 1.]), np.array([0, .15, .401, .649, .9, 1.])]:
        result = check_profile({'source': np.array([0, .15, .4, .65, .9, 1.]), 'destination': destination,
                                'dimension': 40, 'settings': settings, 'alpha': 0.})
        print(result)
    # Сильніша деформація мусить бути відхилена цим бюджетом.
    lower, upper = integral_bounds(np.array([0, .15, .4, .65, .9, 1.]),
                                  np.array([0, .15, .41, .64, .9, 1.]), 40, 100.)
    assert lower > np.log(2.)
    print('Профіль із перевищенням бюджету відхилено')
