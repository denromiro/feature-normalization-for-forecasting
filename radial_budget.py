# Умова на інтеграл спотворення оберненого відображення походить з означення
# сім’ї перед теоремою 4.3 у статті Sevost’yanov, Targonskii та Ilkevych
# https://arxiv.org/html/2509.02008v3#S4
# Квантили радіусів задають бажаний профіль. Значення бюджету задає експеримент.
import numpy as np
import pandas as pd
from scipy.special import betainc, logsumexp


def radial_transform(values, source, destination):
    radius = np.linalg.norm(values, axis=1)
    new_radius = np.interp(radius, source, destination)
    factor = np.divide(new_radius, radius, out=np.ones_like(radius), where=radius > 0)
    return values * factor[:, None]


def log_phi(value, phi_scale):
    # log(exp(value/scale)-1), без переповнення exp для великих спотворень.
    z = np.asarray(value) / phi_scale
    return z + np.log(-np.expm1(-z))


def integral_bounds(source, destination, dimension, phi_scale, subdivisions=32):
    # Обчислюємо інтеграл у координатах образу T. Кожна оболонка має вагу за її об’ємом і радіусом.
    # На кожному відрізку rho(r)=a*r+b, t=r*rho'/rho,
    # K_I(T(x),T^-1)=K_O(x,T)=max(t,1)^n/t.
    # На одному афінному відрізку K не зростає. Лівий край дає верхню оцінку.
    parts = np.linspace(0, 1, subdivisions + 1)
    r = source[:-1, None] + np.diff(source)[:, None] * parts
    s = destination[:-1, None] + np.diff(destination)[:, None] * parts
    slope = np.diff(destination) / np.diff(source)
    t = np.divide(slope[:, None] * r, s, out=np.ones_like(r), where=s > 0)
    log_k = dimension * np.maximum(np.log(t), 0) - np.log(t)
    k = np.exp(log_k)
    # Інтеграл s^(n-1)/(1+s^2)^n ds через неповну бета-функцію.
    # Нормування на ваговий об'єм B^n: I_{1/2}(n/2,n/2)=1/2.
    mass = 2 * np.diff(betainc(dimension / 2, dimension / 2, s * s / (1 + s * s)), axis=1)
    assert (mass >= 0).all() and np.isclose(mass.sum(), 1, atol=1e-12)
    keep = mass > 0
    log_mass = np.log(mass[keep])
    baseline = log_phi(1., phi_scale)
    lower = logsumexp(log_mass + log_phi(k[:, 1:][keep], phi_scale) - baseline)
    upper = logsumexp(log_mass + log_phi(k[:, :-1][keep], phi_scale) - baseline)
    return float(lower), float(upper)


def fit_radial_budget(values, settings):
    # values містить лише ознаки завершених промо з періоду калібрування.
    radius = np.linalg.norm(values, axis=1)
    low, high = settings['fixed_inner_radius'], settings['fixed_outer_radius']
    inside = radius[(radius > low)&(radius < high)]
    quantiles = np.linspace(.05, .95, 19)
    if len(inside):
        raw = pd.DataFrame({'source': np.quantile(inside, quantiles),
                            'destination': low + (high - low) * quantiles})
        # Повторні радіуси не можна розвести в різні точки гомеоморфізмом.
        knots = raw.groupby('source', as_index=False)['destination'].mean()
        knots = knots[(knots['source'] > low + 1e-10)&(knots['source'] < high - 1e-10)]
        if len(knots):
            distinct = np.r_[True, np.diff(knots['source']) > 1e-10]
            knots = knots.loc[distinct]
        source = np.r_[0, low, knots['source'], high, 1.]
        goal = np.r_[0, low, knots['destination'], high, 1.]
    else:
        source = goal = np.array([0, low, high, 1.])
    assert (np.diff(source) > 0).all() and (np.diff(goal) > 0).all()
    rows = []
    # Для кожного alpha перевіряємо верхню оцінку інтеграла до навчання регресії.
    log_budget_limit = np.log(settings['budget'] * (1 - settings['budget_margin']))
    for alpha in np.linspace(0, 1, settings['alpha_steps'] + 1):
        destination = source + alpha * (goal - source)
        lower, upper = integral_bounds(source, destination, values.shape[1], settings['phi_scale'],
                                      settings['integral_subdivisions'])
        feasible = upper <= log_budget_limit
        rows.append({'alpha': alpha, 'log_budget_lower': lower, 'log_budget_upper': upper,
                     'feasible': feasible, 'radial_loss': float(np.mean((destination - goal)**2))})
    candidates = pd.DataFrame(rows)
    # Мінімальна відстань до бажаного профілю серед допустимих кандидатів.
    feasible_candidates = candidates.loc[candidates['feasible']]
    selected = feasible_candidates.sort_values(['radial_loss', 'alpha']).iloc[0]
    alpha = float(selected['alpha'])
    destination = source + alpha * (goal - source)
    profile = {'source': source, 'goal': goal, 'destination': destination, 'alpha': alpha,
            'dimension': values.shape[1], 'settings': settings, 'inside_rows': len(inside),
            'log_budget_upper': float(selected['log_budget_upper'])}
    return profile, candidates
