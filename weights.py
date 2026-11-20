# Вага навчального рядка залежить від його відстані до найближчих рядків плану промо.
# Це прикладне правило експерименту. Його застосовуємо в координатах кожної моделі.
import numpy as np
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits


def proximity_weights(history_coordinates, target_coordinates, calibration, settings):
    history_coordinates = np.ascontiguousarray(history_coordinates, dtype=float)
    target_coordinates = np.ascontiguousarray(target_coordinates, dtype=float)
    nearest = NearestNeighbors(n_neighbors=settings['neighbors'], algorithm='brute', n_jobs=1)
    nearest.fit(target_coordinates)
    # Обробляємо історію частинами, щоб матриця попарних відстаней помістилася в пам’яті.
    parts = []
    with threadpool_limits(limits=settings['threads']):
        for start in range(0, len(history_coordinates), 4096):
            distances, _ = nearest.kneighbors(history_coordinates[start:start + 4096])
            parts.append(distances.mean(axis=1))
    distance = np.concatenate(parts)
    bandwidth = float(np.median(distance[calibration]))
    if bandwidth <= 1e-12:
        positive = distance[calibration & (distance > 1e-12)]
        bandwidth = float(np.median(positive)) if len(positive) else 1.
    floor = settings['weight_floor']
    # Медіана відстаней на періоді калібрування задає масштаб спадання ваги.
    raw_weight = floor + (1 - floor) * np.exp(-(distance / bandwidth)**2)
    assert np.isfinite(raw_weight).all() and raw_weight.min() >= floor and raw_weight.max() <= 1
    return raw_weight, distance, bandwidth
