"""Атрибуты МО по динамике расходов: не «что покупают», а «как живёт ряд относительно страны».

Разложение (как в работе по прогнозированию): для категории c  log y_it = a_i + n_ct + z_it, где n_ct - общая
траектория категории (двусторонние эффекты по панели), a_i - уровень МО, z_it - собственное отклонение МО.
Атрибуты считаются по окну месяцев (весь период или полугодие) и только по известным месяцам:

  sync      - доля дисперсии месячных изменений лога МО, объяснённая изменениями общей траектории (R^2), среднее по категориям;
  beta      - эластичность месячных изменений МО к изменениям общей траектории (усилитель > 1, гаситель < 1);
  season    - собственный декабрьский пик: z в декабре минус среднее z соседних месяцев (сверх национального пика);
  drift     - наклон z за окно, % в год (МО отстаёт от страны или обгоняет её);
  mkt_speed - наклон z маркетплейсов минус наклон z «Всех категорий»: насколько быстрее страны приходят маркетплейсы;
  noise     - робастный шум z (MAD месячных изменений), %;
  food_share, mkt_share - доли продовольствия и маркетплейсов в «Всех категориях» (контекст корзины, не основной признак).
"""
import warnings

import numpy as np
import pandas as pd


def two_way(Y, iters=30):
    """y_it = a_i + n_t на несбалансированной панели (чередующиеся средние)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n = np.nanmean(Y, 0)
        a = np.zeros(Y.shape[0])
        for _ in range(iters):
            a = np.nan_to_num(np.nanmean(Y - n, 1))
            n = np.nanmean(Y - a[:, None], 0)
    return a, n


def decompose(P):
    """z[i, c, t] - собственное отклонение МО; n[c, t] - общая траектория категории."""
    Y = np.log(P.V)
    Z = np.full_like(Y, np.nan)
    N = np.full((Y.shape[1], Y.shape[2]), np.nan)
    for c in range(Y.shape[1]):
        a, n = two_way(Y[:, c, :])
        Z[:, c, :] = Y[:, c, :] - a[:, None] - n[None, :]
        N[c] = n
    return Z, N


def _slope(y):
    t = np.arange(y.shape[-1], dtype=float)
    ok = np.isfinite(y)
    out = np.full(y.shape[0], np.nan)
    for i in range(y.shape[0]):
        if ok[i].sum() >= 4:
            out[i] = np.polyfit(t[ok[i]], y[i, ok[i]], 1)[0] * 12
    return out


def window_features(P, Z, N, months):
    """Атрибуты по окну месяцев (список индексов). Возвращает DataFrame по МО."""
    Y = np.log(P.V)
    m = np.array(months)
    feats = {}
    sync, beta = [], []
    for c in range(Y.shape[1]):
        dy = np.diff(Y[:, c, m], axis=1)
        dn = np.diff(N[c, m])[None, :]
        ok = np.isfinite(dy)
        dnc = np.where(ok, dn, np.nan)
        cov = np.nanmean((dy - np.nanmean(dy, 1, keepdims=True)) * (dnc - np.nanmean(dnc, 1, keepdims=True)), 1)
        var_n = np.nanvar(dnc, 1)
        var_y = np.nanvar(dy, 1)
        b = cov / np.where(var_n > 0, var_n, np.nan)
        r2 = cov ** 2 / np.where((var_n > 0) & (var_y > 0), var_n * var_y, np.nan)
        sync.append(r2)
        beta.append(b)
    feats["sync"] = np.nanmean(np.array(sync), 0)
    feats["beta"] = np.nanmedian(np.array(beta), 0)
    # собственный декабрьский пик (если декабрь в окне): z(дек) - среднее z(окт, ноя, янв)
    mon = np.array([int(P.months[k][5:]) for k in m])
    zt = Z[:, 0, m]
    dec = np.where(mon == 12)[0]
    if len(dec):
        s = []
        for d in dec:
            nb = [x for x in (d - 2, d - 1, d + 1) if 0 <= x < len(m)]
            s.append(zt[:, d] - np.nanmean(zt[:, nb], 1))
        feats["season"] = np.nanmean(np.array(s), 0) * 100
    feats["drift"] = _slope(Z[:, 0, m]) * 100
    feats["mkt_speed"] = (_slope(Z[:, 2, m]) - _slope(Z[:, 0, m])) * 100
    dz = np.diff(Z[:, :, m], axis=2)
    med = np.nanmedian(dz, axis=2, keepdims=True)
    feats["noise"] = np.nanmedian(1.4826 * np.nanmedian(np.abs(dz - med), axis=2), axis=1) * 100
    tot = np.nanmean(P.V[:, 0, m], 1)
    feats["food_share"] = np.nanmean(P.V[:, 1, m], 1) / tot
    feats["mkt_share"] = np.nanmean(P.V[:, 2, m], 1) / tot
    return pd.DataFrame(feats, index=P.ids)


def shift_history(P, Z, roads, ref_months=6, thr_noise=6.0, k_nb=10):
    """Подтверждённые сдвиги задним числом (как в работе по прогнозированию): среднее z за 3 мес. после месяца t
    отличается от 3 мес. до больше чем на thr_noise единиц шума ряда. Два канала: сам ряд «Всех категорий» и ряд
    относительно 10 ближайших соседей по дорогам. Возвращает число сдвигов по каналам и месяц первого."""
    z = Z[:, 0, :]
    D = roads.copy()
    np.fill_diagonal(D, np.inf)
    nb = np.argsort(np.where(np.isfinite(D), D, np.inf), axis=1)[:, :k_nb]
    rel = z - np.nanmean(z[nb], axis=1)
    out = {}
    for name, x in (("own", z), ("rel", rel)):
        dx = np.diff(x[:, :ref_months], axis=1)
        s = np.nanstd(dx, 1) / np.sqrt(2)
        T = x.shape[1]
        sh = np.zeros_like(x, bool)
        for t in range(max(3, ref_months), T - 2):
            jump = np.nanmean(x[:, t:t + 3], 1) - np.nanmean(x[:, t - 3:t], 1)
            sh[:, t] = np.abs(jump) > thr_noise * np.where(s > 0, s, np.nan)
        # серия соседних месяцев - один сдвиг
        starts = sh & ~np.concatenate([np.zeros((sh.shape[0], 1), bool), sh[:, :-1]], 1)
        out[f"shifts_{name}"] = starts.sum(1)
    return pd.DataFrame(out, index=P.ids)
