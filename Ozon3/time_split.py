import datetime as dt
from dataclasses import dataclass

import config as cfg


@dataclass
class Fold:
    name: str
    cutoff: dt.date  # включительно, последний день истории
    target_start: dt.date
    target_end: dt.date  # включительно


def _ensure_date(d: dt.date | str) -> dt.date:
    """Приводит дату к datetime.date для защиты от типов pandas.Timestamp."""
    if hasattr(d, "date"):
        return d.date()
    if isinstance(d, str):
        return dt.datetime.strptime(d, "%Y-%m-%d").date()
    return d


def build_folds(n_folds: int = 6, step_days: int = 20) -> list[Fold]:
    cutoff = _ensure_date(cfg.HIST_END)
    target_start = _ensure_date(cfg.TARGET_START)
    target_end = _ensure_date(cfg.TARGET_END)

    test_fold = Fold("test", cutoff, target_start, target_end)

    # Собираем фолды от самого свежего к правому краю истории
    raw_folds = []
    cur_cutoff = cutoff - dt.timedelta(days=cfg.TARGET_LEN_DAYS)

    for i in range(n_folds):
        t_start = cur_cutoff + dt.timedelta(days=1)
        t_end = t_start + dt.timedelta(days=cfg.TARGET_LEN_DAYS - 1)

        assert t_end <= cutoff, (
            f"target_end={t_end} выходит за пределы истории ({cutoff}). "
            "Это баг в параметрах сплита."
        )

        raw_folds.append((cur_cutoff, t_start, t_end))
        cur_cutoff = cur_cutoff - dt.timedelta(days=step_days)

    # Разворачиваем хронологически (от старых к новым)
    raw_folds = raw_folds[::-1]

    cv_folds = []
    for idx, (c_off, t_st, t_en) in enumerate(raw_folds):
        assert c_off >= _ensure_date(cfg.HIST_START), (
            f"Фолд fold_{idx} ({c_off}) уходит раньше начала истории ({cfg.HIST_START}). "
            "Уменьшите n_folds или step_days."
        )
        cv_folds.append(Fold(f"fold_{idx}", c_off, t_st, t_en))

    return cv_folds + [test_fold]


if __name__ == "__main__":
    for f in build_folds(n_folds=3):
        print(f)