"""
Time-based валидация.

Каждый фолд = (cutoff_date, target_start, target_end):
  - фичи считаются по истории СТРОГО ДО cutoff_date (включительно)
  - таргет = сумма gmv за [target_start, target_end] (30 дней)

Схема повторяет реальный сдвиг теста, поэтому дистанция между
cutoff и target_start везде одинаковая (1 день), а длина таргет-окна
всегда 30 дней - это критично, чтобы CV был репрезентативен.
"""
from dataclasses import dataclass
import datetime as dt
import config as cfg


@dataclass
class Fold:
    name: str
    cutoff: dt.date          # включительно, последний день истории
    target_start: dt.date
    target_end: dt.date      # включительно


def build_folds(n_folds: int = 3, step_days: int = 30) -> list[Fold]:
    """
    Строит n_folds скользящих фолдов, заканчивающихся перед финальным тестом,
    плюс сам финальный тест-фолд (без таргета, для сабмита).

    Пример при n_folds=3, step_days=30 (окно таргета = 30 дней):
      fold_0: cutoff=2025-11-16 -> target [2025-11-17 .. 2025-12-16]
      fold_1: cutoff=2025-12-16 -> target [2025-12-17 .. 2026-01-15]
      fold_2: cutoff=2026-01-15 -> target [2026-01-16 .. 2026-02-13]
      test  : cutoff=2026-02-13 -> target [2026-02-14 .. 2026-03-15]  (сдаём)
    """
    folds = []
    # идём от теста назад
    cutoff = cfg.HIST_END
    target_start = cfg.TARGET_START
    target_end = cfg.TARGET_END

    # сначала добавим тестовый "фолд" (таргета для него у нас нет)
    test_fold = Fold("test", cutoff, target_start, target_end)

    cv_folds = []
    cur_cutoff = cutoff - dt.timedelta(days=step_days)
    for i in range(n_folds):
        t_start = cur_cutoff + dt.timedelta(days=1)
        t_end = t_start + dt.timedelta(days=cfg.TARGET_LEN_DAYS - 1)
        cv_folds.append(Fold(f"fold_{n_folds - 1 - i}", cur_cutoff, t_start, t_end))
        cur_cutoff = cur_cutoff - dt.timedelta(days=step_days)

    cv_folds = cv_folds[::-1]  # по возрастанию времени

    for f in cv_folds:
        assert f.cutoff >= cfg.HIST_START, (
            f"Фолд {f.name} уходит раньше начала истории ({f.cutoff} < {cfg.HIST_START}). "
            "Уменьши n_folds или step_days."
        )

    return cv_folds + [test_fold]


if __name__ == "__main__":
    for f in build_folds(n_folds=3):
        print(f)
