import datetime as dt
from dataclasses import dataclass

import config as cfg


@dataclass
class Fold:
    name: str
    cutoff: dt.date
    target_start: dt.date
    target_end: dt.date


def _ensure_date(d: dt.date | str) -> dt.date:
    if isinstance(d, dt.datetime):
        return d.date()

    if hasattr(d, "date") and not isinstance(d, dt.date):
        return d.date()

    if isinstance(d, str):
        return dt.datetime.strptime(d, "%Y-%m-%d").date()

    return d


def build_folds(
    n_folds: int = 6,
    step_days: int | None = None,
) -> list[Fold]:

    hist_end = _ensure_date(cfg.HIST_END)
    hist_start = _ensure_date(cfg.HIST_START)

    target_len = cfg.TARGET_LEN_DAYS

    # Для временной CV лучше не допускать пересечения target-периодов.
    if step_days is None:
        step_days = target_len

    if step_days < target_len:
        raise ValueError(
            f"step_days={step_days} меньше длины target={target_len}. "
            "Это приводит к перекрытию target-периодов."
        )

    # Последний CV cutoff должен быть непосредственно перед
    # последним известным target-периодом.
    current_cutoff = hist_end - dt.timedelta(days=target_len)

    raw_folds = []

    for _ in range(n_folds):
        target_start = current_cutoff + dt.timedelta(days=1)
        target_end = current_cutoff + dt.timedelta(days=target_len)

        if current_cutoff < hist_start:
            break

        if target_end > hist_end:
            raise ValueError(
                f"Target {target_start}..{target_end} выходит "
                f"за HIST_END={hist_end}"
            )

        raw_folds.append(
            (
                current_cutoff,
                target_start,
                target_end,
            )
        )

        current_cutoff -= dt.timedelta(days=step_days)

    # От старого к новому
    raw_folds.reverse()

    folds = []

    for i, (cutoff, target_start, target_end) in enumerate(raw_folds):
        folds.append(
            Fold(
                name=f"fold_{i}",
                cutoff=cutoff,
                target_start=target_start,
                target_end=target_end,
            )
        )

    # Проверяем отсутствие пересечения target-периодов
    for i in range(1, len(folds)):
        prev = folds[i - 1]
        cur = folds[i]

        if cur.target_start <= prev.target_end:
            raise AssertionError(
                f"Пересечение target-периодов: "
                f"{prev.name}={prev.target_start}..{prev.target_end}, "
                f"{cur.name}={cur.target_start}..{cur.target_end}"
            )

    # Test-фолд
    test_fold = Fold(
        name="test",
        cutoff=hist_end,
        target_start=_ensure_date(cfg.TARGET_START),
        target_end=_ensure_date(cfg.TARGET_END),
    )

    return folds + [test_fold]


if __name__ == "__main__":

    folds = build_folds(n_folds=6)

    for fold in folds:
        print(
            f"{fold.name}: "
            f"cutoff={fold.cutoff} | "
            f"target={fold.target_start}..{fold.target_end}"
        )