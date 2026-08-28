import datetime as dt
from dataclasses import dataclass
from typing import Sequence

import config as cfg


@dataclass(frozen=True)
class Fold:
    name: str
    cutoff: dt.date
    target_start: dt.date
    target_end: dt.date


def _ensure_date(value: dt.date | str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    if hasattr(value, "date"):
        converted = value.date()
        if isinstance(converted, dt.date):
            return converted
    raise TypeError(f"Expected date-like value, got {type(value).__name__}: {value!r}")


def build_cv_folds(
    n_folds: int = cfg.N_FOLDS,
    step_days: int | None = cfg.STEP_DAYS,
) -> list[Fold]:
    """Build labeled 30-day snapshots from oldest to newest."""

    if not isinstance(n_folds, int) or n_folds < 1:
        raise ValueError(f"n_folds must be a positive integer, got {n_folds!r}")

    hist_start = _ensure_date(cfg.HIST_START)
    hist_end = _ensure_date(cfg.HIST_END)
    target_len = int(cfg.TARGET_LEN_DAYS)
    step_days = target_len if step_days is None else step_days

    if not isinstance(step_days, int) or step_days < target_len:
        raise ValueError(
            f"step_days={step_days!r} must be an integer >= target length "
            f"{target_len}; otherwise validation target windows overlap."
        )

    current_cutoff = hist_end - dt.timedelta(days=target_len)
    raw_folds: list[tuple[dt.date, dt.date, dt.date]] = []

    for _ in range(n_folds):
        target_start = current_cutoff + dt.timedelta(days=1)
        target_end = current_cutoff + dt.timedelta(days=target_len)
        if current_cutoff < hist_start:
            break
        if target_end > hist_end:
            raise ValueError(
                f"Target {target_start}..{target_end} exceeds HIST_END={hist_end}."
            )
        raw_folds.append((current_cutoff, target_start, target_end))
        current_cutoff -= dt.timedelta(days=step_days)

    if len(raw_folds) != n_folds:
        raise ValueError(
            f"Requested {n_folds} folds, but only {len(raw_folds)} fit inside "
            f"HIST_START={hist_start}..HIST_END={hist_end}."
        )

    raw_folds.reverse()
    folds = [
        Fold(
            name=f"fold_{index}",
            cutoff=cutoff,
            target_start=target_start,
            target_end=target_end,
        )
        for index, (cutoff, target_start, target_end) in enumerate(raw_folds)
    ]
    validate_fold_contract(folds)
    return folds


def build_test_fold() -> Fold:
    return Fold(
        name="test",
        cutoff=_ensure_date(cfg.HIST_END),
        target_start=_ensure_date(cfg.TARGET_START),
        target_end=_ensure_date(cfg.TARGET_END),
    )


def validate_fold_contract(
    cv_folds: Sequence[Fold],
    test_fold: Fold | None = None,
) -> None:
    """Fail fast unless folds satisfy the competition temporal contract."""

    folds = list(cv_folds)
    if not folds:
        raise ValueError("At least one labeled CV fold is required.")
    if any(fold.name == "test" for fold in folds):
        raise ValueError("Pass only labeled folds in cv_folds; test_fold is separate.")
    if len({fold.name for fold in folds}) != len(folds):
        raise ValueError("Fold names must be unique.")

    target_len = int(cfg.TARGET_LEN_DAYS)
    hist_end = _ensure_date(cfg.HIST_END)

    for index, fold in enumerate(folds):
        cutoff = _ensure_date(fold.cutoff)
        target_start = _ensure_date(fold.target_start)
        target_end = _ensure_date(fold.target_end)
        actual_len = (target_end - target_start).days + 1

        if target_start != cutoff + dt.timedelta(days=1):
            raise AssertionError(
                f"{fold.name}: target_start={target_start} must equal "
                f"cutoff+1={cutoff + dt.timedelta(days=1)}."
            )
        if actual_len != target_len:
            raise AssertionError(
                f"{fold.name}: target length={actual_len}, expected {target_len}."
            )
        if target_end > hist_end:
            raise AssertionError(
                f"{fold.name}: target_end={target_end} exceeds HIST_END={hist_end}."
            )

        if index == 0:
            continue
        previous = folds[index - 1]
        if cutoff <= _ensure_date(previous.cutoff):
            raise AssertionError("CV folds must be strictly ordered oldest to newest.")
        if target_start <= _ensure_date(previous.target_end):
            raise AssertionError(
                f"Target overlap: {previous.name}={previous.target_start}..{previous.target_end}, "
                f"{fold.name}={target_start}..{target_end}."
            )

        max_training_target_end = max(
            _ensure_date(train_fold.target_end) for train_fold in folds[:index]
        )
        if max_training_target_end > cutoff:
            raise AssertionError(
                f"{fold.name}: max training target_end={max_training_target_end} "
                f"exceeds validation cutoff={cutoff}."
            )

    if test_fold is not None:
        test_cutoff = _ensure_date(test_fold.cutoff)
        test_start = _ensure_date(test_fold.target_start)
        test_end = _ensure_date(test_fold.target_end)
        if test_fold.name != "test":
            raise AssertionError(f"Expected test fold name, got {test_fold.name!r}.")
        if test_cutoff != hist_end:
            raise AssertionError(f"test cutoff={test_cutoff}, expected HIST_END={hist_end}.")
        if test_start != test_cutoff + dt.timedelta(days=1):
            raise AssertionError("Test target must start on the day after HIST_END.")
        if (test_end - test_start).days + 1 != target_len:
            raise AssertionError("Test target window must be exactly 30 days.")
        if test_start != _ensure_date(cfg.TARGET_START) or test_end != _ensure_date(cfg.TARGET_END):
            raise AssertionError("Test fold does not match configured competition dates.")


def walk_forward_splits(
    folds: Sequence[Fold] | None = None,
) -> list[tuple[int, list[Fold], Fold]]:
    """Return only safe ``train=folds[:i]`` temporal splits.

    ``folds`` may be the backward-compatible output of :func:`build_folds`
    (including the test fold) or labeled CV folds only.
    """

    supplied = list(build_folds() if folds is None else folds)
    test_folds = [fold for fold in supplied if fold.name == "test"]
    if len(test_folds) > 1:
        raise ValueError("At most one test fold is allowed.")
    cv_folds = [fold for fold in supplied if fold.name != "test"]
    validate_fold_contract(cv_folds, test_folds[0] if test_folds else None)

    splits: list[tuple[int, list[Fold], Fold]] = []
    for validation_index in range(1, len(cv_folds)):
        training_folds = cv_folds[:validation_index]
        validation_fold = cv_folds[validation_index]
        if max(fold.target_end for fold in training_folds) > validation_fold.cutoff:
            raise AssertionError("Unsafe temporal split constructed.")
        splits.append((validation_index, training_folds, validation_fold))
    return splits


def build_folds(
    n_folds: int = cfg.N_FOLDS,
    step_days: int | None = cfg.STEP_DAYS,
) -> list[Fold]:
    """Backward-compatible API returning labeled folds followed by test."""

    cv_folds = build_cv_folds(n_folds=n_folds, step_days=step_days)
    test_fold = build_test_fold()
    validate_fold_contract(cv_folds, test_fold)
    return cv_folds + [test_fold]


if __name__ == "__main__":
    all_folds = build_folds()
    for fold in all_folds:
        print(
            f"{fold.name}: cutoff={fold.cutoff} | "
            f"target={fold.target_start}..{fold.target_end}"
        )

    print("\nWalk-forward validation splits:")
    for _, training_folds, validation_fold in walk_forward_splits(all_folds):
        train_anchors = ",".join(str(fold.cutoff) for fold in training_folds)
        train_target_end = max(fold.target_end for fold in training_folds)
        print(
            f"validation={validation_fold.name} | train anchors={train_anchors} | "
            f"train target_end={train_target_end} | validation cutoff={validation_fold.cutoff} | "
            f"validation target={validation_fold.target_start}..{validation_fold.target_end}"
        )
