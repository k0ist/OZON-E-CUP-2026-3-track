from dataclasses import dataclass
import datetime as dt
import config as cfg


@dataclass
class Fold:
    name: str
    cutoff: dt.date          # включительно, последний день истории
    target_start: dt.date
    target_end: dt.date      # включительно


def build_folds(n_folds: int = 6, step_days: int = 20) -> list[Fold]:
    folds = []
    # идём от теста назад
    cutoff = cfg.HIST_END
    target_start = cfg.TARGET_START
    target_end = cfg.TARGET_END

    # сначала добавим тестовый "фолд" (таргета для него у нас нет)
    test_fold = Fold("test", cutoff, target_start, target_end)

    # ВАЖНО: самый "свежий" CV-фолд обязан оставлять ровно TARGET_LEN_DAYS дней
    # ДО HIST_END для таргета - иначе (если step_days < TARGET_LEN_DAYS) таргет
    # для него уйдёт за пределы истории и будет тихо обрезан build_target'ом.
    # Поэтому первый шаг = TARGET_LEN_DAYS, а дальше уже шагаем по step_days.
    cv_folds = []
    cur_cutoff = cutoff - dt.timedelta(days=cfg.TARGET_LEN_DAYS)
    for i in range(n_folds):
        t_start = cur_cutoff + dt.timedelta(days=1)
        t_end = t_start + dt.timedelta(days=cfg.TARGET_LEN_DAYS - 1)
        assert t_end <= cfg.HIST_END, (
            f"fold_{n_folds - 1 - i}: target_end={t_end} выходит за пределы истории "
            f"({cfg.HIST_END}) - таргет был бы обрезан. Это баг в build_folds."
        )
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