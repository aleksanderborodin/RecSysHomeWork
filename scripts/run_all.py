import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core import data, experiment as ex, split
from core.seed import seed_everything


def main(force: bool) -> None:
    seed_everything(config.SEED)
    events, items, _ = data.load_processed()
    stages = split.make_stages(events, items)
    val, test = stages["val"], stages["test"]
    print("val:", val.stats, "\ntest:", test.stats, flush=True)

    best = {}
    _, best["pop_recent"] = ex.tune_popularity(val, force)
    _, best["knn"] = ex.tune_itemknn(val, force)
    _, best["ease"] = ex.tune_ease(val, force)
    _, best["ials"] = ex.tune_ials(val, force)
    _, best["custom"] = ex.tune_signal(val, best["ials"], force)
    _, _, best["fm"] = ex.tune_fm(val, force)
    ex.save_best(best)
    print("best:", best, flush=True)

    print(ex.signal_ablation(val, best["ials"], best["custom"], force=force))
    table, per_user = ex.run_test(test, best, force)
    print(table)
    ab = ex.signal_ablation(test, best["ials"], best["custom"], per_user[ex.NAMES["custom"]]["ndcg"], force)
    print(ab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="пересчитать всё, игнорируя кеш")
    main(parser.parse_args().force)
