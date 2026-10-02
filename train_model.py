"""
train_model.py - train once, save the model, commit it to GitHub.

    python train_model.py                  # downloads the dataset from Kaggle
    python train_model.py --csv data.csv   # or use a local CSV
    python train_model.py --tune           # Randomized Search (slower, usually better)

Output: model/brute_force_model.joblib  (loaded automatically by app.py)
Run it with the SAME Python and scikit-learn versions the deployed app uses.
"""
import argparse

import brute_force_detector as bfd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Path to a local dataset CSV (default: download from Kaggle)")
    ap.add_argument("--out", default="model/brute_force_model.joblib")
    ap.add_argument("--tune", action="store_true", help="Use Randomized Search")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=None,
                    help="Limit tree depth to shrink the model file")
    args = ap.parse_args()

    df = bfd.load_from_csv(args.csv) if args.csv else bfd.load_from_kaggle()
    missing = bfd.validate_columns(df)
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {missing}")

    clean, report = bfd.clean_data(df)
    print(f"Rows: {report['rows_before']} -> {report['rows_after']} "
          f"({report['duplicates_removed']} duplicates removed)")

    trained = bfd.train_model(
        clean, tune=args.tune, n_estimators=args.trees, max_depth=args.max_depth,
        search_iterations=40, cv_folds=5,
    )
    for name, val in trained.metrics.items():
        print(f"{name:10s}: {val * 100:.2f}%")
    if trained.best_params:
        print("Best parameters:", trained.best_params)

    size = bfd.save_model(trained, args.out)
    print(f"Saved {args.out} ({size / 1e6:.1f} MB)")
    if size > 90e6:
        print("WARNING: GitHub rejects files over 100 MB. Re-run with --max-depth 15 or fewer --trees.")


if __name__ == "__main__":
    main()
