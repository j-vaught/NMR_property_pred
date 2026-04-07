#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Morgan FP property prediction")
    parser.add_argument("--approach", choices=["direct", "arrhenius"], default="direct")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-only", action="store_true", help="Only build dataset, skip training")
    args = parser.parse_args()

    from shared.config import Paths, DataConfig, SplitConfig, ModelConfig, TrainConfig
    from phase1.data_pipeline import build_dataset
    from phase1.train import train_direct, evaluate_direct

    paths = Paths()
    data_config = DataConfig()
    split_config = SplitConfig(seed=args.seed)
    model_config = ModelConfig()
    train_config = TrainConfig(
        approach=args.approach,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
    )

    dataset = build_dataset(paths, data_config, split_config)

    if args.data_only:
        print("\nData pipeline verified. Exiting.")
        return

    if args.approach == "direct":
        model, sc_v, sc_s, history = train_direct(dataset, model_config, train_config)
        device = next(model.parameters()).device
        results = evaluate_direct(
            model, dataset["fp_df"], dataset["visc_df"], dataset["st_df"],
            dataset["splits"]["test"], sc_v, sc_s, device
        )
    else:
        print("Arrhenius training not yet implemented in run.py")
        return

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    for prop, metrics in results.items():
        print(f"\n{prop}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
