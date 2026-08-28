import argparse

import yaml

from src.llm import build_llm_client
from src.runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/fischer_2004_mvp.yaml",
        help="Path to the YAML experiment config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    llm = build_llm_client(config["model"], preflight=True)

    runner = ExperimentRunner(
        llm=llm,
        config_path=args.config,
        **config["experiment"],
    )
    runner.run()


if __name__ == "__main__":
    main()
