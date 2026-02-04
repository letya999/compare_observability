#!/usr/bin/env python3
"""CLI script to run test scenarios."""

import argparse
import sys
from pathlib import Path

from scenarios import ScenarioRunner, list_scenarios


def main():
    parser = argparse.ArgumentParser(
        description="Run observability comparison test scenarios"
    )
    parser.add_argument(
        "--scenarios",
        "-s",
        nargs="+",
        help="Specific scenarios to run (default: all)",
    )
    parser.add_argument(
        "--providers",
        "-p",
        nargs="+",
        help="Observability providers to use",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available scenarios and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for name in list_scenarios():
            print(f"  - {name}")
        return 0

    # Initialize runner
    runner = ScenarioRunner(
        providers=args.providers,
        output_dir=args.output,
    )

    # Run scenarios
    if args.scenarios:
        for scenario_name in args.scenarios:
            runner.run_scenario(scenario_name)
    else:
        runner.run_all()

    # Export and summarize
    runner.export_results()
    runner.print_summary()
    runner.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
