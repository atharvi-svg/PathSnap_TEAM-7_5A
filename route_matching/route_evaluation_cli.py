"""Command line entry point for development route evaluation."""
import argparse
from .route_evaluation import evaluate_partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Path to Student folder")
    parser.add_argument("--partition", default="student_test", choices=["train", "student_test"])
    parser.add_argument("--output", default="route_matching/hmm_comparison.csv")
    parser.add_argument("--max-trips", type=int)
    args = parser.parse_args()
    result = evaluate_partition(args.data_root, args.partition, args.output, max_trips=args.max_trips)
    if not result.empty:
        print(result.select_dtypes("number").mean(numeric_only=True).to_string())

if __name__ == "__main__":
    main()
