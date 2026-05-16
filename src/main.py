"""
Main executable CLI interface

"""

from __future__ import annotations

import argparse

from src.outfit_generator import generate_outfits, complete_outfit


def print_outfit(result: dict, rank: int):
    score = result["score"]
    items = result["items"]

    print("=" * 60)
    print(f"OUTFIT #{rank}")
    print("-" * 60)

    for item in items:
        print(f"• {item['zara_category']}  |  ID: {item['item_id']}")
        print(f"  Crop: {item['crop_path']}")

    print("\nScore Breakdown:")
    print(f"  Total Score   : {score.total:.3f}")
    print(f"  Rating Stars  : {'⭐' * score.rating_stars}")
    print(f"  Style Score   : {score.style_score:.3f}")
    print(f"  Event Score   : {score.event_score:.3f}")
    print(f"  Color Score   : {score.color_score:.3f}")
    print(f"  Category Score: {score.category_score:.3f}")

    if score.learned_score is not None:
        print(f"  Learned Score : {score.learned_score:.3f}")

    print("\nExplanation:")
    for note in score.explanation:
        print(f"  - {note}")

    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(description="Fashion AI CLI")

    subparsers = parser.add_subparsers(dest="command")

    # Event mode
    event_parser = subparsers.add_parser("event")
    event_parser.add_argument("event_key", type=str, help="Event key from events_config")
    event_parser.add_argument("--topk", type=int, default=5)

    # Complete mode
    complete_parser = subparsers.add_parser("complete")
    complete_parser.add_argument("item_id", type=str)
    complete_parser.add_argument("event_key", type=str)
    complete_parser.add_argument("--topk", type=int, default=5)

    args = parser.parse_args()

    if args.command == "event":
        results = generate_outfits(args.event_key, top_k=args.topk)

        if not results:
            print("No outfits generated. Add more wardrobe items.")
            return

        for i, r in enumerate(results, start=1):
            print_outfit(r, i)

    elif args.command == "complete":
        results = complete_outfit(args.item_id, args.event_key, top_k=args.topk)

        if not results:
            print("No matching outfits found.")
            return

        for i, r in enumerate(results, start=1):
            print_outfit(r, i)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
