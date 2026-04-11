#!/usr/bin/env python3
"""HuggingFace Hub download statistics tracker.

Tracks daily downloads for all models and datasets owned by a user.
Outputs per-repo and aggregated stats. Appends to a CSV log for historical tracking.

Usage:
    python scripts/hf_download_stats.py                    # default user
    python scripts/hf_download_stats.py --author someone   # specific user
    python scripts/hf_download_stats.py --log              # append to CSV log
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


def get_stats(author: str) -> dict:
    api = HfApi()

    models = list(api.list_models(author=author, sort="downloads"))
    datasets = list(api.list_datasets(author=author, sort="downloads"))

    model_stats = []
    for m in models:
        model_stats.append({
            "id": m.modelId,
            "downloads": m.downloads or 0,
            "likes": m.likes or 0,
            "last_modified": m.lastModified.strftime("%Y-%m-%d") if m.lastModified else "",
        })

    dataset_stats = []
    for d in datasets:
        dataset_stats.append({
            "id": d.id,
            "downloads": d.downloads or 0,
            "likes": d.likes or 0,
            "last_modified": d.lastModified.strftime("%Y-%m-%d") if d.lastModified else "",
        })

    return {
        "models": model_stats,
        "datasets": dataset_stats,
        "total_model_downloads": sum(m["downloads"] for m in model_stats),
        "total_dataset_downloads": sum(d["downloads"] for d in dataset_stats),
        "total_downloads": sum(m["downloads"] for m in model_stats) + sum(d["downloads"] for d in dataset_stats),
        "total_models": len(model_stats),
        "total_datasets": len(dataset_stats),
        "total_likes": sum(m["likes"] for m in model_stats) + sum(d["likes"] for d in dataset_stats),
    }


def print_stats(stats: dict, author: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*65}")
    print(f"  HuggingFace Download Stats: {author}")
    print(f"  {now}")
    print(f"{'='*65}")

    # Models
    print(f"\n  Models ({stats['total_models']})")
    print(f"  {'-'*61}")
    if stats["models"]:
        print(f"  {'Model':<50} {'Downloads':>10}")
        print(f"  {'-'*50} {'-'*10}")
        for m in stats["models"]:
            name = m["id"].replace(f"{author}/", "")
            print(f"  {name:<50} {m['downloads']:>10,}")
        print(f"  {'-'*50} {'-'*10}")
        print(f"  {'TOTAL':<50} {stats['total_model_downloads']:>10,}")
    else:
        print("  (none)")

    # Datasets
    print(f"\n  Datasets ({stats['total_datasets']})")
    print(f"  {'-'*61}")
    if stats["datasets"]:
        print(f"  {'Dataset':<50} {'Downloads':>10}")
        print(f"  {'-'*50} {'-'*10}")
        for d in stats["datasets"]:
            name = d["id"].replace(f"{author}/", "")
            print(f"  {name:<50} {d['downloads']:>10,}")
        print(f"  {'-'*50} {'-'*10}")
        print(f"  {'TOTAL':<50} {stats['total_dataset_downloads']:>10,}")
    else:
        print("  (none)")

    # Aggregated
    print(f"\n  {'='*61}")
    print(f"  {'GRAND TOTAL DOWNLOADS':<50} {stats['total_downloads']:>10,}")
    print(f"  {'TOTAL LIKES':<50} {stats['total_likes']:>10,}")
    print(f"  {'='*61}\n")


def append_log(stats: dict, author: str, log_dir: str) -> None:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    ts = now.strftime("%Y-%m-%d %H:%M:%S")

    # Per-repo log
    repo_file = log_path / "downloads_per_repo.csv"
    write_header = not repo_file.exists()
    with open(repo_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["date", "timestamp", "type", "repo_id", "downloads", "likes"])
        for m in stats["models"]:
            writer.writerow([date_str, ts, "model", m["id"], m["downloads"], m["likes"]])
        for d in stats["datasets"]:
            writer.writerow([date_str, ts, "dataset", d["id"], d["downloads"], d["likes"]])

    # Aggregated daily log
    agg_file = log_path / "downloads_daily.csv"
    write_header = not agg_file.exists()
    with open(agg_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["date", "timestamp", "models", "datasets", "model_downloads",
                             "dataset_downloads", "total_downloads", "total_likes"])
        writer.writerow([
            date_str, ts, stats["total_models"], stats["total_datasets"],
            stats["total_model_downloads"], stats["total_dataset_downloads"],
            stats["total_downloads"], stats["total_likes"],
        ])

    # Update README dashboard
    _write_readme(stats, author, log_path, agg_file)

    print(f"  Logged to: {repo_file}")
    print(f"  Logged to: {agg_file}")
    print(f"  Updated:   {log_path / 'README.md'}")


def _write_readme(stats: dict, author: str, log_path: Path, agg_file: Path) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = []
    w = lines.append

    w(f"# HuggingFace Download Stats")
    w(f"")
    w(f"**Author:** [{author}](https://huggingface.co/{author})")
    w(f"**Last updated:** {now}")
    w(f"**Updated daily by GitHub Actions**")
    w(f"")

    # Summary
    w(f"## Summary")
    w(f"")
    w(f"| Metric | Count |")
    w(f"|--------|------:|")
    w(f"| Models | {stats['total_models']} |")
    w(f"| Datasets | {stats['total_datasets']} |")
    w(f"| Total Downloads | **{stats['total_downloads']:,}** |")
    w(f"| Total Likes | {stats['total_likes']} |")
    w(f"")

    # Models table
    w(f"## Models ({stats['total_models']})")
    w(f"")
    w(f"| # | Model | Downloads | Likes | Last Modified |")
    w(f"|--:|-------|----------:|------:|--------------:|")
    for i, m in enumerate(stats["models"], 1):
        name = m["id"].replace(f"{author}/", "")
        link = f"[{name}](https://huggingface.co/{m['id']})"
        w(f"| {i} | {link} | {m['downloads']:,} | {m['likes']} | {m['last_modified']} |")
    w(f"| | **Total** | **{stats['total_model_downloads']:,}** | | |")
    w(f"")

    # Datasets table
    w(f"## Datasets ({stats['total_datasets']})")
    w(f"")
    w(f"| # | Dataset | Downloads | Likes | Last Modified |")
    w(f"|--:|---------|----------:|------:|--------------:|")
    for i, d in enumerate(stats["datasets"], 1):
        name = d["id"].replace(f"{author}/", "")
        link = f"[{name}](https://huggingface.co/{d['id']})"
        w(f"| {i} | {link} | {d['downloads']:,} | {d['likes']} | {d['last_modified']} |")
    w(f"| | **Total** | **{stats['total_dataset_downloads']:,}** | | |")
    w(f"")

    # Historical trend from CSV
    w(f"## Daily Trend")
    w(f"")
    if agg_file.exists():
        w(f"| Date | Models | Datasets | Model DL | Dataset DL | Total DL | Likes |")
        w(f"|------|-------:|---------:|---------:|-----------:|---------:|------:|")
        with open(agg_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for row in rows[-30:]:  # last 30 days
                w(f"| {row['date']} | {row['models']} | {row['datasets']} "
                  f"| {int(row['model_downloads']):,} | {int(row['dataset_downloads']):,} "
                  f"| {int(row['total_downloads']):,} | {row['total_likes']} |")
    else:
        w(f"*No historical data yet. Check back tomorrow.*")
    w(f"")

    readme_path = log_path / "README.md"
    readme_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="HuggingFace Hub download statistics")
    parser.add_argument("--author", default="ermiaazarkhalili", help="HF Hub username")
    parser.add_argument("--log", action="store_true", help="Append stats to CSV log files")
    parser.add_argument("--log-dir", default="logs/hf_stats", help="Directory for CSV logs")
    args = parser.parse_args()

    stats = get_stats(args.author)
    print_stats(stats, args.author)

    if args.log:
        append_log(stats, args.author, args.log_dir)


if __name__ == "__main__":
    main()
