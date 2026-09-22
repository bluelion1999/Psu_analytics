"""Command-line entry point: `psu ingest` and `psu status`."""
from __future__ import annotations

import argparse
import logging
import sys

from psu import config, db
from psu.build import build
from psu.client import BudgetExceeded, CachedClient, Fetch, MissingApiKey
from psu.ingest import ingest
from psu.train import load_features, report_markdown, train_and_save
from psu.transform import GarbageTime


def make_fetch(settings: config.Settings) -> Fetch:
    if not settings.api_key:

        def missing_key(endpoint, params):
            raise MissingApiKey(
                f"CFBD_API_KEY is not set and {endpoint} {params} is not cached. "
                "Copy .env.example to .env and add your free key from https://collegefootballdata.com/key"
            )

        return missing_key
    from psu.cfbd_api import make_cfbd_fetch  # imported lazily: cfbd is slow to import

    return make_cfbd_fetch(settings.api_key)


def _print_counts(counts: dict[str, int]) -> None:
    for table, n in counts.items():
        print(f"{table:<18} {n:>10,}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="psu", description="Penn State football analytics")
    sub = parser.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest", help="Pull CFBD data into DuckDB (cached; safe to re-run)")
    ing.add_argument("--seasons", help='e.g. "2024", "2022-2026" or "2022,2024-2025" (default: every season)')
    ing.add_argument("--max-calls", type=int, help="Stop before making more than this many API calls")
    sub.add_parser("status", help="Show row counts per table")
    bld = sub.add_parser("build", help="Compute metric tables from ingested data (no API calls)")
    bld.add_argument("--garbage", default="38,28,22", help='Q2,Q3,Q4 garbage-time margins, or "off"')
    bld.add_argument("--alpha", type=float, default=50.0, help="Ridge shrinkage for opponent adjustment")
    trn = sub.add_parser("train", help="Backtest and train the game model; write predictions (no API calls)")
    trn.add_argument("--alpha", type=float, default=50.0, help="Ridge shrinkage for rolling team ratings")
    trn.add_argument("--shrink-plays", type=int, default=300, help="Plays before a season's own data outweighs last season")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = config.load_settings()

    if args.command == "status":
        if settings.db_path.exists():
            con = db.connect(settings.db_path, read_only=True)
            try:
                _print_counts(db.row_counts(con))
            finally:
                con.close()
        else:
            _print_counts({name: 0 for name in db.SPECS})
        return 0

    if args.command == "build":
        try:
            garbage = GarbageTime.parse(args.garbage)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        con = db.connect(settings.db_path)
        try:
            if db.row_counts(con)["plays"] == 0:
                print("error: no plays loaded yet; run `psu ingest` first", file=sys.stderr)
                return 2
            _print_counts(build(con, garbage=garbage, alpha=args.alpha))
        finally:
            con.close()
        return 0

    if args.command == "train":
        con = db.connect(settings.db_path)
        try:
            if db.row_counts(con)["plays"] == 0:
                print("error: no plays loaded yet; run `psu ingest` first", file=sys.stderr)
                return 2
            features = load_features(con, alpha=args.alpha, shrink_plays=args.shrink_plays)
            try:
                report = train_and_save(
                    con, features, current_season=settings.current_season, out_dir=settings.db_path.parent
                )
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2
        finally:
            con.close()
        print(report_markdown(report))
        return 0

    try:
        seasons = config.parse_seasons(
            args.seasons or f"{config.FIRST_SEASON}-{settings.current_season}", last=settings.current_season
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    client = CachedClient(
        settings.raw_dir,
        make_fetch(settings),
        max_calls=settings.max_calls if args.max_calls is None else args.max_calls,
        min_interval_s=settings.min_interval_s,
    )
    con = db.connect(settings.db_path)
    try:
        result = ingest(
            client,
            con,
            seasons,
            current=settings.current_season,
            final_after=settings.final_after,
            refresh_after=settings.refresh_after,
        )
    except MissingApiKey as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except BudgetExceeded as e:
        _print_counts(db.row_counts(con))
        print(f"stopped: {e}", file=sys.stderr)
        return 3
    finally:
        con.close()
    _print_counts(result.row_counts)
    print(f"API calls this run: {result.api_calls}")
    return 0
