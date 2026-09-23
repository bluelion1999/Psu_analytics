"""Command-line entry point: `psu ingest`, `psu status`, `psu build`, `psu train` and `psu simulate`."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable

import duckdb

from psu import config, db
from psu.build import build
from psu.client import BudgetExceeded, CachedClient, Fetch, MissingApiKey
from psu.ingest import IngestResult, ingest
from psu.simulate import MissingModel, SimResult, load_sigma, simulate_season, write_results
from psu.simulate import report_markdown as sim_report
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


def _no_plays(con: duckdb.DuckDBPyConnection) -> bool:
    if db.row_counts(con)["plays"] == 0:
        print("error: no plays loaded yet; run `psu ingest` first", file=sys.stderr)
        return True
    return False


def cmd_status(args: argparse.Namespace, settings: config.Settings) -> int:
    if settings.db_path.exists():
        con = db.connect(settings.db_path, read_only=True)
        try:
            _print_counts(db.row_counts(con))
        finally:
            con.close()
    else:
        _print_counts({name: 0 for name in db.SPECS})
    return 0


def _ingest(settings: config.Settings, seasons: list[int], max_calls: int | None) -> tuple[int, IngestResult | None]:
    client = CachedClient(
        settings.raw_dir,
        make_fetch(settings),
        max_calls=settings.max_calls if max_calls is None else max_calls,
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
        return 2, None
    except BudgetExceeded as e:
        _print_counts(db.row_counts(con))
        print(f"stopped: {e}", file=sys.stderr)
        return 3, None
    finally:
        con.close()
    _print_counts(result.row_counts)
    print(f"API calls this run: {result.api_calls}")
    return 0, result


def cmd_ingest(args: argparse.Namespace, settings: config.Settings) -> int:
    try:
        seasons = config.parse_seasons(
            args.seasons or f"{config.FIRST_SEASON}-{settings.current_season}", last=settings.current_season
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return _ingest(settings, seasons, args.max_calls)[0]


def cmd_build(args: argparse.Namespace, settings: config.Settings) -> int:
    try:
        garbage = GarbageTime.parse(args.garbage)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    con = db.connect(settings.db_path)
    try:
        if _no_plays(con):
            return 2
        _print_counts(build(con, garbage=garbage, alpha=args.alpha))
    finally:
        con.close()
    return 0


def cmd_train(args: argparse.Namespace, settings: config.Settings) -> int:
    con = db.connect(settings.db_path)
    try:
        if _no_plays(con):
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


def _simulate(
    settings: config.Settings, *, team: str, n_sims: int, seed: int, tau: float
) -> tuple[int, SimResult | None]:
    try:
        sigma = load_sigma(settings.db_path.parent)
        con = db.connect(settings.db_path)
        try:
            result = simulate_season(
                con, season=settings.current_season, sigma=sigma, team=team, n_sims=n_sims, seed=seed, tau=tau
            )
            write_results(con, result, settings.db_path.parent)
        finally:
            con.close()
    except (MissingModel, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2, None
    print(sim_report(result))
    return 0, result


def cmd_simulate(args: argparse.Namespace, settings: config.Settings) -> int:
    return _simulate(settings, team=args.team, n_sims=args.sims, seed=args.seed, tau=args.tau)[0]


HANDLERS: dict[str, Callable[[argparse.Namespace, config.Settings], int]] = {
    "status": cmd_status,
    "ingest": cmd_ingest,
    "build": cmd_build,
    "train": cmd_train,
    "simulate": cmd_simulate,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="psu", description="Penn State football analytics")
    sub = parser.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest", help="Pull CFBD data into DuckDB (cached; safe to re-run)")
    ing.add_argument("--seasons", help='e.g. "2024", "2022-2026" or "2022,2024-2025" (default: every season)')
    ing.add_argument("--max-calls", type=int, help="Stop before making more than this many API calls")
    sub.add_parser("status", help="Show row counts per table")
    bld = sub.add_parser("build", help="Compute metric tables from ingested data (no API calls)")
    bld.add_argument("--garbage", default=config.GARBAGE, help='Q2,Q3,Q4 garbage-time margins, or "off"')
    bld.add_argument("--alpha", type=float, default=config.BUILD_ALPHA, help="Ridge shrinkage for opponent adjustment")
    trn = sub.add_parser("train", help="Backtest and train the game model; write predictions (no API calls)")
    trn.add_argument("--alpha", type=float, default=config.TRAIN_ALPHA, help="Ridge shrinkage for rolling team ratings")
    trn.add_argument(
        "--shrink-plays",
        type=int,
        default=config.SHRINK_PLAYS,
        help="Plays before a season's own data outweighs last season",
    )
    sim = sub.add_parser("simulate", help="Simulate the rest of the season (no API calls)")
    sim.add_argument("--sims", type=int, default=config.SIM_N, help="Number of simulated seasons")
    sim.add_argument("--seed", type=int, default=config.SIM_SEED, help="Random seed (same seed, same results)")
    sim.add_argument(
        "--tau", type=float, default=config.SIM_TAU, help="Spread (points) of each team's season-long strength draw"
    )
    sim.add_argument("--team", default=config.TEAM, help="Team to report on")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = config.load_settings()
    try:
        return HANDLERS[args.command](args, settings)
    except duckdb.IOException as e:
        print(
            f"error: {settings.db_path} is in use by another process; close it and retry ({e})",
            file=sys.stderr,
        )
        return 2
