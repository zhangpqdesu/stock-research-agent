from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import sleep

import pandas as pd
import tushare as ts
from dotenv import load_dotenv
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]


def call_with_retry(fn, retries: int = 5, sleep_seconds: int = 2):
    last_error = None
    for i in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            sleep(sleep_seconds * (i + 1))
    raise last_error


def build_index_weight_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    ranges = []
    current = start
    while current <= end:
        chunk_end = min(current + pd.Timedelta(days=179), end)
        ranges.append((current.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        current = chunk_end + pd.Timedelta(days=1)
    return ranges


def resolve_hs300_codes(pro, index_code: str, start_date: str, end_date: str) -> list[str]:
    weights = []
    for start, end in tqdm(build_index_weight_ranges(start_date, end_date), desc="index_weight", unit="chunk"):
        df = call_with_retry(
            lambda s=start, e=end: pro.index_weight(
                index_code=index_code,
                start_date=s,
                end_date=e,
            )
        )
        if df is not None and not df.empty:
            weights.append(df)

    if not weights:
        raise RuntimeError("No HS300 constituents returned from index_weight.")

    weights = pd.concat(weights, ignore_index=True).drop_duplicates()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    latest_date = weights["trade_date"].max()
    latest_weights = weights.loc[weights["trade_date"].eq(latest_date)].copy()
    return sorted(latest_weights["con_code"].dropna().unique().tolist())


def download_per_code(pro, codes: list[str], out_dir: Path, start_date: str, end_date: str, api_name: str) -> list[tuple[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for ts_code in tqdm(codes, desc=api_name, unit="stock"):
        out_path = out_dir / f"{ts_code.replace('.', '_')}.csv"
        if out_path.exists():
            continue
        try:
            if api_name == "daily":
                df = call_with_retry(
                    lambda c=ts_code: pro.daily(
                        ts_code=c,
                        start_date=start_date,
                        end_date=end_date,
                    )
                )
            elif api_name == "daily_basic":
                df = call_with_retry(
                    lambda c=ts_code: pro.daily_basic(
                        ts_code=c,
                        start_date=start_date,
                        end_date=end_date,
                        fields=(
                            "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
                            "pe,pb,ps_ttm,dv_ttm,total_mv,circ_mv"
                        ),
                    )
                )
            else:
                raise ValueError(f"Unsupported api_name: {api_name}")

            if df is None:
                df = pd.DataFrame()
            df.to_csv(out_path, index=False)
        except Exception as exc:  # noqa: BLE001
            failed.append((ts_code, str(exc)))
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HS300 daily and daily_basic history with tqdm progress bars.")
    parser.add_argument("--start-date", default="20240101")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"))
    parser.add_argument("--index-code", default="000300.SH")
    parser.add_argument("--out-dir", default=str(ROOT / "hs300_cache"))
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Missing TUSHARE_TOKEN. Put it in .env or export it in your shell.")

    # Tushare often needs direct access outside a proxy/VPN path.
    for key in ("NO_PROXY", "no_proxy"):
        current = os.getenv(key, "").strip()
        parts = [item for item in current.split(",") if item]
        for host in ("api.waditu.com", "waditu.com"):
            if host not in parts:
                parts.append(host)
        os.environ[key] = ",".join(parts)

    ts.set_token(token)
    pro = ts.pro_api(token)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = resolve_hs300_codes(pro, args.index_code, args.start_date, args.end_date)
    print(f"Resolved {len(codes)} HS300 constituents from {args.index_code}.")

    failed_daily = download_per_code(pro, codes, out_dir / "daily", args.start_date, args.end_date, "daily")
    failed_basic = download_per_code(pro, codes, out_dir / "daily_basic", args.start_date, args.end_date, "daily_basic")

    pd.DataFrame(failed_daily, columns=["ts_code", "error"]).to_csv(out_dir / "failed_daily.csv", index=False)
    pd.DataFrame(failed_basic, columns=["ts_code", "error"]).to_csv(out_dir / "failed_daily_basic.csv", index=False)

    print("Download completed.")
    print(f"daily failed: {len(failed_daily)}")
    print(f"daily_basic failed: {len(failed_basic)}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
