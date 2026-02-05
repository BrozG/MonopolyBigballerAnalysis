#!/usr/bin/env python3
"""Scrape last 72 hours of paginated casino game results using Playwright.

Install:
    pip install -r requirements.txt

Run:
    python scrape_last72h.py --url "https://example.com/tracker" --output results.csv

Change URL:
    Replace the value passed to --url.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import pandas as pd
from dateutil import parser as dtparser
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright


LOGGER = logging.getLogger("casino_tracker_scraper")

# Flexible timestamp pattern coverage for common date-time strings.
TS_PATTERN = re.compile(
    r"("  # Capture the full timestamp candidate.
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APMapm]{2})?(?:\s?UTC|[+-]\d{2}:?\d{2})?"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APMapm]{2})?(?:\s?UTC|[+-]\d{2}:?\d{2})?"
    r")"
)
MULTIPLIER_PATTERN = re.compile(r"\b(?:x|X)?\s?(\d+(?:\.\d+)?)\s?[xX]\b|\b(\d+(?:\.\d+)?)\s?(?:multiplier|x)\b")
BONUS_PATTERN = re.compile(r"\b(bonus|free\s?spin|jackpot|boost|feature|wild|scatter)\b", re.IGNORECASE)


@dataclass(frozen=True)
class GameResult:
    timestamp: str
    outcome: str
    bonus_type: Optional[str]
    multiplier: Optional[float]


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def human_delay(min_s: float = 0.8, max_s: float = 2.2) -> None:
    time.sleep(random.uniform(min_s, max_s))


def parse_timestamp(text: str, now_utc: datetime) -> Optional[datetime]:
    match = TS_PATTERN.search(text)
    if not match:
        return None

    candidate = match.group(1)
    try:
        dt = dtparser.parse(candidate)
        # Normalize naive timestamps to local timezone, then convert to UTC.
        if dt.tzinfo is None:
            local_tz = datetime.now().astimezone().tzinfo
            dt = dt.replace(tzinfo=local_tz)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        LOGGER.debug("Failed to parse timestamp candidate: %s", candidate)
        return None


def extract_outcome(text: str) -> str:
    # Drop known timestamp chunk and clean separators.
    cleaned = TS_PATTERN.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" |:-\n\t")
    if not cleaned:
        return "UNKNOWN"

    # Fallback extraction from token patterns like "Result: 23" or "Outcome Red".
    outcome_match = re.search(r"(?:result|outcome|number|win)\s*[:\-]?\s*([\w#./+-]+)", cleaned, flags=re.IGNORECASE)
    if outcome_match:
        return outcome_match.group(1).strip()
    return cleaned[:120]


def extract_bonus(text: str) -> Optional[str]:
    bonus_match = BONUS_PATTERN.search(text)
    return bonus_match.group(1).lower() if bonus_match else None


def extract_multiplier(text: str) -> Optional[float]:
    match = MULTIPLIER_PATTERN.search(text)
    if not match:
        return None
    raw_value = next((g for g in match.groups() if g), None)
    if not raw_value:
        return None
    try:
        return float(raw_value)
    except ValueError:
        return None


def set_72h_filter_if_available(page: Page) -> None:
    LOGGER.info("Checking if a 72-hour timeframe filter is available...")
    candidate_selectors = [
        "button:has-text('72')",
        "button:has-text('72h')",
        "button:has-text('72 hours')",
        "a:has-text('72')",
        "[role='button']:has-text('72')",
        "label:has-text('72')",
    ]

    for selector in candidate_selectors:
        locator = page.locator(selector)
        if locator.count() == 0:
            continue
        for i in range(locator.count()):
            el = locator.nth(i)
            try:
                if el.is_visible(timeout=1000):
                    LOGGER.info("Applying 72-hour filter using selector: %s", selector)
                    el.click(timeout=3000)
                    page.wait_for_load_state("networkidle", timeout=10000)
                    human_delay()
                    return
            except PlaywrightTimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Could not apply selector %s: %s", selector, exc)

    # Fallback: look for a select/dropdown containing 72-hour option.
    for select_selector in ["select", "[role='combobox']"]:
        select_el = page.locator(select_selector)
        if select_el.count() == 0:
            continue
        for i in range(select_el.count()):
            el = select_el.nth(i)
            try:
                if not el.is_visible(timeout=1000):
                    continue
                options_text = el.inner_text().lower()
                if "72" in options_text:
                    LOGGER.info("Applying 72-hour filter via dropdown")
                    el.select_option(label=re.compile(r"72", re.IGNORECASE))
                    page.wait_for_load_state("networkidle", timeout=10000)
                    human_delay()
                    return
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Dropdown fallback failed: %s", exc)

    LOGGER.info("No explicit 72-hour filter found; continuing with existing timeframe.")


def get_result_containers(page: Page):
    # Generic containers that often represent rows/items in dynamic lists.
    selectors = ["tr", "[role='row']", "li", "article", "div[class*='row']", "div[class*='item']"]
    seen_handles = []
    for selector in selectors:
        loc = page.locator(selector)
        count = min(loc.count(), 500)
        for idx in range(count):
            handle = loc.nth(idx)
            try:
                if handle.is_visible(timeout=200):
                    seen_handles.append(handle)
            except Exception:
                continue
    return seen_handles


def parse_current_page_results(page: Page, cutoff_utc: datetime, now_utc: datetime) -> tuple[list[GameResult], bool]:
    rows = get_result_containers(page)
    page_results: list[GameResult] = []
    found_older_than_cutoff = False

    for row in rows:
        try:
            text = row.inner_text(timeout=1000)
        except Exception:
            continue

        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 8:
            continue

        ts = parse_timestamp(text, now_utc)
        if not ts:
            continue

        if ts < cutoff_utc:
            found_older_than_cutoff = True
            continue

        outcome = extract_outcome(text)
        bonus_type = extract_bonus(text)
        multiplier = extract_multiplier(text)

        page_results.append(
            GameResult(
                timestamp=ts.isoformat(),
                outcome=outcome,
                bonus_type=bonus_type,
                multiplier=multiplier,
            )
        )

    return page_results, found_older_than_cutoff


def click_next_if_possible(page: Page) -> bool:
    next_selectors = [
        "button:has-text('Next')",
        "a:has-text('Next')",
        "[role='button']:has-text('Next')",
        "button[aria-label*='Next' i]",
        "a[aria-label*='Next' i]",
        "[rel='next']",
    ]

    for selector in next_selectors:
        next_btn = page.locator(selector)
        if next_btn.count() == 0:
            continue

        for i in range(next_btn.count()):
            btn = next_btn.nth(i)
            try:
                if not btn.is_visible(timeout=800):
                    continue

                aria_disabled = (btn.get_attribute("aria-disabled") or "").lower() == "true"
                class_name = (btn.get_attribute("class") or "").lower()
                is_disabled_css = "disabled" in class_name

                disabled = False
                try:
                    disabled = btn.is_disabled()
                except Exception:
                    pass

                if aria_disabled or is_disabled_css or disabled:
                    LOGGER.info("Next button exists but is disabled.")
                    return False

                LOGGER.info("Clicking Next page...")
                btn.click(timeout=5000)
                page.wait_for_load_state("networkidle", timeout=15000)
                human_delay(1.0, 2.8)
                return True
            except PlaywrightTimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Failed clicking next for selector %s: %s", selector, exc)

    LOGGER.info("No usable Next button found.")
    return False


def dedupe_results(results: Iterable[GameResult]) -> list[GameResult]:
    seen = set()
    unique = []
    for item in results:
        key = (item.timestamp, item.outcome, item.bonus_type, item.multiplier)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def scrape_last_72h(url: str, output_csv: str, headless: bool = True, max_pages: int = 100) -> pd.DataFrame:
    now_utc = datetime.now(timezone.utc)
    cutoff_utc = now_utc - timedelta(hours=72)
    LOGGER.info("Starting scrape. Cutoff UTC timestamp: %s", cutoff_utc.isoformat())

    collected: list[GameResult] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_load_state("networkidle", timeout=20000)
            human_delay(1.0, 2.5)

            set_72h_filter_if_available(page)

            for page_no in range(1, max_pages + 1):
                LOGGER.info("Parsing page %s", page_no)

                page_results, older_than_cutoff_seen = parse_current_page_results(page, cutoff_utc, now_utc)
                collected.extend(page_results)
                LOGGER.info("Collected %d rows from page %d", len(page_results), page_no)

                if older_than_cutoff_seen:
                    LOGGER.info("Encountered records older than 72 hours; stopping pagination.")
                    break

                if not click_next_if_possible(page):
                    LOGGER.info("Pagination finished (no next page).")
                    break

        except PlaywrightTimeoutError as exc:
            LOGGER.error("Timeout while scraping: %s", exc)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Unexpected error during scraping: %s", exc)
        finally:
            context.close()
            browser.close()

    unique_results = dedupe_results(collected)
    df = pd.DataFrame([asdict(r) for r in unique_results], columns=["timestamp", "outcome", "bonus_type", "multiplier"])

    if not df.empty:
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    df.to_csv(output_csv, index=False)
    LOGGER.info("Saved %d unique rows to %s", len(df), output_csv)

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape last 72 hours of casino game results.")
    parser.add_argument("--url", required=True, help="Target tracker URL.")
    parser.add_argument("--output", default="casino_results_72h.csv", help="Output CSV path.")
    parser.add_argument("--headed", action="store_true", help="Run with visible browser (non-headless).")
    parser.add_argument("--max-pages", type=int, default=100, help="Safety cap for pagination depth.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    LOGGER.info("URL: %s", args.url)
    LOGGER.info("Output CSV: %s", args.output)

    df = scrape_last_72h(
        url=args.url,
        output_csv=args.output,
        headless=not args.headed,
        max_pages=args.max_pages,
    )

    LOGGER.info("DataFrame is available in memory as variable `df` in this runtime context.")
    LOGGER.info("Final record count: %d", len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
