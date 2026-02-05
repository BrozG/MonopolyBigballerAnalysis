# Casino Tracker 72h Scraper

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

## Run
```bash
python scrape_last72h.py --url "https://your-tracker-url" --output casino_results_72h.csv
```

## Change URL
Set your target URL with `--url`.

## Optional flags
- `--headed` to run with visible browser.
- `--max-pages 150` to change pagination cap.
- `--verbose` for debug logging.
