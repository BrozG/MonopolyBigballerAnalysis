# 🎰 Monopoly Big Baller — 72h Results Scraper

> A vibe-coded Python scraper that tracks and collects the last 72 hours of Monopoly Big Baller casino game results for analysis.

---

## 🤔 Why Does This Exist?

Monopoly Big Baller results are shown live on tracker sites but there's no easy way to download or analyze the data. This scraper grabs the last 72 hours of results and saves them to a CSV so you can analyze patterns, roll frequencies, bonus round triggers and more.

> ⚠️ **Disclaimer:** This is a data analysis tool only. Always gamble responsibly.

---

## ✨ Features

- 🕐 Scrapes last **72 hours** of game results automatically
- 📄 Exports clean data to **CSV** for analysis
- 🌐 Uses **Playwright** for reliable browser automation
- 📑 Handles **pagination** automatically (up to 150 pages)
- 🐛 Optional **verbose/debug** mode

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Playwright | Browser automation & scraping |
| Chromium | Headless browser |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/BrozG/MonopolyBigballerAnalysis
cd MonopolyBigballerAnalysis
```

### 2. Set up virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
playwright install chromium
```

### 4. Run the scraper
```bash
python scrape_last72h.py --url "https://your-tracker-url" --output casino_results_72h.csv
```

---

## ⚙️ Options

| Flag | Description | Default |
|---|---|---|
| `--url` | Target tracker URL | Required |
| `--output` | Output CSV filename | `results.csv` |
| `--headed` | Run with visible browser window | Headless |
| `--max-pages` | Max pages to paginate through | 150 |
| `--verbose` | Enable debug logging | Off |

---

## 📊 Output

The scraper saves results to a `.csv` file that you can open in Excel, Google Sheets, or pandas for analysis:

```
timestamp, result, multiplier, bonus_round, ...
```

---

## 📄 License

MIT
