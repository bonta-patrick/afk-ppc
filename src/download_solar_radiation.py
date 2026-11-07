
from pathlib import Path

import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
PARAMS = {
    "latitude": -33.8688,   # Sydney
    "longitude": 151.2093,
    "start_date": "2010-01-01",
    "end_date": "2013-12-31",
    "hourly": "shortwave_radiation",  # W/m², medie pe ora precedentă
    "timezone": "Australia/Sydney",
}

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "sydney_shortwave_radiation_2010-2013.csv"


def main() -> None:
    resp = requests.get(URL, params=PARAMS, timeout=120)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "shortwave_radiation": hourly["shortwave_radiation"],
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Salvat {len(df)} rânduri în {OUT}")


if __name__ == "__main__":
    main()
