#!/usr/bin/env python3
"""Downloads the demo sample images from Wikimedia Commons.

Images land in data/samples/ (git-ignored); licenses and authors are listed
in docs/DATA.md. Re-run any time — downloads are skipped when the file
already exists.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SAMPLES = [
    {
        "file": "dashcam_signal_ahead.jpg",
        "title": "NB Bemiss Rd Traffic light ahead dashcam.jpg",
        "author": "Michael Rivera",
        "license": "CC0",
    },
    {
        "file": "dashcam_intersection.jpg",
        "title": "WB W Alden Ave at Azalea Dr Intersection, Valdosta dashcam.jpg",
        "author": "Michael Rivera",
        "license": "CC BY 4.0",
    },
    {
        "file": "urban_street_pedestrians.jpg",
        "title": "South on Mott, approaching Grand Street (3932229285).jpg",
        "author": "Patrick Nielsen Hayden",
        "license": "CC BY 2.0",
    },
    {
        "file": "traffic_jam.jpg",
        "title": "World Class Traffic Jam 2 (7454479488).jpg",
        "author": "joiseyshowaa",
        "license": "CC BY-SA 2.0",
    },
]

WIDTH = 1600  # server-side downscale keeps the repo demo light


def main() -> None:
    out_dir = Path(__file__).parents[1] / "data" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sample in SAMPLES:
        target = out_dir / sample["file"]
        if target.exists():
            print(f"exists  {target}")
            continue
        quoted = urllib.parse.quote(sample["title"])
        url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{quoted}?width={WIDTH}"
        print(f"fetch   {sample['title']!r} -> {target}")
        request = urllib.request.Request(url, headers={"User-Agent": "traffic-data-engine/0.1"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request) as response:
                    target.write_bytes(response.read())
                break
            except urllib.error.HTTPError as error:
                if error.code == 429 and attempt < 3:  # rate limited: back off and retry
                    wait = 20 * (attempt + 1)
                    print(f"        429, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise
        time.sleep(2)  # be polite to the thumbnail service
    print("done")


if __name__ == "__main__":
    main()
