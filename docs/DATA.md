# Sample data

The demo images are fetched from Wikimedia Commons by
`scripts/fetch_samples.py` into `data/samples/` (git-ignored). The rendered
detection results in `docs/images/` are derivatives of these works.

| Local file | Source (Wikimedia Commons) | Author | License |
| --- | --- | --- | --- |
| `dashcam_signal_ahead.jpg` | [NB Bemiss Rd Traffic light ahead dashcam.jpg](https://commons.wikimedia.org/wiki/File:NB_Bemiss_Rd_Traffic_light_ahead_dashcam.jpg) | Michael Rivera | CC0 |
| `dashcam_intersection.jpg` | [WB W Alden Ave at Azalea Dr Intersection, Valdosta dashcam.jpg](https://commons.wikimedia.org/wiki/File:WB_W_Alden_Ave_at_Azalea_Dr_Intersection,_Valdosta_dashcam.jpg) | Michael Rivera | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `urban_street_pedestrians.jpg` | [South on Mott, approaching Grand Street (3932229285).jpg](https://commons.wikimedia.org/wiki/File:South_on_Mott,_approaching_Grand_Street_(3932229285).jpg) | Patrick Nielsen Hayden | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| `traffic_jam.jpg` | [World Class Traffic Jam 2 (7454479488).jpg](https://commons.wikimedia.org/wiki/File:World_Class_Traffic_Jam_2_(7454479488).jpg) | joiseyshowaa | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/) |

For larger-scale experiments, point the CLI at your own image folders (e.g.
BDD100K after registering at bdd-data.berkeley.edu).
