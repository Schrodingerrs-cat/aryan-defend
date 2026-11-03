# test_aryan recordings: what exists on Aryan's machine vs what is in git

The raw recordings (~24 GB) are too big for GitHub (`test_aryan/.gitignore` ignores `data/` on purpose). `data_small/` here holds only the
small derived files (csv / json / jsonl / yaml / png, ~15 MB, same `<timestamp>/` layout). Everything else is listed below so it is clear what a
re-record or a copy from the original machine would have to bring back. Videos (`*.mp4`, ~0.9 GB) and the `.npy` arrays (~22 GB) are NOT in git.

| recording | total | npy | jpg | mp4 | csv/json/yaml/png |
|---|---|---|---|---|---|
| `20261023_185532` | 10 MB | 0 MB (0) | 0 MB (0) | 10 MB (1) | 0 MB (5) |
| `20261023_185747` | 21 MB | 0 MB (0) | 0 MB (0) | 21 MB (1) | 0 MB (5) |
| `20261023_192028` | 110 MB | 0 MB (0) | 0 MB (0) | 109 MB (2) | 1 MB (5) |
| `20261023_200603` | 30 MB | 0 MB (0) | 0 MB (0) | 30 MB (1) | 0 MB (4) |
| `20261023_201719` | 0 MB | 0 MB (0) | 0 MB (0) | 0 MB (0) | 0 MB (2) |
| `20261023_201826` | 277 MB | 0 MB (0) | 0 MB (0) | 269 MB (4) | 2 MB (11) |
| `20261023_203515` | 2.3 GB | 2.3 GB (1102) | 0 MB (0) | 18 MB (1) | 1 MB (7) |
| `20261023_203914` | 192 MB | 0 MB (0) | 0 MB (0) | 185 MB (3) | 2 MB (11) |
| `20261023_232635` | 2.2 GB | 2.2 GB (1044) | 0 MB (0) | 24 MB (1) | 0 MB (7) |
| `20261023_232738` | 2.2 GB | 2.2 GB (1059) | 0 MB (0) | 19 MB (1) | 0 MB (7) |
| `20261023_232853` | 2.2 GB | 2.2 GB (1070) | 0 MB (0) | 18 MB (1) | 1 MB (7) |
| `20261023_233152` | 1.4 GB | 1.4 GB (669) | 0 MB (0) | 10 MB (1) | 0 MB (7) |
| `20261023_233222` | 501 MB | 498 MB (240) | 0 MB (0) | 4 MB (1) | 0 MB (7) |
| `20261023_233346` | 2.8 GB | 2.8 GB (1342) | 0 MB (0) | 34 MB (1) | 0 MB (7) |
| `20261024_003214` | 4.6 GB | 4.5 GB (2193) | 0 MB (0) | 55 MB (1) | 1 MB (5) |
| `20261024_003622` | 546 MB | 543 MB (262) | 0 MB (0) | 3 MB (1) | 0 MB (5) |
| `20261024_003647` | 2.3 GB | 2.3 GB (1101) | 0 MB (0) | 18 MB (1) | 0 MB (4) |
| `20261024_003747` | 574 MB | 568 MB (274) | 0 MB (0) | 5 MB (1) | 0 MB (5) |
| `20261024_003820` | 821 MB | 815 MB (393) | 0 MB (0) | 6 MB (1) | 0 MB (5) |
| `20261024_094227` | 1.3 GB | 0 MB (0) | 1.2 GB (7974) | 117 MB (2) | 4 MB (9) |
| `20261024_102617` | 19 MB | 0 MB (0) | 17 MB (171) | 1 MB (1) | 0 MB (6) |
| `20261024_103112` | 124 MB | 0 MB (0) | 112 MB (970) | 12 MB (1) | 0 MB (6) |
| `20261024_140937` | 25 MB | 0 MB (0) | 23 MB (148) | 2 MB (1) | 0 MB (6) |
| `20261024_141050` | 34 MB | 0 MB (0) | 31 MB (183) | 3 MB (1) | 0 MB (6) |
| `20261024_141243` | 111 MB | 0 MB (0) | 101 MB (771) | 9 MB (1) | 0 MB (9) |

## Videos kept in git (`test_aryan/videos/`, via LFS)
The three most useful for localisation and the virtual map; the other ~28 videos (~1 GB) are only on Aryan's machine.

| file | what it shows |
|---|---|
| `20261024_094227_localization_viewer.mp4` | latest localisation viewer render (tag detections + estimated pose) of the 094227 flight |
| `20261024_103112_flight_live_map.mp4` | flight that built `maps/live_20261024_103112.yaml` (live map building) |
| `20261024_141243_flight_latest.mp4` | most recent recording (`maps/live_20261024_141243.yaml`) |
