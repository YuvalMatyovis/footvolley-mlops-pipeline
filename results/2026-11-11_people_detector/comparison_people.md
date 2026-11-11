# People detector comparison

| metric | yolo11n | rfdetr-nano | rfdetr-small |
| :--- | ---: | ---: | ---: |
| frames | 3465 | 3465 | 3465 |
| avg players per frame | 3.31 | 3.26 | 3.29 |
| frames with >= 3 players % | 83 | 83 | 83 |
| frames with exactly 4 players % | 35 | 39 | 41 |
| count flicker % (lower = steadier) | 13.0 | 6.9 | 6.4 |
| seat occupancy % (S1 / S2 / S3 / S4) | 81 / 61 / 94 / 84 | 67 / 78 / 94 / 84 | 67 / 80 / 95 / 84 |
| people model s/frame | 0.016 | 0.022 | 0.031 |
| **passes vs manual** (window 0.5s, offset 0.1s) |  |  |  |
| system passes / manual | 47 / 46 | 44 / 46 | 35 / 46 |
| matched | 27 | 25 | 26 |
| missed | 19 | 21 | 20 |
| invented | 20 | 19 | 9 |
| receiver correct | 12 | 10 | 11 |
| who-to-whom correct | 3 | 4 | 7 |

Runs: rally_56s_yolo11n_2026-11-11_1208, rally_56s_rfdetr-nano_2026-11-11_1214, rally_56s_rfdetr-small_2026-11-11_1219
