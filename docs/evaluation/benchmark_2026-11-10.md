# Benchmark (pilot), 2026-11-10

One machine, CPU only. **Synthetic scale:** the large images are the 820x1024 portrait fixture enlarged, not real photos. Latency is `segment_face` end to end excluding HTTP; decode is `load_rgb_image` on a q90 JPEG.

- Python 3.13.7, packages {'numpy': '2.5.4', 'onnxruntime': '1.31.1', 'mediapipe': '1.2.2', 'Pillow': '13.0.0'}
- macOS-15.6-arm64-arm-64bit-Mach-O, Apple M2 Pro, OpenCV threads 10
- Power at start: drawing from 'Battery Power'. Sleep inside the window: 0 sleep event(s) between 23:16:02 and 23:19:13.
- Source fingerprint `825baa49cda1420c6b4df6ae821487a12629021c54c226d9924a296dee7468ef`; ru_maxrss unit on this machine: bytes (Darwin) (converted to MB)

## Stage latency (1 discarded + 30 timed runs; ms, p50 / p95 / p99)

| size | pixels | decode | landmarks | preprocess | inference | postprocess | zones | encode | total |
|---|---|---|---|---|---|---|---|---|---|
| 0.8 MP | 799,200 | 3.2 / 3.6 / 5.3 | 8.7 / 9.0 / 10.3 | 4.5 / 5.0 / 5.2 | 82.9 / 85.9 / 98.7 | 23.7 / 24.6 / 24.8 | 12.3 / 13.0 / 13.7 | 4.2 / 4.6 / 8.4 | 140.6 / 146.9 / 157.7 |
| 2.0 MP | 1,998,700 | 7.9 / 9.1 / 9.5 | 9.2 / 9.8 / 9.9 | 4.6 / 4.8 / 5.0 | 82.1 / 83.9 / 84.4 | 29.8 / 30.9 / 31.7 | 29.0 / 30.2 / 30.8 | 9.5 / 9.9 / 10.0 | 172.8 / 175.8 / 185.1 |
| 10.0 MP | 9,994,857 | 36.3 / 40.8 / 42.3 | 12.8 / 13.8 / 14.6 | 8.2 / 8.7 / 9.0 | 82.1 / 83.4 / 84.9 | 39.7 / 40.4 / 41.1 | 147.2 / 153.6 / 154.8 | 45.7 / 47.6 / 48.0 | 374.0 / 379.5 / 381.8 |
| 25.0 MP | 24,996,238 | 88.4 / 101.8 / 107.0 | 15.8 / 17.7 / 18.2 | 12.1 / 13.4 / 14.3 | 82.2 / 88.2 / 92.2 | 45.4 / 49.5 / 50.4 | 366.6 / 380.5 / 386.2 | 113.0 / 118.1 / 119.7 | 728.1 / 746.3 / 752.4 |

## Concurrency and peak memory (each row a fresh process; parser max concurrency 2 as in the app)

Peak RSS is of the whole process (includes the MediaPipe landmarker, about 200 MB). Latency is per request including time waiting for a parser slot.

| size | threads | requests | failures | throughput (req/s) | latency p50 / p95 (ms) | peak RSS (MB) |
|---|---|---|---|---|---|---|
| 2.0 MP | 1 | 6 | 0 | 5.67 | 176 / 179 | 680 |
| 2.0 MP | 2 | 12 | 0 | 7.77 | 242 / 285 | 713 |
| 2.0 MP | 4 | 24 | 0 | 8.15 | 481 / 485 | 736 |
| 25.0 MP | 1 | 6 | 0 | 1.31 | 759 / 779 | 1521 |
| 25.0 MP | 2 | 12 | 0 | 2.50 | 775 / 851 | 1880 |
| 25.0 MP | 4 | 24 | 0 | 3.88 | 959 / 1168 | 2214 |

## Label-map resolution cap vs agreement (dev split, 239 detected images)

The label map is downscaled (nearest) to the given long side and scaled back before scoring against EasyPortrait annotations (dataset-level IoU, full image). This only tests the output resolution; the production parser already computes logits at a working size of at most 1024. EasyPortrait images are mostly 1.4-1.9k px, so a 2048 cap changes nothing for them (it matters for large phone photos, not tested here).

| cap (long side) | images changed | skin | lips | eye | brow |
|---|---|---|---|---|---|
| none | 0 | 0.8624 | 0.7912 | 0.7087 | 0.7003 |
| 2048 | 29 | 0.8627 | 0.7933 | 0.7109 | 0.7027 |
| 1024 | 180 | 0.8624 | 0.7968 | 0.7150 | 0.7099 |

## Limitations

- Single machine and a single face upscaled; real large photos may differ.
- Concurrent runs use 6 requests per thread; peak RSS is a maximum, not a distribution.
- The cap experiment cannot show the memory or latency benefit of a real cap, only its agreement cost.
