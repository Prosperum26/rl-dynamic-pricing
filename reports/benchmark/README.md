# Benchmark reports

Thư mục chứa báo cáo đánh giá chuẩn hóa (CSV + Markdown).

## Chạy benchmark

```bash
python -m scripts.run_benchmark
```

Tùy chọn:

```bash
python -m scripts.run_benchmark --pricing-episodes 30 --tag my_run
python -m scripts.run_benchmark --skip-pricing   # chỉ demand + simulator
```

## Đầu ra

Mỗi lần chạy tạo subfolder `YYYYMMDD_HHMMSS/` (hoặc `--tag`):

| File | Mô tả |
|------|--------|
| `benchmark_results.csv` | Bảng dài: mọi metric, model, category |
| `demand_benchmark.csv` | Demand forecasting |
| `pricing_benchmark.csv` | PPO vs baselines trong simulator |
| `simulator_benchmark.csv` | Độ nhạy demand theo giá |
| `BENCHMARK_REPORT.md` | Báo cáo tổng hợp (tiếng Việt) + bảng |
| `REFERENCES.md` | Trích dẫn nguồn benchmark |
| `run_metadata.json` | Metadata lần chạy |

File `BENCHMARK_REPORT.md` và `benchmark_results.csv` ở thư mục gốc `reports/benchmark/` là bản copy từ lần chạy mới nhất (`LATEST_RUN.txt` ghi đường dẫn đầy đủ).

## Nguồn benchmark

Định nghĩa trong `benchmarks/standards.py` — gồm Hyndman & Athanasopoulos (FPP3), M5/Makridakis (WMAPE), Ferreira et al. (dynamic pricing baselines), Sutton & Barto (episodic evaluation), Stable-Baselines3, và ngưỡng MVP nội bộ project.
