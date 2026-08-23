CHR-CP IPCCC 2026 code package

This archive contains the runnable CHR-CP implementation, tests, model configurations,
benchmark cache, paper figure utilities, and the completed IPCCC result artifact under
paper_assets/results/.

Install dependencies:
  python -m pip install -r requirements.txt

Run offline checks:
  python -m pytest -q tests/test_benchmark_runner.py::test_gsm8k_loader tests/test_benchmark_runner.py::test_progress_tracker tests/test_confidence.py::test_verbalized_parser tests/test_confidence.py::test_strip_confidence tests/test_l1_routing.py::test_rule_classifier tests/test_l1_routing.py::test_pool_config_consistency tests/test_l2_routing.py::test_budget_thresholds tests/test_l3_cache.py::test_distillation_parser tests/test_new_tier_ladder.py::test_cross_vendor_boundaries
