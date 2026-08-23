# CHR-CP Paper Assets

This directory contains the IPCCC paper figures and the completed CHR-CP experiment result summary.

## Generated Figures

Run:

```bash
python paper_assets/generate_figures.py
```

Outputs:

| File | Suggested use |
|---|---|
| `figures/figure_01_system_architecture.svg` | Method overview figure |
| `figures/figure_02_pricai_positioning.svg` | Introduction / motivation positioning |
| `figures/figure_03_cost_accuracy_map.svg` | Cost-accuracy analysis |
| `figures/figure_04_phase1b_sensitivity.svg` | Sensitivity analysis |
| `figures/figure_05_benchmark_dashboard.svg` | Current result dashboard |
| `figures/figure_06_decision_trace_template.svg` | Case-study trace template |
| `figures/figure_07_tier_ladder_cache.svg` | Model ladder and cache boundary figure |
| `figures/figure_08_fuzzing_reframe_bridge.svg` | Reframing risk / fuzzing bridge figure |

`figure_contact_sheet.html` previews all SVG figures in one page.

## Data Caveat

`data/results_summary.json` is the structured aggregate of the completed CHR-CP experiments used by the manuscript tables and figures. It is kept with the artifact so the reported values and plots can be regenerated deterministically.
