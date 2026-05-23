# CHR-CP: Confidence-Gated Hierarchical Routing with Cache-Preserved Switching

Implementation of CHR-CP for heterogeneous multi-agent LLM routing under API constraints.

## Day 1 Setup

### Environment

```bash
# Create conda env (if not yet)
conda create -n mas_chrcp python=3.11 -y
conda activate mas_chrcp

# Install dependencies
pip install -r requirements.txt
```

### Configure API Keys

```bash
# Copy template and fill in your keys
cp .env.example .env
# Edit .env with your DeepSeek and Qwen keys
```

### Verify

```bash
python -m tests.test_clients
```

If all 4 tiers print ✓, environment is ready.

## Model Pool

| Tier | Model | Provider | Mode | Role |
|------|-------|----------|------|------|
| T4 (top) | deepseek-v4-pro | DeepSeek | thinking | Critical aggregation, ESCALATE target |
| T3 (strong) | qwen-max | Alibaba | non-thinking | Mid-level reasoning, cross-vendor |
| T2 (mid) | deepseek-v4-flash | DeepSeek | thinking | Routine reasoning steps |
| T1 (weak) | deepseek-v4-flash | DeepSeek | non-thinking | Compression, simple subtasks |

## Project Structure

See main paper for full architecture (CHR-CP framework).