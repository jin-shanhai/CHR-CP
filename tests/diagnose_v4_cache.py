"""Diagnose: does deepseek-v4-flash actually cache prompts?

We test with progressively longer prompts and inspect the raw usage object
to discover the real cache field name (if any).
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from rich.console import Console
load_dotenv(PROJECT_ROOT / ".env")

from chr_cp.clients import ClientPool

console = Console()
pool = ClientPool.from_config(PROJECT_ROOT / "configs" / "models.yaml")

# Build a long prompt (~500 tokens) to ensure we cross cache block boundaries
LONG_PROMPT = """You are a helpful assistant participating in a multi-step reasoning task.
Always be precise, concise, and explain your reasoning step by step.
When solving math problems, identify the operation, perform it carefully,
and double-check by reverse computation. Never guess.

Problem context: We are building a multi-agent system that routes tasks
to different AI models based on their difficulty and the cost of each model.
The system needs to make decisions at every step about whether to continue
with the current model, branch into multiple samples for voting, or escalate
to a stronger model. Each decision is informed by an uncertainty signal
computed from the model's verbalized confidence and the consistency of
multiple sampled responses.

Your role: Solve the following problem step by step.

Question: What is 13 + 27?"""

console.rule("[bold cyan]Diagnose V4-Flash Cache Behavior[/bold cyan]")

for tier in ["T2", "T3"]:
    console.print(f"\n[bold]== Tier {tier} ({pool.configs[tier]['model_id']}) ==[/bold]")
    
    for i in range(3):
        response = pool.invoke(
            tier=tier,
            messages=[{"role": "user", "content": LONG_PROMPT}],
            temperature=0.0,
            max_tokens=64,
        )
        
        # Extract everything we can about cache from the raw response
        usage = response.raw_response.usage
        usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else vars(usage)
        
        console.print(f"  Call {i+1}:")
        console.print(f"    prompt_tokens     = {response.prompt_tokens}")
        console.print(f"    completion_tokens = {response.completion_tokens}")
        console.print(f"    cache_hit (parsed)= {response.cache_hit_tokens}")
        console.print(f"    raw usage dict    = {usage_dict}")
        console.print()