"""Diagnose: are cache_hit_tokens actually being recorded?"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from rich.console import Console
load_dotenv(PROJECT_ROOT / ".env")

from chr_cp.clients import ClientPool
from chr_cp.prompts import StablePrefixBuilder
from chr_cp.prompts.role_templates import get_role

console = Console()
pool = ClientPool.from_config(PROJECT_ROOT / "configs" / "models.yaml")

# Build a stable prefix and call T1 twice with same prefix
builder = StablePrefixBuilder(task="What is 2+2?")
messages = builder.build_messages(
    role_prompt=get_role("solver").build(),
    step_payload="Compute the sum.",
)

console.print("[bold]Call 1 (cache miss expected):[/bold]")
r1 = pool.invoke(tier="T1", messages=messages, max_tokens=128)
console.print(f"  prompt_tokens: {r1.prompt_tokens}")
console.print(f"  cache_hit_tokens: {r1.cache_hit_tokens}")
console.print(f"  raw response usage: {getattr(r1, 'raw_usage', 'N/A')}")

console.print("\n[bold]Call 2 (cache hit expected):[/bold]")
r2 = pool.invoke(tier="T1", messages=messages, max_tokens=128)
console.print(f"  prompt_tokens: {r2.prompt_tokens}")
console.print(f"  cache_hit_tokens: {r2.cache_hit_tokens}")
console.print(f"  raw response usage: {getattr(r2, 'raw_usage', 'N/A')}")

console.print("\n[bold]Call 3 (cache hit expected):[/bold]")
r3 = pool.invoke(tier="T1", messages=messages, max_tokens=128)
console.print(f"  prompt_tokens: {r3.prompt_tokens}")
console.print(f"  cache_hit_tokens: {r3.cache_hit_tokens}")