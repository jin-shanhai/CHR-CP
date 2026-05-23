"""Day 1 verification: ensure all 4 tiers can be invoked successfully.

Run with:
    python -m tests.test_clients
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from chr_cp.clients import ClientPool, Tier
from chr_cp.prompts import StablePrefixBuilder
from chr_cp.utils import CostTracker


console = Console()


def test_basic_invocation():
    """Test 1: each of the 4 tiers can answer a simple question."""
    console.rule("[bold cyan]Test 1: Basic Invocation Across 4 Tiers[/bold cyan]")
    
    config_path = PROJECT_ROOT / "configs" / "models.yaml"
    pool = ClientPool.from_config(config_path)
    tracker = CostTracker()
    
    test_prompt = "请用一句话回答:1+1等于几?"
    messages = [{"role": "user", "content": test_prompt}]
    
    table = Table(title="4-Tier Invocation Results")
    table.add_column("Tier", style="cyan", no_wrap=True)
    table.add_column("Model", style="magenta")
    table.add_column("Output", style="green")
    table.add_column("Tokens", justify="right")
    table.add_column("Cache Hit", justify="right")
    table.add_column("Cost", justify="right", style="yellow")
    table.add_column("Latency", justify="right")
    table.add_column("Status", style="bold")
    
    results = {}
    for tier_name in pool.list_tiers():
        try:
            response = pool.invoke(
                tier=tier_name,
                messages=messages,
                temperature=0.3,
                max_tokens=128,
            )
            tracker.record(response, task_id="test_basic", benchmark="warmup")
            
            cache_str = (
                f"{response.cache_hit_tokens}/{response.prompt_tokens}"
                if response.cache_hit_tokens is not None
                else "N/A"
            )
            
            table.add_row(
                tier_name,
                response.model_id,
                response.content[:50].replace("\n", " ") + ("..." if len(response.content) > 50 else ""),
                f"{response.prompt_tokens}+{response.completion_tokens}",
                cache_str,
                f"${response.cost_usd:.6f}",
                f"{response.latency_seconds:.2f}s",
                "[green]✓[/green]",
            )
            results[tier_name] = True
        except Exception as e:
            table.add_row(
                tier_name, "—", str(e)[:60], "—", "—", "—", "—",
                "[red]✗[/red]",
            )
            results[tier_name] = False
    
    console.print(table)
    return results, tracker


def test_stable_prefix():
    """Test 2: stable prefix engineering achieves cache hit on second call.
    
    Strategy:
    - Build a stable prefix with the same system+task
    - Call T1 twice with the SAME prefix (different step payload)
    - Second call should have cache_hit > 0 on DeepSeek
    """
    console.rule("[bold cyan]Test 2: Stable Prefix Cache Behavior (T1)[/bold cyan]")
    
    config_path = PROJECT_ROOT / "configs" / "models.yaml"
    pool = ClientPool.from_config(config_path)
    
    # Build a long enough prefix to trigger DeepSeek caching
    # (DeepSeek caches prefixes of 64+ tokens typically)
    long_task = (
        "Analyze the following arithmetic problems carefully. "
        "For each problem, identify the operation type, perform the calculation, "
        "and verify the result by reverse computation. "
        "Make sure to show all intermediate steps clearly. "
        "Pay special attention to operator precedence and any edge cases. "
        "If a problem involves division, check for zero denominators. "
        "If a problem involves negative numbers, ensure correct sign handling. "
        "Your output must be in JSON format with fields: type, steps, answer, verification."
    )
    
    builder = StablePrefixBuilder(task=long_task)
    
    table = Table(title="Stable Prefix Cache Hit Test")
    table.add_column("Call", style="cyan")
    table.add_column("Step Payload", style="magenta")
    table.add_column("Prompt Tokens", justify="right")
    table.add_column("Cache Hit", justify="right", style="yellow")
    table.add_column("Status", style="bold")
    
    payloads = [
        "Problem 1: 25 + 17 = ?",
        "Problem 2: 84 - 39 = ?",  # different payload but SAME prefix
    ]
    
    cache_hit_history = []
    for i, payload in enumerate(payloads):
        messages = builder.build_messages(
            role_prompt="You are an arithmetic verification agent.",
            step_payload=payload,
        )
        
        try:
            response = pool.invoke(
                tier=Tier.T1,
                messages=messages,
                max_tokens=200,
            )
            cache_hit = response.cache_hit_tokens or 0
            cache_hit_history.append(cache_hit)
            
            status = "[green]✓[/green]"
            if i == 1 and cache_hit == 0:
                status = "[yellow]⚠ no cache hit on 2nd call[/yellow]"
            
            table.add_row(
                f"Call {i+1}",
                payload[:30],
                str(response.prompt_tokens),
                str(cache_hit),
                status,
            )
        except Exception as e:
            table.add_row(f"Call {i+1}", payload[:30], "—", "—", f"[red]✗ {e}[/red]")
            return False
    
    console.print(table)
    
    # Validate: second call should have cache hit
    if len(cache_hit_history) >= 2 and cache_hit_history[1] > 0:
        console.print(
            f"[green]✓ Cache mechanism working: {cache_hit_history[1]} tokens hit on 2nd call[/green]"
        )
        return True
    else:
        console.print(
            "[yellow]⚠ Cache hit not observed. This may be normal on very short prefixes "
            "or if DeepSeek cache hasn't propagated yet. Try running again in 1 minute.[/yellow]"
        )
        return False


def test_cost_tracking():
    """Test 3: cost tracker accurately accumulates across calls."""
    console.rule("[bold cyan]Test 3: Cost Tracker[/bold cyan]")
    
    config_path = PROJECT_ROOT / "configs" / "models.yaml"
    pool = ClientPool.from_config(config_path)
    tracker = CostTracker()
    
    # Make a few calls across different tiers
    messages = [{"role": "user", "content": "Say 'hello' in one word."}]
    
    for tier_name in ["T1", "T2"]:  # only fast/cheap tiers for this test
        try:
            response = pool.invoke(
                tier=tier_name,
                messages=messages,
                max_tokens=20,
            )
            tracker.record(response, task_id="cost_test", benchmark="cost_test")
        except Exception as e:
            console.print(f"[red]Failed {tier_name}: {e}[/red]")
            return False
    
    summary = tracker.summary()
    console.print("[bold]Cost Tracker Summary:[/bold]")
    
    table = Table()
    table.add_column("Tier")
    table.add_column("Calls", justify="right")
    table.add_column("In Tokens", justify="right")
    table.add_column("Out Tokens", justify="right")
    table.add_column("Cache Hit Rate", justify="right")
    table.add_column("Cost USD", justify="right", style="yellow")
    table.add_column("Avg Latency", justify="right")
    
    for tier, stats in summary["by_tier"].items():
        table.add_row(
            tier,
            str(stats["calls"]),
            str(stats["prompt_tokens"]),
            str(stats["completion_tokens"]),
            f"{stats['cache_hit_rate']*100:.1f}%",
            f"${stats['cost_usd']:.6f}",
            f"{stats['avg_latency']:.2f}s",
        )
    
    console.print(table)
    console.print(f"[bold green]Total cost: ${summary['total_cost_usd']:.6f}[/bold green]")
    return True


def main():
    """Run all Day 1 verification tests."""
    # Load .env from project root
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        console.print(f"[red]ERROR: {env_path} not found.[/red]")
        console.print("[yellow]Copy .env.example to .env and fill in your API keys.[/yellow]")
        sys.exit(1)
    
    load_dotenv(env_path)
    
    # Check keys are present
    if not os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "").startswith("sk-YOUR"):
        console.print("[red]ERROR: DEEPSEEK_API_KEY not set or placeholder still in .env[/red]")
        sys.exit(1)
    if not os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "").startswith("sk-YOUR"):
        console.print("[red]ERROR: DASHSCOPE_API_KEY not set or placeholder still in .env[/red]")
        sys.exit(1)
    
    console.print("[bold green]🚀 CHR-CP Day 1 Verification[/bold green]\n")
    
    # Run tests
    results = {}
    
    tier_results, _ = test_basic_invocation()
    results["basic_invocation"] = all(tier_results.values())
    
    console.print()
    results["stable_prefix"] = test_stable_prefix()
    
    console.print()
    results["cost_tracking"] = test_cost_tracking()
    
    # Final summary
    console.rule("[bold]Final Summary[/bold]")
    summary_table = Table()
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Result", style="bold")
    
    for test_name, passed in results.items():
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        summary_table.add_row(test_name, status)
    
    console.print(summary_table)
    
    if all(results.values()):
        console.print("\n[bold green]🎉 All Day 1 tests passed! Ready to proceed to Day 2.[/bold green]")
        sys.exit(0)
    else:
        console.print("\n[bold red]❌ Some tests failed. Fix issues before proceeding.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()