"""Sanity check for new tier ladder (v2).

Verifies all 4 tiers can be invoked successfully and reports basic metrics.
Run BEFORE any benchmark experiments.

Cost: typically < $0.01 per full run.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from chr_cp.clients import ClientPool


console = Console()


SANITY_PROMPT = "Solve: 13 + 27 = ? Reply with just the number."


def test_all_tiers():
    """Ping each of T1-T4 with a trivial prompt."""
    console.rule("[bold cyan]New Tier Ladder Sanity Check[/bold cyan]")
    
    config_path = PROJECT_ROOT / "configs" / "models.yaml"
    pool = ClientPool.from_config(config_path)
    
    table = Table(title="Tier Ladder Verification")
    table.add_column("Tier", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Provider")
    table.add_column("Status", style="bold")
    table.add_column("Latency", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Cache Hit", justify="right")
    table.add_column("Response", style="dim")
    
    all_pass = True
    
    for tier_name in ["T1", "T2", "T3", "T4"]:
        tier_config = pool.configs.get(tier_name)
        if tier_config is None:
            table.add_row(
                tier_name, "—", "—", "[red]✗ NOT REGISTERED[/red]",
                "—", "—", "—", "Missing in models.yaml",
            )
            all_pass = False
            continue
        
        try:
            t_start = time.time()
            response = pool.invoke(
                tier=tier_name,
                messages=[{"role": "user", "content": SANITY_PROMPT}],
                temperature=0.1,
                max_tokens=128,
            )
            elapsed = time.time() - t_start
            
            content_short = response.content.strip().replace("\n", " ")[:50]
            if "40" in response.content:
                status = "[green]✓ OK[/green]"
            else:
                status = "[yellow]⚠ unexpected[/yellow]"
            
            cache_str = (
                f"{response.cache_hit_tokens}"
                if response.cache_hit_tokens is not None
                else "N/A"
            )
            
            table.add_row(
                tier_name,
                response.model_id,
                response.provider,
                status,
                f"{elapsed:.2f}s",
                f"${response.cost_usd:.6f}",
                cache_str,
                content_short,
            )
        except Exception as e:
            err_short = f"{type(e).__name__}: {str(e)[:60]}"
            table.add_row(
                tier_name,
                tier_config.get("model_id", "?"),
                tier_config.get("provider", "?"),
                "[red]✗ FAILED[/red]",
                "—",
                "—",
                "—",
                err_short,
            )
            all_pass = False
            console.print(f"\n[red]Detailed error for {tier_name}:[/red]")
            console.print(f"  {type(e).__name__}: {e}\n")
    
    console.print(table)
    return all_pass


def test_cross_vendor_boundaries():
    """Verify cross-vendor mapping reflects the new tier ladder."""
    console.rule("[bold cyan]Cross-Vendor Boundary Check[/bold cyan]")
    
    from chr_cp.routing.l3_cache import L3Config
    config = L3Config()
    
    expected_boundaries = [
        ("T1", "T2", True,  "zhipu → deepseek (cross)"),
        ("T2", "T3", False, "deepseek → deepseek (same)"),
        ("T3", "T4", True,  "deepseek → openai (cross)"),
        ("T1", "T3", True,  "zhipu → deepseek (cross)"),
        ("T1", "T4", True,  "zhipu → openai (cross)"),
        ("T2", "T4", True,  "deepseek → openai (cross)"),
    ]
    
    table = Table(title="Cross-Vendor Boundary Verification")
    table.add_column("From")
    table.add_column("To")
    table.add_column("Expected")
    table.add_column("Actual")
    table.add_column("Description", style="dim")
    table.add_column("Status", style="bold")
    
    all_pass = True
    
    def is_cross(t1, t2):
        return config.tier_to_provider[t1] != config.tier_to_provider[t2]
    
    for src, dst, expected, desc in expected_boundaries:
        actual = is_cross(src, dst)
        ok = actual == expected
        if not ok:
            all_pass = False
        table.add_row(
            src, dst,
            str(expected),
            str(actual),
            desc,
            "[green]✓[/green]" if ok else "[red]✗[/red]",
        )
    
    console.print(table)
    return all_pass


def test_repeat_call_cache_behavior():
    """Sanity-check cache hit reporting on T2 (DeepSeek)."""
    console.rule("[bold cyan]Cache Hit Sanity (T2 DeepSeek)[/bold cyan]")
    
    config_path = PROJECT_ROOT / "configs" / "models.yaml"
    pool = ClientPool.from_config(config_path)
    
    # Use a longer prompt to ensure something to cache
    long_prompt = (
        "You are a helpful assistant participating in a multi-step reasoning task. "
        "Always be precise, concise, and explain your reasoning step by step. "
        "When solving math problems, identify the operation, perform it carefully, "
        "and double-check by reverse computation. Never guess. "
        "Problem context: We are building a multi-agent system that routes tasks "
        "to different AI models based on their difficulty and the cost of each model. "
        "The system needs to make decisions at every step about whether to continue "
        "with the current model, branch into multiple samples for voting, or escalate "
        "to a stronger model. Each decision is informed by an uncertainty signal "
        "computed from the model's verbalized confidence and the consistency of "
        "multiple sampled responses. "
        "Your role: Solve the following problem step by step. "
        "Question: What is 13 + 27?"
    )
    try:
        r1 = pool.invoke(
            tier="T2",
            messages=[{"role": "user", "content": long_prompt}],
            temperature=0.0,
            max_tokens=64,
        )
        r2 = pool.invoke(
            tier="T2",
            messages=[{"role": "user", "content": long_prompt}],
            temperature=0.0,
            max_tokens=64,
        )
        
        console.print(
            f"Call 1: prompt_tokens={r1.prompt_tokens}, "
            f"cache_hit={r1.cache_hit_tokens}"
        )
        console.print(
            f"Call 2: prompt_tokens={r2.prompt_tokens}, "
            f"cache_hit={r2.cache_hit_tokens}"
        )
        
        if r2.cache_hit_tokens and r2.cache_hit_tokens > 0:
            hit_ratio = r2.cache_hit_tokens / r2.prompt_tokens
            console.print(
                f"[green]✓ Cache hit confirmed on T2 "
                f"({r2.cache_hit_tokens}/{r2.prompt_tokens} tokens = {hit_ratio*100:.1f}%)[/green]"
            )
            return True
        else:
            console.print(
                f"[red]✗ No cache hit on second call. "
                f"This indicates either the prompt is too short (<64 tokens) "
                f"or cache parsing is broken.[/red]"
            )
            return False  # HARD FAILURE
    except Exception as e:
        console.print(f"[red]✗ Call failed: {type(e).__name__}: {e}[/red]")
        return False


def main():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        console.print(f"[red]ERROR: {env_path} not found[/red]")
        sys.exit(1)
    load_dotenv(env_path)
    
    console.print("[bold green]🚀 New Tier Ladder Verification[/bold green]\n")
    
    results = {}
    results["all_tiers_callable"] = test_all_tiers()
    console.print()
    results["cross_vendor_correct"] = test_cross_vendor_boundaries()
    console.print()
    results["t2_cache_works"] = test_repeat_call_cache_behavior()
    
    console.rule("[bold]Summary[/bold]")
    summary = Table()
    summary.add_column("Test")
    summary.add_column("Result", style="bold")
    
    for name, passed in results.items():
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        summary.add_row(name, status)
    
    console.print(summary)
    
    if all(results.values()):
        console.print(
            "\n[bold green]🎉 New tier ladder is ready. "
            "Proceed to benchmark experiments.[/bold green]"
        )
        sys.exit(0)
    else:
        console.print(
            "\n[bold red]❌ Some tiers failed. Fix before running experiments.[/bold red]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()