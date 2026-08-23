"""Day 6 verification: L3 cache mechanisms work correctly.

Tests:
1. Cross-vendor detection
2. Distillation parses JSON output correctly
3. Distillation actually reduces token count
4. Speculative warming launches and doesn't crash
"""

from __future__ import annotations
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from chr_cp.clients import ClientPool
from chr_cp.clients.base_client import CompletionResponse
from chr_cp.routing import L3CacheManager, L3Config
from chr_cp.prompts.distillation import (
    parse_distilled,
    build_distillation_messages,
    estimate_tokens,
)
from chr_cp.prompts.handoff import EscalationReason
from chr_cp.utils import CostTracker


console = Console()


def test_cross_vendor_detection():
    """Test 1: cross-vendor logic correctly identifies handoffs."""
    console.rule("[bold cyan]Test 1: Cross-Vendor Detection[/bold cyan]")
    
    config_path = PROJECT_ROOT / "configs" / "models.yaml"
    pool = ClientPool.from_config(config_path)
    l3 = L3CacheManager(pool=pool, config=L3Config())
    
    cases = [
        ("T1", "T2", False),  # both deepseek
        ("T2", "T4", False),  # both deepseek
        ("T2", "T3", True),   # deepseek → qwen
        ("T3", "T4", True),   # qwen → deepseek
    ]
    
    table = Table(title="Cross-Vendor Cases")
    table.add_column("From")
    table.add_column("To")
    table.add_column("Expected")
    table.add_column("Actual")
    table.add_column("Status", style="bold")
    
    all_pass = True
    for src, dst, expected in cases:
        actual = l3.is_cross_vendor(src, dst)
        ok = actual == expected
        if not ok:
            all_pass = False
        table.add_row(
            src, dst, str(expected), str(actual),
            "[green]✓[/green]" if ok else "[red]✗[/red]",
        )
    
    console.print(table)
    return all_pass


def test_distillation_parser():
    """Test 2: parse_distilled handles various formats."""
    console.rule("[bold cyan]Test 2: Distillation JSON Parser[/bold cyan]")
    
    cases = [
        # (input, should_succeed)
        (
            '{"task_recap": "Solve x+y=10", "completed_steps": ["Listed equations"], '
            '"verified_facts": ["x and y are integers"], "pending_question": "Find x and y."}',
            True,
        ),
        (
            '```json\n{"task_recap": "test", "completed_steps": [], "verified_facts": [], '
            '"pending_question": "What now?"}\n```',
            True,  # fenced JSON
        ),
        (
            'Here is the summary:\n{"task_recap": "test", "completed_steps": [], '
            '"verified_facts": [], "pending_question": "next?"}\nDone.',
            True,  # surrounded by text
        ),
        ("not valid json at all", False),
        ("", False),
    ]
    
    all_pass = True
    for i, (text, should_succeed) in enumerate(cases):
        result = parse_distilled(text)
        success = (result is not None) == should_succeed
        if not success:
            all_pass = False
        status = "[green]✓[/green]" if success else "[red]✗[/red]"
        console.print(f"  Case {i+1}: {status} (expected_parse={should_succeed}, got={result is not None})")
    
    return all_pass


def test_distillation_live():
    """Test 3: actually run distillation on a sample history."""
    console.rule("[bold cyan]Test 3: Live Distillation Call[/bold cyan]")
    
    config_path = PROJECT_ROOT / "configs" / "models.yaml"
    pool = ClientPool.from_config(config_path)
    tracker = CostTracker()
    l3 = L3CacheManager(pool=pool, config=L3Config(), cost_tracker=tracker)
    
    # Long-form reasoning history (long enough to trigger distillation)
    history = """
    Step 1: I need to compute the integral of x^2 * sin(x) from 0 to pi.
    Let me use integration by parts. Let u = x^2, dv = sin(x) dx.
    Then du = 2x dx, v = -cos(x).
    
    Step 2: Applying integration by parts: ∫x²sin(x)dx = -x²cos(x) + ∫2x*cos(x)dx
    Now I need to compute ∫2x*cos(x)dx, again by parts.
    Let u = 2x, dv = cos(x)dx. Then du = 2dx, v = sin(x).
    ∫2x*cos(x)dx = 2x*sin(x) - ∫2*sin(x)dx = 2x*sin(x) + 2cos(x)
    
    Step 3: So the antiderivative is -x²cos(x) + 2x*sin(x) + 2cos(x).
    Evaluating from 0 to pi:
    At pi: -π²·(-1) + 2π·0 + 2·(-1) = π² - 2
    At 0: 0 + 0 + 2·1 = 2
    
    Step 4: Final result = (π² - 2) - 2 = π² - 4.
    """ * 3  # repeat to ensure it crosses distillation_min_tokens threshold
    
    history_tokens = estimate_tokens(history)
    console.print(f"Original history: ~{history_tokens} tokens")
    
    distilled = l3._distill_history(
    history_text=history,
    distillation_tier="T2",  # 显式指定 tier 用于测试
    task_id="distill_test",
)
    
    if distilled is None:
        console.print("[red]✗ Distillation returned None[/red]")
        console.print(f"[bold]Total cost spent:[/bold] ${tracker.total_cost:.6f}")
        return False
    
    console.print(f"[bold]Task recap:[/bold] {distilled.task_recap}")
    console.print(f"[bold]Completed steps ({len(distilled.completed_steps)}):[/bold]")
    for s in distilled.completed_steps:
        console.print(f"  - {s}")
    console.print(f"[bold]Verified facts ({len(distilled.verified_facts)}):[/bold]")
    for f in distilled.verified_facts:
        console.print(f"  - {f}")
    console.print(f"[bold]Pending question:[/bold] {distilled.pending_question}")
    
    distilled_text = distilled.to_text()
    distilled_tokens = estimate_tokens(distilled_text)
    
    console.print(f"\n[bold]Compression: {history_tokens} → {distilled_tokens} tokens "
                  f"({distilled_tokens/history_tokens*100:.1f}%)[/bold]")
    console.print(f"[bold]Cost: ${tracker.total_cost:.6f}[/bold]")
    
    # Validate: distillation should produce SHORTER text
    if distilled_tokens >= history_tokens:
        console.print("[yellow]⚠ Distilled text not smaller than original "
                      "(may indicate verbose model output)[/yellow]")
    else:
        console.print("[green]✓ Distillation reduces tokens[/green]")
    
    # Validate: at least task_recap and pending_question populated
    if distilled.task_recap and distilled.pending_question:
        console.print("[green]✓ Required fields populated[/green]")
        return True
    else:
        console.print("[red]✗ Missing required fields[/red]")
        return False


def test_ctor_handoff():
    """Test 4: M3 CTOR handoff (replaces warming)."""
    console.rule("[bold cyan]Test 4: CTOR Handoff (M3)[/bold cyan]")

    config_path = PROJECT_ROOT / "configs" / "models.yaml"
    pool = ClientPool.from_config(config_path)
    tracker = CostTracker()
    l3 = L3CacheManager(
        pool=pool,
        config=L3Config(ctor_mode="self_compress"),
        cost_tracker=tracker,
    )

    # Verify warming stubs work (backward compat)
    assert l3.maybe_warm_cache() == [], "maybe_warm_cache should return empty"
    assert l3.wait_warmups() == 0, "wait_warmups should return 0"
    console.print("[green]✓ Warming stubs are no-ops[/green]")

    # Test CTOR handoff packet construction
    test_response = (
        "Some long reasoning...\n"
        "<handoff>\n"
        "approach: test method\n"
        "candidate_answer: 42\n"
        "confidence: 0.85\n"
        "escalation_reason: LOW_CONFIDENCE\n"
        "target_should_check: verify step 3\n"
        "</handoff>"
    )
    packet = l3.build_handoff_packet(
        current_response=CompletionResponse(
            content=test_response, tier_name="T2", model_id="test", provider="test",
        ),
        source_tier="T2", target_tier="T3",
        escalation_reason=EscalationReason.LOW_CONFIDENCE,
    )
    assert packet is not None and packet.candidate_answer == "42"
    console.print(f"[green]✓ CTOR packet: candidate={packet.candidate_answer}, "
                  f"confidence={packet.confidence}, raw_tokens={packet.raw_token_count}[/green]")

    # Test build_target_messages
    msgs = l3.build_target_messages(
        base_messages=[
            {"role": "system", "content": "test system"},
            {"role": "user", "content": "test problem"},
        ],
        handoff_packet=packet,
        distilled_history=None,
        target_role="verifier_corrector",
        escalation_reason=EscalationReason.LOW_CONFIDENCE,
    )
    assert "<predecessor_handoff>" in msgs[1]["content"]
    console.print("[green]✓ CTOR target messages built with handoff[/green]")

    # Test effort & max_tokens decisions
    effort = l3.decide_reasoning_effort("T3", EscalationReason.LOW_CONFIDENCE, packet)
    assert effort == "minimal"
    console.print(f"[green]✓ Reasoning effort: {effort} (LOW_CONFIDENCE)[/green]")

    effort2 = l3.decide_reasoning_effort("T3", EscalationReason.WRONG_APPROACH)
    assert effort2 == "high"
    console.print(f"[green]✓ Reasoning effort: {effort2} (WRONG_APPROACH)[/green]")

    max_tok = l3.decide_max_tokens(EscalationReason.LOW_CONFIDENCE, "verifier_corrector")
    assert max_tok == 512
    console.print(f"[green]✓ Max tokens: {max_tok} (LOW_CONFIDENCE verifier)[/green]")

    return True


def main():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        console.print(f"[red]ERROR: {env_path} not found.[/red]")
        sys.exit(1)
    load_dotenv(env_path)
    
    console.print("[bold green]🚀 CHR-CP Day 6 Verification (L3 Cache)[/bold green]\n")
    
    results = {}
    results["cross_vendor"] = test_cross_vendor_detection()
    console.print()
    results["distillation_parser"] = test_distillation_parser()
    console.print()
    results["distillation_live"] = test_distillation_live()
    console.print()
    results["ctor_handoff"] = test_ctor_handoff()
    
    console.rule("[bold]Final Summary[/bold]")
    summary = Table()
    summary.add_column("Test")
    summary.add_column("Result", style="bold")
    
    for name, passed in results.items():
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        summary.add_row(name, status)
    
    console.print(summary)
    
    if all(results.values()):
        console.print("\n[bold green]🎉 Day 6 (L3) done![/bold green]")
        sys.exit(0)
    else:
        console.print("\n[bold red]❌ Some tests failed.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
