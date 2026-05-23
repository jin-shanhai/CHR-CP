"""Main experiment runner.

Runs: <method> × <benchmark> over N samples.
- Concurrent execution (ThreadPoolExecutor)
- Checkpoint to survive interruption
- Per-sample result with full trace

Usage:
    python -m experiments.run_main \
        --method chrcp \
        --benchmark gsm8k \
        --n_samples 100 \
        --concurrency 4
"""

from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Optional

import numpy as np

from loguru import logger
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn,
    MofNCompleteColumn,
)
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chr_cp.clients import ClientPool
from chr_cp.routing import (
    CHRCPOrchestrator, OrchestratorConfig,
    L1Config, L2Config, L3Config,
    RoutingAction,
)
from chr_cp.utils import CostTracker
from chr_cp.benchmarks import Benchmark, BenchmarkSample, EvaluationResult
from chr_cp.benchmarks.gsm8k import GSM8KBenchmark
from chr_cp.benchmarks.humaneval import HumanEvalBenchmark
from chr_cp.benchmarks.mmlu import MMLUBenchmark
from chr_cp.benchmarks.mmlu_pro import MMLUProBenchmark
from chr_cp.benchmarks.gpqa import GPQABenchmark
from chr_cp.benchmarks.aime import AIMEBenchmark
from chr_cp.benchmarks.math500 import MATHBenchmark
from experiments.progress_tracker import ProgressTracker


console = Console()


# ============================================================
# Benchmark registry
# ============================================================

def _make_math_hard():
    return MATHBenchmark(levels=[4, 5])

def _make_math_full():
    return MATHBenchmark(levels=None)

BENCHMARK_REGISTRY = {
    "gsm8k": GSM8KBenchmark,
    "humaneval": HumanEvalBenchmark,
    "mmlu": MMLUBenchmark,
    "mmlu_pro": MMLUProBenchmark,
    "gpqa": GPQABenchmark,
    "aime": AIMEBenchmark,
    "math": _make_math_full,           # all levels
    "math_hard": _make_math_hard,      # only L4-L5
}


def load_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(...)
    builder = BENCHMARK_REGISTRY[name]
    # If it's a class, instantiate; if it's a callable, call it
    if callable(builder) and not isinstance(builder, type):
        return builder()
    return builder()

# ============================================================
# Method registry
# ============================================================

def build_chrcp_method(pool: ClientPool, cost_tracker: CostTracker,
                        tau_low: float = 0.10, tau_high: float = 0.50,
                        alpha: float = 0.7, k_samples: int = 5,
                        cache_sensitivity: float = 0.3,
                        disable_branch: bool = False,
                        distillation_mode: str = "adaptive",
                        enable_ctor: bool = True,
                        ctor_mode: str = "self_compress",
                        ctor_target_role: str = "verifier_corrector",
                        ctor_max_handoff_tokens: int = 400,
                        enable_difficulty_routing: bool = True):
    """Build CHR-CP orchestrator (full method, all mechanisms enabled)."""
    config = OrchestratorConfig(
        enable_difficulty_routing=enable_difficulty_routing,
        l1_config=L1Config(mode="rule"),
        l2_config=L2Config(tau_low=tau_low, tau_high=tau_high,
                           alpha=alpha, k_samples=k_samples,
                           disable_branch=disable_branch),
        l3_config=L3Config(
            enable_m3_ctor=enable_ctor,
            ctor_mode=ctor_mode,
            ctor_target_role=ctor_target_role,
            ctor_max_handoff_tokens=ctor_max_handoff_tokens,
            distillation_mode=distillation_mode,
        ),
        per_task_budget_usd=0.50,
        max_tokens=16384,
        cache_sensitivity=cache_sensitivity,
    )
    return CHRCPOrchestrator(pool=pool, config=config, cost_tracker=cost_tracker)


def build_single_t1_method(pool: ClientPool, cost_tracker: CostTracker, **kwargs):
    """Single-agent baseline using only T1 (cheapest)."""
    return SingleTierBaseline(pool=pool, tier="T1", cost_tracker=cost_tracker)


def build_single_t4_method(pool: ClientPool, cost_tracker: CostTracker, **kwargs):
    """Single-agent baseline using only T4 (strongest)."""
    return SingleTierBaseline(pool=pool, tier="T4", cost_tracker=cost_tracker)


class SingleTierBaseline:
    """Baseline: just call one tier with the prompt directly. No MAS."""

    # Benchmark-specific format instructions so the model outputs parseable answers
    BENCHMARK_FORMATS = {
        "mmlu_pro": "Respond with the letter (A-J) inside \\boxed{}, e.g., \\boxed{C}.",
        "gpqa": "Respond with the letter (A-D) inside \\boxed{}, e.g., \\boxed{B}.",
        "aime": "AIME answers are integers 0-999. Provide your answer in \\boxed{N}.",
        "humaneval": "Provide a complete Python implementation in ```python ... ``` block.",
        "math": "End with \\boxed{ANSWER}.",
    }

    def __init__(self, pool: ClientPool, tier: str, cost_tracker: CostTracker):
        self.pool = pool
        self.tier = tier
        self.cost_tracker = cost_tracker

    def run(self, task: str, task_id: str = None, benchmark: str = None):
        """Mimic CHRCPResult interface for compatibility with the runner."""
        from chr_cp.routing.orchestrator import CHRCPResult

        fmt = self.BENCHMARK_FORMATS.get(benchmark or "", "")
        system_msg = f"You are a careful problem solver. {fmt}"
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": task},
        ]
        t0 = time.time()
        try:
            response = self.pool.invoke(
                tier=self.tier,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
            )
            self.cost_tracker.record(
                response, task_id=task_id, benchmark=benchmark,
            )
            return CHRCPResult(
                final_answer=response.content,
                final_tier=self.tier,
                total_cost_usd=response.cost_usd,
                total_latency_seconds=time.time() - t0,
                total_calls=1,
            )
        except Exception as e:
            logger.error(f"[{task_id}] {self.tier} baseline failed: {e}")
            return CHRCPResult(
                final_answer="",
                final_tier=self.tier,
                total_cost_usd=0.0,
                total_latency_seconds=time.time() - t0,
                total_calls=0,
            )


# ============================================================
# Task 5: Static-3-agent baseline (T2 → T3 → T4, no routing)
# ============================================================

SYSTEM_PROMPT_SUMMARIZE = "You are a careful problem solver. State your answer in \\boxed{}."

class Static3AgentBaseline:
    """Static 3-agent pipeline: T2 → T3 → T4, no VC² decisions."""

    def __init__(self, pool: ClientPool, cost_tracker: CostTracker,
                 max_tokens: int = 16384):
        self.pool = pool
        self.cost_tracker = cost_tracker
        self.max_tokens = max_tokens

    def run(self, task: str, task_id: str = None, benchmark: str = None):
        from chr_cp.routing.orchestrator import CHRCPResult
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARIZE},
            {"role": "user", "content": task},
        ]
        total_cost = 0.0
        total_latency = 0.0
        total_calls = 0
        last_response = None
        history: list = []

        for tier in ["T2", "T3", "T4"]:
            if last_response:
                msgs = messages + [
                    {"role": "assistant", "content": last_response.content},
                    {"role": "user", "content": "Verify and improve the above answer if needed."},
                ]
            else:
                msgs = messages
            t0 = time.time()
            resp = self.pool.invoke(tier=tier, messages=msgs, max_tokens=self.max_tokens)
            total_cost += resp.cost_usd
            total_latency += time.time() - t0
            total_calls += 1
            last_response = resp
            history.append({"action": "STATIC", "tier": tier})
            if self.cost_tracker:
                self.cost_tracker.record(resp, task_id=task_id,
                                         step_id=f"static_{tier}", routing_action="STATIC")

        return CHRCPResult(
            final_answer=last_response.content if last_response else "",
            final_tier="T4",
            total_cost_usd=total_cost,
            total_latency_seconds=total_latency,
            total_calls=total_calls,
        )


def build_static_3agent_method(pool: ClientPool, cost_tracker: CostTracker, **kwargs):
    return Static3AgentBaseline(pool=pool, cost_tracker=cost_tracker)


# ============================================================
# Task 6: RouteLLM-style baseline
# ============================================================

SYSTEM_PROMPT_WITH_CONFIDENCE = (
    "You are a careful problem solver. "
    "State your answer in \\boxed{}. "
    "End with: <confidence>X.X/10</confidence>"
)

class RouteLLMStyleBaseline:
    """RouteLLM-style training-free routing: T1 → confidence check → T4."""

    def __init__(self, pool: ClientPool, cost_tracker: CostTracker,
                 threshold: float = 0.8):
        self.pool = pool
        self.cost_tracker = cost_tracker
        self.threshold = threshold

    def run(self, task: str, task_id: str = None, benchmark: str = None):
        from chr_cp.routing.orchestrator import CHRCPResult
        from chr_cp.confidence.verbalized import VerbalizedConfidenceParser
        parser = VerbalizedConfidenceParser()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_WITH_CONFIDENCE},
            {"role": "user", "content": task},
        ]
        t_start = time.time()

        t1_resp = self.pool.invoke(tier="T1", messages=messages, max_tokens=16384)
        total_cost = t1_resp.cost_usd
        total_calls = 1
        conf = parser.parse(t1_resp.content)

        if conf.normalized >= self.threshold:
            return CHRCPResult(
                final_answer=t1_resp.content,
                final_tier="T1",
                total_cost_usd=total_cost,
                total_latency_seconds=time.time() - t_start,
                total_calls=total_calls,
            )

        t4_resp = self.pool.invoke(tier="T4", messages=messages, max_tokens=16384)
        total_cost += t4_resp.cost_usd
        total_calls += 1
        return CHRCPResult(
            final_answer=t4_resp.content,
            final_tier="T4",
            total_cost_usd=total_cost,
            total_latency_seconds=time.time() - t_start,
            total_calls=total_calls,
        )


def build_routellm_style_method(pool: ClientPool, cost_tracker: CostTracker, **kwargs):
    return RouteLLMStyleBaseline(pool=pool, cost_tracker=cost_tracker)


# ============================================================
# Task 7: OI-MAS adapted baseline
# ============================================================

class OIMASAdaptedBaseline:
    """OI-MAS adapted: Lead + 2 critics at T3, 4 calls total."""

    def __init__(self, pool: ClientPool, cost_tracker: CostTracker):
        self.pool = pool
        self.cost_tracker = cost_tracker

    def run(self, task: str, task_id: str = None, benchmark: str = None):
        from chr_cp.routing.orchestrator import CHRCPResult
        base_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARIZE},
            {"role": "user", "content": task},
        ]
        t_start = time.time()
        total_cost = 0.0
        total_calls = 0

        # 1. Lead
        lead = self.pool.invoke(tier="T3", messages=base_msgs, max_tokens=16384)
        total_cost += lead.cost_usd
        total_calls += 1

        # 2-3. Two critics (parallel via ThreadPoolExecutor)
        critic_prompt = base_msgs + [
            {"role": "assistant", "content": lead.content},
            {"role": "user", "content": "Critique the above answer. Identify any errors or missing steps."},
        ]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_a = ex.submit(self.pool.invoke, tier="T3", messages=critic_prompt,
                           max_tokens=16384)
            f_b = ex.submit(self.pool.invoke, tier="T3", messages=critic_prompt,
                           max_tokens=16384, temperature=0.8)
            critic_a = f_a.result()
            critic_b = f_b.result()
        total_cost += critic_a.cost_usd + critic_b.cost_usd
        total_calls += 2

        # 4. Synthesis
        syn_prompt = base_msgs + [
            {"role": "assistant", "content": lead.content},
            {"role": "user", "content": (
                f"Two reviewers provided feedback:\n\n"
                f"Reviewer A: {critic_a.content[:500]}\n\n"
                f"Reviewer B: {critic_b.content[:500]}\n\n"
                f"Provide your final answer in \\boxed{{}}."
            )},
        ]
        final = self.pool.invoke(tier="T3", messages=syn_prompt, max_tokens=16384)
        total_cost += final.cost_usd
        total_calls += 1

        return CHRCPResult(
            final_answer=final.content,
            final_tier="T3",
            total_cost_usd=total_cost,
            total_latency_seconds=time.time() - t_start,
            total_calls=total_calls,
        )


def build_oimas_adapted_method(pool: ClientPool, cost_tracker: CostTracker, **kwargs):
    return OIMASAdaptedBaseline(pool=pool, cost_tracker=cost_tracker)


METHOD_REGISTRY = {
    "chrcp": build_chrcp_method,
    "single_t1": build_single_t1_method,
    "single_t4": build_single_t4_method,
    "static_3agent": build_static_3agent_method,
    "routellm_style": build_routellm_style_method,
    "oimas_adapted": build_oimas_adapted_method,
}


# ============================================================
# Per-sample runner
# ============================================================

def run_one_sample(
    method,
    benchmark_obj: Benchmark,
    sample: BenchmarkSample,
) -> dict:
    """Run one sample through method, evaluate, return record."""
    t_start = time.time()
    
    try:
        result = method.run(
            task=sample.prompt,
            task_id=sample.sample_id,
            benchmark=sample.benchmark,
        )
    except Exception as e:
        logger.error(f"Sample {sample.sample_id} failed: {e}")
        return {
            "sample_id": sample.sample_id,
            "benchmark": sample.benchmark,
            "correct": False,
            "score": 0.0,
            "error": f"{type(e).__name__}: {e}",
            "cost_usd": 0.0,
            "latency_seconds": time.time() - t_start,
        }
    
    # Evaluate
    eval_result = benchmark_obj.evaluate(sample, result.final_answer)
    
    # Extract method configuration for traceability
    method_config = {}
    if hasattr(method, "config"):
        cfg = method.config
        method_config["max_tokens"] = getattr(cfg, "max_tokens", None)
        method_config["per_task_budget_usd"] = getattr(cfg, "per_task_budget_usd", None)
        if hasattr(cfg, "l2_config"):
            l2 = cfg.l2_config
            method_config["k_samples"] = getattr(l2, "k_samples", None)
            method_config["tau_low"] = getattr(l2, "tau_low", None)
            method_config["tau_high"] = getattr(l2, "tau_high", None)
            method_config["alpha"] = getattr(l2, "alpha", None)

    # Build record
    record = {
        "sample_id": sample.sample_id,
        "benchmark": sample.benchmark,
        "reference_answer": sample.reference,
        "correct": eval_result.correct,
        "score": eval_result.score,
        "extracted_answer": eval_result.extracted_answer,
        "final_answer": result.final_answer,
        "eval_error": eval_result.error,
        "eval_metadata": getattr(eval_result, "metadata", None) or {},
        "cost_usd": result.total_cost_usd,
        "latency_seconds": result.total_latency_seconds,
        "n_calls": result.total_calls,
        "final_tier": result.final_tier,
        "method_config": method_config,
        "api_calls": [
            {
                "step_idx": si,
                "tier": resp.tier_name,
                "model": resp.model_id,
                "provider": resp.provider,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "total_tokens": resp.total_tokens,
                "cache_hit_tokens": resp.cache_hit_tokens,
                "cost_usd": resp.cost_usd,
                "latency_seconds": resp.latency_seconds,
                "content": resp.content,
                "reasoning_content": resp.reasoning_content,
            }
            for si, step in enumerate(result.step_results)
            for resp in step.all_responses
        ],
    }

    # CHR-CP specific fields
    if hasattr(result, "num_stay"):
        record.update({
            "num_stay": result.num_stay,
            "num_branch": result.num_branch,
            "num_escalate": result.num_escalate,
            "distillations": result.distillations_triggered,
            "avg_cache_factor": result.avg_cache_factor,
            "l1_category": (
                result.l1_config.category.value if result.l1_config else None
            ),
            "decisions": [
                {
                    "action": d.action.value,
                    "current_tier": d.current_tier,
                    "target_tier": d.target_tier,
                    "uncertainty": d.uncertainty,
                    "u_verbalized": d.u_verbalized,
                    "u_consistency": d.u_consistency,
                    "tau_low": d.threshold_low,
                    "tau_high": d.threshold_high,
                    "ca2r_cache_factor": d.ca2r_cache_factor,
                    "ca2r_h_target": d.ca2r_h_target,
                    "reason": d.reason,
                }
                for step in result.step_results for d in step.decisions
            ],
            "ctor_handoffs": result.ctor_handoffs,
            # Difficulty-aware routing audit fields
            "routing_mode": getattr(result, "routing_mode", "cascade"),
            "start_tier": getattr(result, "start_tier", "?"),
            "difficulty_probe": {
                "domain": diff_probe.domain,
                "self_assessment": diff_probe.t1_self_assessment,
                "reasoning_depth": diff_probe.reasoning_depth,
                "needs_expert_knowledge": diff_probe.needs_expert_knowledge,
                "difficulty_score": diff_probe.difficulty_score,
                "t1_tentative_answer": diff_probe.t1_tentative_answer[:200],
                "probe_cost_usd": diff_probe.probe_cost_usd,
                "is_direct_candidate": diff_probe.is_direct_candidate,
            } if (diff_probe := getattr(result, "difficulty_probe", None)) else None,
            "escalation_jumps": getattr(result, "escalation_jumps", []),
            "branch_disabled_reason": getattr(result, "branch_disabled_reason", ""),
            "t1_crosscheck": getattr(result, "t1_crosscheck", "skipped"),
        })

    return record


# ============================================================
# Main runner
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=list(METHOD_REGISTRY.keys()))
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=list(BENCHMARK_REGISTRY.keys()))
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of test samples (use -1 for all)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of concurrent workers")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint")
    parser.add_argument("--max_cost_usd", type=float, default=10.0,
                        help="Hard cost cap; abort if exceeded")
    # === Sensitivity grid parameters ===
    parser.add_argument("--k_samples", type=int, default=5,
                        help="K for VC² consistency sampling (3, 5, or 7)")
    parser.add_argument("--tau_low", type=float, default=0.10,
                        help="STAY/BRANCH threshold (default: 0.10)")
    parser.add_argument("--tau_high", type=float, default=0.50,
                        help="BRANCH/ESCALATE threshold (default: 0.50)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="VC² signal fusion weight (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--cache_sensitivity", type=float, default=0.3,
                        help="CA²R cache sensitivity β (0 = open-loop, A1 ablation)")
    parser.add_argument("--disable_branch", action="store_true",
                        help="Disable BRANCH action, force STAY/ESCALATE only (A2 ablation)")
    parser.add_argument("--enable_ctor", type=lambda x: x.lower() == "true", default=True,
                        help="Enable M3 CTOR (Cross-Tier Output Reuse)")
    parser.add_argument("--ctor_mode", type=str, default="self_compress",
                        choices=["self_compress", "external_compress", "raw_passthrough", "off"],
                        help="CTOR compression strategy")
    parser.add_argument("--ctor_target_role", type=str, default="verifier_corrector",
                        choices=["verifier_corrector", "fresh_solver"],
                        help="CTOR target tier role binding")
    parser.add_argument("--ctor_max_handoff_tokens", type=int, default=400,
                        help="CTOR handoff packet hard upper bound")
    parser.add_argument("--enable_difficulty_routing", type=lambda x: x.lower() == "true",
                        default=True,
                        help="Enable difficulty-aware routing (A6 ablation)")
    parser.add_argument("--distillation_mode", type=str, default="adaptive",
                        choices=["adaptive", "fixed_t1", "fixed_t2", "fixed_t4"],
                        help="M2 distillation tier mode (A4 ablation)")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix for output JSONL filename (e.g., '_grid_D2')")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setup loguru file sink for detailed request/response trace
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / f"{args.benchmark}_{timestamp}.log",
        level="DEBUG",
        format="{time} | {level} | {message}",
    )
    
    load_dotenv(PROJECT_ROOT / ".env")
    
    n_samples = None if args.n_samples == -1 else args.n_samples
    
    console.rule(f"[bold cyan]CHR-CP Experiment: {args.method} × {args.benchmark}[/bold cyan]")
    console.print(
        f"n_samples={n_samples}, concurrency={args.concurrency}, "
        f"max_cost=${args.max_cost_usd}"
    )
    
    # Load benchmark
    benchmark_obj = load_benchmark(args.benchmark)
    samples = benchmark_obj.load(n_samples=n_samples)
    console.print(f"Loaded {len(samples)} samples from {args.benchmark}")
    
    # Setup output
    output_path = (
        Path(args.results_dir) / args.method / f"{args.benchmark}{args.output_suffix}.jsonl"
    )
    tracker = ProgressTracker(output_path)
    
    if args.resume:
        console.print(f"[yellow]Resuming: {tracker.done_count} samples already done[/yellow]")
    elif tracker.done_count > 0:
        console.print(
            f"[red]Output file has {tracker.done_count} existing records. "
            f"Use --resume to continue, or delete the file to restart.[/red]"
        )
        sys.exit(1)
    
    # Filter remaining samples
    remaining = [s for s in samples if not tracker.is_done(s.sample_id)]
    console.print(f"Remaining: {len(remaining)} samples to run")
    
    if not remaining:
        console.print("[green]Nothing to do; all samples already done.[/green]")
        return
    
    # Setup pool + tracker + method
    pool = ClientPool.from_config(PROJECT_ROOT / "configs" / "models.yaml")
    cost_tracker = CostTracker()
    
    method_builder = METHOD_REGISTRY[args.method]
    method = method_builder(pool=pool, cost_tracker=cost_tracker,
                            tau_low=args.tau_low, tau_high=args.tau_high,
                            alpha=args.alpha, k_samples=args.k_samples,
                            cache_sensitivity=args.cache_sensitivity,
                            disable_branch=args.disable_branch,
                            distillation_mode=args.distillation_mode,
                            enable_ctor=args.enable_ctor,
                            ctor_mode=args.ctor_mode,
                            ctor_target_role=args.ctor_target_role,
                            ctor_max_handoff_tokens=args.ctor_max_handoff_tokens,
                            enable_difficulty_routing=args.enable_difficulty_routing)
    
    # === Run with progress bar ===
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TextColumn("[bold green]{task.fields[acc]:.1%}[/bold green] acc"),
        TextColumn("•"),
        TextColumn("[bold yellow]${task.fields[cost]:.4f}[/bold yellow]"),
    ]
    
    n_correct = 0
    n_total = 0
    aborted = False
    
    with Progress(*progress_columns, console=console) as progress:
        bar = progress.add_task(
            f"[{args.method}/{args.benchmark}]",
            total=len(remaining),
            acc=0.0,
            cost=0.0,
        )
        
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_sample = {
                executor.submit(run_one_sample, method, benchmark_obj, s): s
                for s in remaining
            }
            
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    record = future.result()
                except Exception as e:
                    record = {
                        "sample_id": sample.sample_id,
                        "benchmark": sample.benchmark,
                        "correct": False,
                        "error": f"{type(e).__name__}: {e}",
                        "cost_usd": 0.0,
                    }
                
                tracker.write(record)
                
                if record.get("correct"):
                    n_correct += 1
                n_total += 1
                
                progress.update(
                    bar,
                    advance=1,
                    acc=n_correct / max(n_total, 1),
                    cost=cost_tracker.total_cost,
                )
                
                # Cost guard
                if cost_tracker.total_cost > args.max_cost_usd:
                    console.print(
                        f"[red]Cost cap exceeded "
                        f"(${cost_tracker.total_cost:.2f} > ${args.max_cost_usd}); "
                        f"aborting remaining work[/red]"
                    )
                    aborted = True
                    for f in future_to_sample:
                        f.cancel()
                    break
    
    # Final summary
    all_records = tracker.all_records()
    n_total_all = len(all_records)
    n_correct_all = sum(1 for r in all_records if r.get("correct"))
    total_cost = sum(r.get("cost_usd", 0.0) for r in all_records)
    avg_latency = (
        sum(r.get("latency_seconds", 0.0) for r in all_records) / max(n_total_all, 1)
    )
    
    console.rule("[bold]Final Summary[/bold]")
    console.print(f"Method: {args.method}")
    console.print(f"Benchmark: {args.benchmark}")
    console.print(f"Total samples: {n_total_all}")
    console.print(f"[bold green]Accuracy: {n_correct_all}/{n_total_all} = "
                  f"{n_correct_all/max(n_total_all,1)*100:.2f}%[/bold green]")
    console.print(f"[bold yellow]Total cost: ${total_cost:.4f}[/bold yellow]")
    console.print(f"Avg latency: {avg_latency:.2f}s")
    console.print(f"Output: {output_path}")
    
    if aborted:
        sys.exit(2)


if __name__ == "__main__":
    main()