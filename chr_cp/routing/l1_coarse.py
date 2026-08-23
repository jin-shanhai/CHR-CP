"""L1 Coarse Router: task-level routing.

Given a task description, decides:
- task_category (math / code / qa / reasoning / open)
- agent_pool: list of tiers and how many of each
- initial_topology: star / chain / mesh / tree

Two modes:
- "rule": zero-shot rule-based classifier (default, no training data needed)
- "learned": MLP classifier trained on auto-labeled data (optional)

For 2-week paper timeline, "rule" is the main mode used in experiments.
"learned" is reserved for ablation: "L1 rule vs L1 learned vs no-L1".
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import re

from loguru import logger


class TaskCategory(str, Enum):
    """5 task categories that drive initial pool configuration."""
    MATH = "math"                    # numerical, GSM8K, MATH
    CODE = "code"                    # HumanEval, MBPP
    KNOWLEDGE_QA = "knowledge_qa"    # MMLU, factual lookup
    LOGICAL_REASONING = "logical"    # BBH, multi-hop reasoning
    OPEN_TEXT = "open_text"          # creative writing, summarization, default


@dataclass
class AgentPoolConfig:
    """L1 output: configuration of the initial agent team."""
    
    # List of (role, tier) pairs, in execution order
    agents: list[tuple[str, str]]  # e.g., [("solver", "T2"), ("verifier", "T2"), ("aggregator", "T4")]
    
    # Communication topology
    # "star": all spokes report to a center aggregator
    # "chain": linear pipeline
    # "mesh": all-to-all
    # "tree": hierarchical
    topology: str
    
    # Detected category (for logging / paper analysis)
    category: TaskCategory
    
    # Confidence in this classification
    classification_confidence: float = 1.0
    
    def __repr__(self) -> str:
        return (
            f"AgentPoolConfig(category={self.category.value}, "
            f"topology={self.topology}, "
            f"agents={self.agents})"
        )


# === Default category-to-config mapping (Section 3.2 of paper) ===
# Each entry maps category → (agents list, topology)
DEFAULT_CONFIG_TABLE = {
    TaskCategory.MATH: {
        "agents": [
            ("solver", "T1"),
            ("verifier", "T2"),
            ("escalator", "T3"),    # NEW: T3 thinking, handles ESCALATE landing
        ],
        "topology": "chain",
    },
    TaskCategory.CODE: {
        "agents": [
            ("solver", "T1"),   
            ("verifier", "T2"),
            ("escalator", "T3"),    # NEW
        ],
        "topology": "chain",
    },
    TaskCategory.KNOWLEDGE_QA: {
        "agents": [
            ("solver", "T1"),
            ("verifier", "T2"),
            ("escalator", "T3"),    # NEW
        ],
        "topology": "chain",
    },
    TaskCategory.LOGICAL_REASONING: {
        "agents": [
            ("solver", "T1"),       # changed from T2 to T1 for consistency
            ("verifier", "T2"),
            ("escalator", "T3"),    # was 3rd-tier verifier; now properly named
        ],
        "topology": "chain",
    },
    TaskCategory.OPEN_TEXT: {
        "agents": [
            ("solver", "T1"),
            ("verifier", "T2"),
            ("escalator", "T3"),    # NEW
        ],
        "topology": "chain",
    },
}

# === Rule-based classifier ===

class _RuleSignals:
    """Compiled keyword/regex signals for rule-based classification."""
    
    # Math indicators
    MATH_KEYWORDS = re.compile(
        r"\b(compute|calculate|solve|integral|derivative|equation|sum|product|"
        r"multiply|divide|prove|theorem|geometry|algebra|arithmetic)\b",
        re.IGNORECASE,
    )
    MATH_SYMBOLS = re.compile(r"[\+\-\*/=<>≤≥∑∫∂√^]")
    MATH_NUMBERS_HEAVY = re.compile(r"\d+\.?\d*")  # we'll count occurrences
    
    # Code indicators
    CODE_FENCED = re.compile(r"```(?:python|javascript|cpp|java|c\+\+|go|rust)?", re.IGNORECASE)
    CODE_KEYWORDS = re.compile(
        r"\b(function|def\s+\w+|class\s+\w+|import|return|implement|"
        r"algorithm|debug|refactor|unit test|API)\b",
        re.IGNORECASE,
    )
    CODE_SYNTAX = re.compile(r"(\(\s*\w+\s*[,)]|\{[^}]*\}|=>\s*\{)")
    
    # MC indicators
    MC_PATTERN = re.compile(
        r"\b[A-E]\)\s|\b[A-E]\.\s|^\s*\([A-E]\)|"  # A) B) (A) etc.
        r"\b(which of the following|select the correct|choose the best)\b",
        re.IGNORECASE | re.MULTILINE,
    )
    
    # Logical reasoning indicators
    LOGICAL_KEYWORDS = re.compile(
        r"\b(if.*then|imply|deduce|infer|conclude|premise|"
        r"contradiction|consistent|valid argument|syllogism|"
        r"because|therefore|hence|consequently)\b",
        re.IGNORECASE,
    )


@dataclass
class L1Config:
    """Configuration for L1 router."""
    
    mode: str = "rule"  # "rule" or "learned"
    
    # Override default config table (advanced)
    config_table: Optional[dict] = None
    
    # Path to learned model checkpoint (if mode="learned")
    learned_checkpoint: Optional[str] = None


class L1Router:
    """L1 Coarse Router.
    
    Usage:
        router = L1Router()  # rule mode by default
        config = router.classify_and_configure(task_description)
        print(config.agents)  # [("solver", "T2"), ...]
    """
    
    def __init__(self, config: Optional[L1Config] = None):
        self.config = config or L1Config()
        self._config_table = self.config.config_table or DEFAULT_CONFIG_TABLE
        
        if self.config.mode == "learned":
            self._learned_model = self._load_learned_model()
        else:
            self._learned_model = None
        
        logger.info(f"L1Router initialized in mode={self.config.mode}")
    
    def classify_and_configure(self, task: str) -> AgentPoolConfig:
        """Main entry: task → AgentPoolConfig."""
        if self.config.mode == "learned" and self._learned_model is not None:
            category, conf = self._classify_learned(task)
        else:
            category, conf = self._classify_rule(task)
        
        entry = self._config_table[category]
        return AgentPoolConfig(
            agents=list(entry["agents"]),  # copy
            topology=entry["topology"],
            category=category,
            classification_confidence=conf,
        )
    
    # ----- Rule-based classifier -----
    
    def _classify_rule(self, task: str) -> tuple[TaskCategory, float]:
        """Classify using hand-crafted rules.
        
        Returns:
            (category, confidence in [0, 1])
        """
        if not task:
            return TaskCategory.OPEN_TEXT, 0.3
        
        scores = {cat: 0.0 for cat in TaskCategory}
        
        # === Math signals ===
        if _RuleSignals.MATH_KEYWORDS.search(task):
            scores[TaskCategory.MATH] += 2.0
        math_symbols = len(_RuleSignals.MATH_SYMBOLS.findall(task))
        scores[TaskCategory.MATH] += min(2.0, math_symbols * 0.3)
        num_count = len(_RuleSignals.MATH_NUMBERS_HEAVY.findall(task))
        if num_count >= 3:
            scores[TaskCategory.MATH] += min(1.5, num_count * 0.2)
        
        # === Code signals ===
        if _RuleSignals.CODE_FENCED.search(task):
            scores[TaskCategory.CODE] += 3.0
        if _RuleSignals.CODE_KEYWORDS.search(task):
            scores[TaskCategory.CODE] += 2.0
        if _RuleSignals.CODE_SYNTAX.search(task):
            scores[TaskCategory.CODE] += 1.0
        
        # === Multiple-choice / Knowledge QA ===
        mc_matches = len(_RuleSignals.MC_PATTERN.findall(task))
        if mc_matches >= 2:
            # 2+ option lines → likely MC question → knowledge_qa
            scores[TaskCategory.KNOWLEDGE_QA] += 2.5
        elif mc_matches == 1:
            scores[TaskCategory.KNOWLEDGE_QA] += 0.5
        
        # Question marks suggest QA (light signal)
        q_marks = task.count("?")
        if q_marks >= 1 and len(task) < 500:
            scores[TaskCategory.KNOWLEDGE_QA] += 0.5
        
        # === Logical reasoning ===
        logical_matches = len(_RuleSignals.LOGICAL_KEYWORDS.findall(task))
        scores[TaskCategory.LOGICAL_REASONING] += min(2.0, logical_matches * 0.5)
        
        # === Pick winner ===
        winner = max(scores, key=lambda k: scores[k])
        max_score = scores[winner]
        
        # If no signal triggered, fall back to OPEN_TEXT
        if max_score < 1.0:
            return TaskCategory.OPEN_TEXT, 0.3
        
        # Confidence: gap between winner and runner-up
        sorted_scores = sorted(scores.values(), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
        # Map gap → confidence (heuristic)
        confidence = min(1.0, 0.5 + gap * 0.15)
        
        return winner, confidence
    
    # ----- Learned classifier (optional) -----
    
    def _load_learned_model(self):
        """Load checkpoint if mode='learned'.
        
        Skipped for Day 5; if you want to enable later:
        1. Generate training data via auto-labeling (use T4 to label 1000 tasks)
        2. Train MLP on top of bge-small-zh embeddings
        3. Save checkpoint and pass path via L1Config.learned_checkpoint
        """
        if self.config.learned_checkpoint is None:
            logger.warning(
                "L1 mode='learned' but no checkpoint provided; "
                "falling back to rule-based classification"
            )
            return None
        
        # Lazy import to avoid mandatory dependency in rule mode
        try:
            import torch
            checkpoint = torch.load(self.config.learned_checkpoint, weights_only=False)
            return checkpoint  # rule-based fallback when no learned classifier is configured
        except Exception as e:
            logger.error(f"Failed to load L1 learned model: {e}")
            return None
    
    def _classify_learned(self, task: str) -> tuple[TaskCategory, float]:
        """Use a trained MLP when configured; otherwise use the deterministic rule path."""
        # Deferred to Day 11 / ablation; for now fall back
        return self._classify_rule(task)
