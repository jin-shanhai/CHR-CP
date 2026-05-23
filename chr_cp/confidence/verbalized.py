"""Verbalized Confidence (VC) signal extraction.

Parses model outputs containing self-reported confidence in the format:
    <confidence>X/10</confidence>

Falls back to several alternative patterns if the strict format fails.
The robustness of fallback parsing matters because:
1. Some models occasionally drop the tags
2. Some output Chinese variants (置信度)
3. Some use percentage (e.g., "confidence: 80%" or "85% confident")

Empirical research shows verbalized confidence correlates ~0.6-0.7 with
actual accuracy, lower than logprobs but available across all API providers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import re

from loguru import logger


@dataclass
class VerbalizedConfidence:
    """Parsed verbalized confidence signal."""
    
    raw_score: Optional[float]  # In [0, 10] scale, None if parse failed
    normalized: float            # In [0, 1] scale, 0.5 if failed
    parse_success: bool
    pattern_used: Optional[str]
    raw_match: Optional[str]
    
    @property
    def uncertainty(self) -> float:
        """Convert confidence to uncertainty: U = 1 - confidence/10."""
        if not self.parse_success:
            # Failed parse → assume moderate uncertainty (do not bias decisions)
            return 0.5
        return 1.0 - self.normalized


class VerbalizedConfidenceParser:
    """Robust parser for verbalized confidence in model outputs.
    
    Patterns tried in order:
    1. Strict tag form: <confidence>X/10</confidence>
    2. Strict tag form: <confidence>X</confidence> (assume out of 10)
    3. Chinese tag form: <置信度>X/10</置信度>
    4. Inline form: confidence: X/10
    5. Percentage forms: 85% confident, confidence: 85%
    """
    
    PATTERNS = [
        # Pattern 1: <confidence>X/10</confidence>
        ("strict_tag_xy", r"<confidence>\s*(\d+(?:\.\d+)?)\s*/\s*10\s*</confidence>"),
        # Pattern 2: <confidence>X</confidence> (assume /10)
        ("tag_only_x", r"<confidence>\s*(\d+(?:\.\d+)?)\s*</confidence>"),
        # Pattern 3: Chinese variant
        ("chinese_tag", r"<置信度>\s*(\d+(?:\.\d+)?)\s*/\s*10\s*</置信度>"),
        # Pattern 4: Inline confidence: X/10
        ("inline", r"(?:confidence|置信度)\s*[::]\s*(\d+(?:\.\d+)?)\s*/\s*10"),
        # Pattern 5a: confidence: 85% (label-first form)
        ("percentage_inline", r"(?:confidence|置信度|certainty)\s*[::]\s*(\d+(?:\.\d+)?)\s*%"),
        # Pattern 5b: 85% confident / 85% confidence / 85% certain / 85% sure
        ("percentage", r"(\d+(?:\.\d+)?)\s*%\s*(?:confiden(?:ce|t)|certain(?:ty)?|sure)"),
    ]
    
    def parse(self, text: str) -> VerbalizedConfidence:
        """Extract verbalized confidence from text.
        
        Args:
            text: Model output text potentially containing confidence
        
        Returns:
            VerbalizedConfidence object
        """
        if not text:
            return VerbalizedConfidence(
                raw_score=None,
                normalized=0.5,
                parse_success=False,
                pattern_used=None,
                raw_match=None,
            )
        
        for pattern_name, pattern in self.PATTERNS:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if not matches:
                continue
            
            # Take the LAST match (in case of multiple, the final one is usually
            # the actual self-assessment, not a random number in the response)
            score_str = matches[-1]
            try:
                score = float(score_str)
                
                # percentage and percentage_inline return 0-100, others 0-10
                if pattern_name in ("percentage", "percentage_inline"):
                    if score < 0 or score > 100:
                        continue
                    normalized = score / 100.0
                    raw_score = normalized * 10
                else:
                    if score < 0 or score > 10:
                        continue
                    raw_score = score
                    normalized = score / 10.0
                
                return VerbalizedConfidence(
                    raw_score=raw_score,
                    normalized=normalized,
                    parse_success=True,
                    pattern_used=pattern_name,
                    raw_match=score_str,
                )
            except ValueError:
                continue
        
        # All patterns failed
        logger.debug(f"VC parse failed for text: {text[-200:]!r}")
        return VerbalizedConfidence(
            raw_score=None,
            normalized=0.5,
            parse_success=False,
            pattern_used=None,
            raw_match=None,
        )
    
    def strip_confidence_tag(self, text: str) -> str:
        """Remove the <confidence>...</confidence> tag from text.
        
        Useful when displaying clean output to users or downstream agents.
        """
        if not text:
            return text
        cleaned = re.sub(
            r"\s*<(?:confidence|置信度)>.*?</(?:confidence|置信度)>\s*",
            " ",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return cleaned.strip()