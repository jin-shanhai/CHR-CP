"""L3 Mechanism 1: Stable Prefix Engineering.

Forces strict prompt structure with monotonically growing cacheable prefix:

    [SYSTEM-FIXED]   永不变 (cacheable)
    [TASK-ANCHOR]    永不变 (cacheable)
    [SHARED-CONTEXT] 只追加 (incrementally cacheable)
    --------- prefix boundary ---------
    [ROLE-DYNAMIC]   每 agent 变化 (not cached)
    [STEP-PAYLOAD]   每步变化 (not cached)

Critical implementation invariant: the output is ALWAYS exactly two messages
    [system_message, user_message]
where system_message.content is byte-monotonically growing across calls
within a session. This is what enables DeepSeek's prefix cache to hit.

If prior_history is provided (e.g. for chain topology), it is FOLDED INTO
the system message as part of SHARED-CONTEXT, NOT inserted as separate
assistant/user turns (which would break the prefix invariant).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_SYSTEM_FIXED = """You are part of a multi-agent collaborative reasoning system.
Follow these rules in every response:
1. Reason step by step before giving the final answer.
2. End your response with a confidence assessment in this exact format:
   <confidence>X/10</confidence>
   where X is an integer from 1 to 10 indicating your certainty.
3. Format your final answer in the format requested by the task.
4. Be concise: avoid restating the problem unless asked."""


@dataclass
class StablePrefix:
    """Container for the cacheable prefix part of a CHR-CP prompt."""
    
    system_fixed: str
    task_anchor: str
    shared_context: list[str] = field(default_factory=list)  # append-only
    
    def to_messages(self) -> list[dict]:
        """Build the cacheable prefix as a single system message.
        
        Critical: ONE system message means a stable byte sequence for
        prefix matching. We never split into multiple system messages.
        """
        system_content = (
            f"{self.system_fixed}\n\n"
            f"=== TASK ===\n{self.task_anchor}"
        )
        
        if self.shared_context:
            shared_block = "\n\n".join(self.shared_context)
            system_content += f"\n\n=== SHARED CONTEXT ===\n{shared_block}"
        
        return [{"role": "system", "content": system_content}]
    
    def append_shared(self, content: str) -> None:
        """Append to shared context."""
        self.shared_context.append(content)

    def clear_shared(self) -> None:
        """Clear shared context for rolling replacement.

        Each agent only needs the immediately preceding agent's output,
        not the full accumulated history. Clearing before appending the
        latest keeps SHARED-CONTEXT compact without affecting prefix cache
        (SYSTEM-FIXED + TASK-ANCHOR remain unchanged).
        """
        self.shared_context.clear()
    
    def __len__(self) -> int:
        msgs = self.to_messages()
        return sum(len(m["content"]) for m in msgs)


class StablePrefixBuilder:
    """Builder for CHR-CP-compliant prompts.
    
    The output is always exactly [system_message, user_message]:
    - system_message contains the cacheable prefix (3 layers)
    - user_message contains the dynamic role + step payload
    
    For chain topology with prior_history, history is folded into the
    system message's SHARED-CONTEXT (preserving cache invariant).
    """
    
    def __init__(
        self,
        task: str,
        system_fixed: Optional[str] = None,
    ):
        self.prefix = StablePrefix(
            system_fixed=system_fixed or DEFAULT_SYSTEM_FIXED,
            task_anchor=task,
        )
    
    def append_shared_context(self, content: str) -> None:
        """Append shared context to the cacheable prefix."""
        self.prefix.append_shared(content)
    
    def build_messages(
        self,
        role_prompt: str,
        step_payload: str,
        prior_history: Optional[list[dict]] = None,
        replace_shared: bool = False,
    ) -> list[dict]:
        """Build a full message list for one agent invocation.
        
        Output structure (ALWAYS two messages):
            [
              {"role": "system", "content": "<cacheable prefix>"},
              {"role": "user",   "content": "<role + step>"}
            ]
        
        Args:
            role_prompt: Per-agent role description.
            step_payload: The actual step content.
            prior_history: Optional list of {"role", "content"} dicts.
                           If provided, each entry is APPENDED to the
                           shared_context layer (preserving cache
                           monotonicity). Should only be passed once per
                           new agent (otherwise duplicates accumulate).
        
        Returns:
            A 2-message list.
        """
        # 1. Fold prior_history into shared_context
        if replace_shared:
            self.prefix.clear_shared()
        if prior_history:
            for msg in prior_history:
                role_tag = msg.get("role", "user").upper()
                content = msg.get("content", "")
                tagged = f"[{role_tag}]: {content}"
                # Only append if not already present (idempotent)
                if tagged not in self.prefix.shared_context:
                    self.prefix.append_shared(tagged)
        
        # 2. Cacheable prefix as ONE system message
        messages = self.prefix.to_messages()
        
        # 3. Single user message (role + step)
        user_content = (
            f"=== YOUR ROLE ===\n{role_prompt}\n\n"
            f"=== CURRENT STEP ===\n{step_payload}"
        )
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def get_prefix_length(self) -> int:
        return len(self.prefix)