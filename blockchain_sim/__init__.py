"""LLM-based blockchain incentive compatibility simulation platform.

This package contains the LLM agent modules that interact with
the Gymnasium blockchain environment. The architecture follows
a strict separation between the physical environment (Gym) and
the LLM decision logic.

Modules:
    translator  - Semantic perception layer (obs → natural language)
    executor    - Execution control layer (action grounding + normalization)
    cognition   - Cognition layer (LLM API + CoT reasoning)
    memory      - Dual-layer memory management
    runner      - Experiment orchestration
"""
