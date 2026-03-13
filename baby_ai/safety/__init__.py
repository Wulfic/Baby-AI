"""Safety: action filtering, internet sandboxing.

Modules:

- **action_filter** — Screens proposed continuous action vectors against
  safety rules (currently a pass-through; override ``filter_action()``
  to add custom constraints).
- **sandbox** — Restricts internet access to a curated allowlist of
  domains, preventing the agent from reaching arbitrary endpoints.
"""
