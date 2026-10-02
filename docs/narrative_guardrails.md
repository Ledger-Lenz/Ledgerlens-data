# Narrative Guardrails

This document describes the factual-grounding guardrails applied to any
generative-text components in the narrative generation path
(`reporting/narrative_builder.py` and `detection/narrative_generator.py`).

## Goal

Narratives must never state a fact (amount, address, timestamp, count, etc.)
that is not directly supported by the underlying structured evidence record.
Any generative model output is treated as untrusted until it has been
cross-checked against that record.

## Design

1. **Grounded generation.** Factual claims should be produced from structured
evidence via templates or explicit field interpolation wherever possible.
Free-form generation is only permitted for connective prose that carries no
new factual content.

2. **Post-generation validation.** After a narrative is produced, a validation
step extracts every factual claim from the text and cross-checks it against
the structured evidence record. Claims that cannot be matched to the evidence
are flagged and the narrative is rejected (or surfaced with warnings) rather
than emitted as-is.

3. **Regression tests.** Tests use known evidence records paired with
expected and forbidden narrative claims. At minimum, one test injects an
intentionally ungrounded claim and asserts that validation catches it, and a
range of correctly grounded narratives are asserted to pass cleanly.

## Reviewer checklist for future narrative changes

- Does the change introduce any new free-generated factual claim?
- Is every new factual claim traceable to a field in the structured evidence?
- Are the validation rules and regression tests updated to cover the change?
- Does an intentionally injected ungrounded claim still get caught?
