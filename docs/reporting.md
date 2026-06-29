# Reporting

## Narrative summaries

Forensic reports (`detection/forensic_report.py`) contain raw scores and
SHAP feature values. `reporting/narrative_builder.build_narrative()`
converts those into a short natural-language paragraph for non-technical
compliance officers and regulators.

### How it works

* One Jinja2 template per detection signal lives in `reporting/templates/`:
  `ring_detected.j2`, `benford_violation.j2`, `velocity_anomaly.j2`, and a
  `low_confidence.j2` fallback used when no signal applies.
* `build_narrative(report_dict)` checks the report for `ring_detection`,
  `benford_analysis` (windowed or flat `chi_square`/`p_value` summary), and
  `velocity_anomaly`, renders one paragraph per signal that's present (ring
  first, then Benford, then velocity), and joins them. If none apply, only
  `low_confidence.j2` is rendered.
* Each paragraph references the top-3 SHAP features from
  `report_dict["top_shap_features"]` by contribution magnitude, using
  plain-English labels from `reporting/feature_labels.py` instead of raw
  feature names.
* Output is capped at 300 words (truncated at a word boundary) to fit
  regulatory report page constraints.
* `REPORT_NARRATIVE_FORMAT` in `config.py` (`plain_text` or `markdown`)
  controls whether feature labels are wrapped in `**bold**`.
* Jinja2 autoescaping is enabled for all `.j2` templates, so any
  user-derived string in a report field (e.g. a malformed wallet label) is
  escaped rather than rendered verbatim.

### Adding a new template

1. Add `reporting/templates/<signal>.j2`.
2. Add a branch in `build_narrative()` that checks for the new signal's key
   in `report_dict` and renders the template, passing `top_features` and
   whatever signal-specific context the template needs.
3. Keep the paragraph self-contained (don't assume another paragraph ran
   first) and short enough that even with all signals present the combined
   narrative stays under 300 words.

### Feature label registry

`reporting/feature_labels.py` maps raw SHAP feature names (e.g.
`benford_mad_1h`) to short plain-English labels (e.g. "1-hour Benford's Law
deviation"). Unregistered feature names fall back to a de-slugified version
of the raw name (underscores replaced with spaces) rather than failing.
