"""Offline stubs for integration development workflows.

Developing against `LedgerLensContractClient` normally requires a funded
Testnet keypair and a deployed `ledgerlens-score` contract (see
`scripts/testnet_setup.py` and `tests/integration/README.md`) — high friction
for iterating on code that merely *calls* the contract client (new scripts,
alert dispatch, manual smoke-testing). `StubContractClient` is a drop-in,
in-memory stand-in with the same public method surface as
`LedgerLensContractClient` — no network, no `stellar_sdk`, no Testnet setup.

Usage:
    from integrations.offline_stubs import get_contract_client

    # Real client if LEDGERLENS_CONTRACT_ID/LEDGERLENS_SUBMITTER_SECRET are
    # set, StubContractClient if LEDGERLENS_OFFLINE=1 is set.
    client = get_contract_client()

    # Or force one explicitly:
    client = get_contract_client(offline=True)
"""

from __future__ import annotations

import os
from typing import Any

_STUB_SECRET = "SSTUBSTUBSTUBSTUBSTUBSTUBSTUBSTUBSTUBSTUBSTUBSTUBSTUBSTUBSTUB"


class StubContractClient:
    """In-memory stand-in for `LedgerLensContractClient`. No network calls.

    Mirrors `submit_score` / `submit_score_with_commitment` /
    `submit_score_with_uncertainty` / `get_score` /
    `propose_threshold_change` / `approve_threshold_change`. Scores submitted
    are readable back via `get_score` within the same process; nothing is
    persisted across runs. `approve_threshold_change` is a single-approval
    stub — it does not model the on-chain multi-sig quorum in
    `governance_contract.rs`, since offline development rarely needs that.
    """

    def __init__(
        self,
        contract_id: str = "STUB_CONTRACT",
        rpc_url: str = "offline://stub",
        network_passphrase: str = "Offline Stub Network",
        submitter_secret: str | None = _STUB_SECRET,
    ):
        self.contract_id = contract_id
        self.rpc_url = rpc_url
        self.network_passphrase = network_passphrase
        self.submitter_secret = submitter_secret
        self._scores: dict[tuple[str, str], dict] = {}
        self._next_proposal_id = 1
        self._proposals: dict[int, int] = {}
        self.calls: list[tuple[str, dict]] = []

    def submit_score(
        self,
        wallet: str,
        asset_pair: str,
        risk_score: dict,
        *,
        commitment: str | None = None,
        trade_data_hash: str | None = None,
        model_version_hash: str | None = None,
    ) -> dict:
        if not self.submitter_secret:
            raise ValueError("LEDGERLENS_SUBMITTER_SECRET is not configured")
        if commitment is not None and (trade_data_hash is None or model_version_hash is None):
            raise ValueError(
                "trade_data_hash and model_version_hash are required when commitment is set"
            )

        self.calls.append(("submit_score", {"wallet": wallet, "asset_pair": asset_pair}))
        stored = dict(risk_score)
        if commitment is not None:
            stored.update(
                commitment=commitment,
                trade_data_hash=trade_data_hash,
                model_version_hash=model_version_hash,
            )
        self._scores[(wallet, asset_pair)] = stored
        return stored

    def submit_score_with_commitment(
        self,
        wallet: str,
        asset_pair: str,
        risk_score: dict,
        commitment: str,
        trade_data_hash: str,
        model_version_hash: str,
    ) -> dict:
        return self.submit_score(
            wallet,
            asset_pair,
            risk_score,
            commitment=commitment,
            trade_data_hash=trade_data_hash,
            model_version_hash=model_version_hash,
        )

    def submit_score_with_uncertainty(
        self, wallet: str, asset_pair: str, risk_score_dict: dict
    ) -> dict:
        if not self.submitter_secret:
            raise ValueError("LEDGERLENS_SUBMITTER_SECRET is not configured")
        self.calls.append(
            ("submit_score_with_uncertainty", {"wallet": wallet, "asset_pair": asset_pair})
        )
        self._scores[(wallet, asset_pair)] = dict(risk_score_dict)
        return risk_score_dict

    def get_score(self, wallet: str, asset_pair: str) -> dict:
        try:
            return self._scores[(wallet, asset_pair)]
        except KeyError:
            raise LookupError(
                f"StubContractClient has no score for wallet={wallet!r} "
                f"asset_pair={asset_pair!r} — call submit_score first"
            ) from None

    def propose_threshold_change(
        self, governance_contract_id: str, new_threshold: int, proposer_secret: str
    ) -> int:
        proposal_id = self._next_proposal_id
        self._next_proposal_id += 1
        self._proposals[proposal_id] = new_threshold
        self.calls.append(("propose_threshold_change", {"proposal_id": proposal_id}))
        return proposal_id

    def approve_threshold_change(
        self, governance_contract_id: str, proposal_id: int, approver_secret: str
    ) -> bool:
        if proposal_id not in self._proposals:
            raise LookupError(f"StubContractClient has no open proposal {proposal_id}")
        self.calls.append(("approve_threshold_change", {"proposal_id": proposal_id}))
        return True


def get_contract_client(*, offline: bool | None = None, **kwargs: Any) -> Any:
    """Return a real `LedgerLensContractClient`, or `StubContractClient` when offline.

    `offline` defaults to the `LEDGERLENS_OFFLINE` env var (true for "1" or
    "true", case-insensitive). Extra `kwargs` are forwarded to whichever
    client is constructed.
    """
    if offline is None:
        offline = os.getenv("LEDGERLENS_OFFLINE", "").strip().lower() in ("1", "true")

    if offline:
        return StubContractClient(**kwargs)

    from integrations.contract_client import LedgerLensContractClient

    return LedgerLensContractClient(**kwargs)


# ---------------------------------------------------------------------------
# Issue #953 — Stub drift detection
# ---------------------------------------------------------------------------


class StubDriftError(Exception):
    """Raised by :class:`StubDriftDetector` when stub and real results diverge.

    The error message includes a structured diff so developers can quickly
    locate which scenario(s) changed.
    """


class StubDriftDetector:
    """Compare stub responses against real integration responses.

    When a real ``LedgerLensContractClient`` behaviour changes (e.g., a new
    required field, a changed return shape, a renamed key), the
    ``StubContractClient`` can silently fall out of sync.  This detector
    codifies the expected invariants as a structured diff and raises
    :exc:`StubDriftError` when any divergence is found.

    Usage::

        stub_results = run_contract_test_scenarios(client=StubContractClient())
        real_results = run_contract_test_scenarios(client=real_client)
        detector = StubDriftDetector()
        detector.compare_scenarios(stub_results, real_results)  # raises on drift

    Design
    ------
    * **Scenario keys** — both dicts must have identical top-level keys (one
      per test scenario).  Missing keys indicate that one side skipped a
      scenario, which is itself a drift signal.
    * **Per-scenario comparison** — for each scenario the detector checks:
      - ``ok`` flag (bool): did the operation succeed?
      - ``keys`` (frozenset): top-level keys of the result payload.
      - ``error_type`` (str | None): type name of any exception raised.
    * **Strict field enforcement** — any discrepancy produces a diff entry.
      The caller decides whether to warn or fail; ``compare_scenarios`` always
      raises :exc:`StubDriftError` on *any* diff entry.
    """

    def compare_scenarios(
        self,
        stub_results: dict[str, Any],
        real_results: dict[str, Any],
    ) -> None:
        """Compare stub results against real integration results.

        Parameters
        ----------
        stub_results:
            Mapping of scenario name → result dict from ``run_contract_test_scenarios``
            run against a :class:`StubContractClient`.
        real_results:
            Same structure from a run against the real
            ``LedgerLensContractClient``.

        Raises
        ------
        StubDriftError
            If any scenario is missing from one side, or if the per-scenario
            comparison detects a field-level divergence.
        """
        diffs: list[str] = []

        stub_keys = set(stub_results)
        real_keys = set(real_results)

        for missing in sorted(stub_keys - real_keys):
            diffs.append(f"Scenario {missing!r} present in stub but missing from real results")
        for extra in sorted(real_keys - stub_keys):
            diffs.append(f"Scenario {extra!r} present in real results but missing from stub")

        for scenario in sorted(stub_keys & real_keys):
            s = stub_results[scenario]
            r = real_results[scenario]
            scenario_diffs = self._diff_scenario(scenario, s, r)
            diffs.extend(scenario_diffs)

        if diffs:
            diff_block = "\n  ".join(diffs)
            raise StubDriftError(
                f"StubContractClient has drifted from real integration "
                f"({len(diffs)} divergence(s)):\n  {diff_block}\n\n"
                f"Update integrations/offline_stubs.py to match the real client behaviour."
            )

    def _diff_scenario(
        self,
        scenario: str,
        stub: dict[str, Any],
        real: dict[str, Any],
    ) -> list[str]:
        """Return a list of diff strings for a single scenario comparison."""
        diffs: list[str] = []

        # Compare ok flag
        if stub.get("ok") != real.get("ok"):
            diffs.append(
                f"[{scenario}] 'ok' mismatch: stub={stub.get('ok')!r} real={real.get('ok')!r}"
            )

        # Compare error type
        stub_err = stub.get("error_type")
        real_err = real.get("error_type")
        if stub_err != real_err:
            diffs.append(
                f"[{scenario}] 'error_type' mismatch: stub={stub_err!r} real={real_err!r}"
            )

        # Compare result keys (only when both succeeded)
        if stub.get("ok") and real.get("ok"):
            stub_payload_keys = frozenset(stub.get("result_keys") or [])
            real_payload_keys = frozenset(real.get("result_keys") or [])
            missing_in_stub = real_payload_keys - stub_payload_keys
            extra_in_stub = stub_payload_keys - real_payload_keys
            if missing_in_stub:
                diffs.append(
                    f"[{scenario}] stub result missing keys present in real: "
                    f"{sorted(missing_in_stub)}"
                )
            if extra_in_stub:
                diffs.append(
                    f"[{scenario}] stub result has extra keys not in real: "
                    f"{sorted(extra_in_stub)}"
                )

        return diffs


def run_contract_test_scenarios(
    client: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Run a canonical set of contract interaction scenarios against *client*.

    Returns a structured dict mapping scenario name → result summary:

    .. code-block:: python

        {
            "submit_and_get_score": {
                "ok": True,
                "result_keys": ["score", "benford_flag", "ml_flag", ...],
                "error_type": None,
            },
            ...
        }

    Parameters
    ----------
    client:
        A contract client instance (real or stub).  Defaults to
        ``StubContractClient()``.

    Returns
    -------
    dict[str, dict[str, Any]]
        Structured results keyed by scenario name.  Each value has:
        - ``ok`` (bool): True if the operation succeeded without exception.
        - ``result_keys`` (list[str] | None): sorted keys of the returned dict,
          or None if the operation failed.
        - ``error_type`` (str | None): type name of the exception, or None.
    """
    if client is None:
        client = StubContractClient()

    results: dict[str, dict[str, Any]] = {}

    _WALLET = "GTEST000000000000000000000000000000000000000000000000"
    _PAIR = "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native"
    _RISK_SCORE = {
        "score": 55,
        "benford_flag": False,
        "ml_flag": True,
        "timestamp": 1_700_000_000,
        "confidence": 80,
    }

    # Scenario 1: submit_score then get_score round-trip
    def _run_submit_and_get() -> dict[str, Any]:
        returned = client.submit_score(_WALLET, _PAIR, _RISK_SCORE)
        fetched = client.get_score(_WALLET, _PAIR)
        combined = {**returned, **fetched}
        return combined

    for scenario_name, fn in [
        ("submit_and_get_score", _run_submit_and_get),
        (
            "submit_score_with_uncertainty",
            lambda: client.submit_score_with_uncertainty(
                _WALLET,
                _PAIR,
                {**_RISK_SCORE, "score_lower": 50, "score_upper": 65, "coverage_guarantee": 0.9},
            ),
        ),
        (
            "get_score_missing_raises",
            lambda: client.get_score("GMISSING000000000000000000000000000000000000000000000000", _PAIR),
        ),
        (
            "propose_threshold_change",
            lambda: client.propose_threshold_change(
                "GOV_CONTRACT_PLACEHOLDER", 75, "SPLACEHOLDER_SECRET"
            ),
        ),
    ]:
        try:
            result = fn()
            if isinstance(result, dict):
                results[scenario_name] = {
                    "ok": True,
                    "result_keys": sorted(result.keys()),
                    "error_type": None,
                }
            else:
                results[scenario_name] = {
                    "ok": True,
                    "result_keys": None,
                    "error_type": None,
                }
        except Exception as exc:  # noqa: BLE001
            results[scenario_name] = {
                "ok": False,
                "result_keys": None,
                "error_type": type(exc).__name__,
            }

    return results
