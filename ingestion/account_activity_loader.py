"""Account activity ingestion via Horizon's effects endpoint.

Reads the `account_created` effect for each account to discover the funding
account (the wallet that issued `create_account`). This data feeds
`detection.wallet_graph.build_funding_graph`, enabling
`funding_source_similarity` and `network_centrality` features.

Horizon endpoint used:
    GET /accounts/{account_id}/effects?type=account_created
"""

from stellar_sdk import Server

from config import config
from ingestion.data_models import AccountActivity
from utils.logging import get_logger
from utils.retry import retry_with_backoff

logger = get_logger(__name__)


@retry_with_backoff(exceptions=(ConnectionError, TimeoutError, OSError))
def _fetch_effects(call_builder):
    return call_builder.call()


def load_account_activity(account_id: str) -> AccountActivity | None:
    """Fetch the ``account_created`` effect for a single account.

    Returns an :class:`AccountActivity` instance with ``funding_account`` set
    to the account that funded the creation, or ``None`` if no
    ``account_created`` effect exists (e.g. genesis accounts or accounts
    outside Horizon's history window).
    """
    server = Server(horizon_url=config.HORIZON_URL)
    call_builder = server.effects().for_account(account_id).limit(200).order(desc=False)

    page = _fetch_effects(call_builder)
    records = page.get("_embedded", {}).get("records", [])

    for record in records:
        if record.get("type") == "account_created":
            return AccountActivity(
                account_id=record["account"],
                account_created_at=record["created_at"],
                funding_account=record.get("funder"),
            )

    return None


def load_accounts_activity(account_ids: list[str]) -> list[AccountActivity]:
    """Batch-fetch :class:`AccountActivity` for a list of accounts.

    Individual lookup failures are logged as warnings and skipped so the
    rest of the batch is unaffected.
    """
    results: list[AccountActivity] = []
    for account_id in account_ids:
        try:
            activity = load_account_activity(account_id)
            if activity is not None:
                results.append(activity)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load activity for %s: %s", account_id, exc)
    return results
