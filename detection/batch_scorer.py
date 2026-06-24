from concurrent.futures import ThreadPoolExecutor, as_completed

from config import config
from detection.model_inference import _score_one


def score_batch(wallets: list[str], max_workers: int = config.BATCH_SCORER_WORKERS) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_one, w): w for w in wallets}
        for future in as_completed(futures):
            wallet = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"wallet": wallet, "error": str(exc)})
    return results
