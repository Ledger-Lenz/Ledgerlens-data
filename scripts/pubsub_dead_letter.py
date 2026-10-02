"""Inspect and replay PubSub dead-lettered messages (Issue #899).

Usage::

    python scripts/pubsub_dead_letter.py list [--path FILE]
    python scripts/pubsub_dead_letter.py show ID [--path FILE]
    python scripts/pubsub_dead_letter.py replay [ID ...] [--path FILE]

``replay`` re-delivers messages by printing them to stdout as JSON lines
(the default operator sink); successfully replayed entries are removed.
"""

from __future__ import annotations

import argparse
import json
import sys

from streaming.pubsub_router import DEFAULT_DEAD_LETTER_PATH, DeadLetterStore, PubSubRouter


def _stdout_handler(client_id: str, message: dict) -> None:
    print(json.dumps({"client_id": client_id, "message": message}, default=str))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["list", "show", "replay"])
    p.add_argument("ids", nargs="*")
    p.add_argument("--path", default=DEFAULT_DEAD_LETTER_PATH)
    args = p.parse_args(argv)

    store = DeadLetterStore(args.path)
    if args.command == "list":
        for e in store.entries():
            print(f"{e['id']}\tretries={e['retry_count']}\t{e['error']}")
    elif args.command == "show":
        for e in store.entries():
            if e["id"] in args.ids:
                print(json.dumps(e, indent=2, default=str))
    else:
        router = PubSubRouter(dead_letter_store=store)
        print(json.dumps(router.replay_dead_letters(_stdout_handler, set(args.ids) or None)),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
