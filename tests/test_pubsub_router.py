from streaming.pubsub_router import PubSubRouter


def test_subscribe_registers_client_for_each_channel() -> None:
    router = PubSubRouter()

    router.subscribe("client-1", ["wallet/GABC", "pair/XLM:USDC"])

    assert router.get_subscribers("wallet/GABC") == {"client-1"}
    assert router.get_subscribers("pair/XLM:USDC") == {"client-1"}
    assert router.get_subscriptions("client-1") == {"wallet/GABC", "pair/XLM:USDC"}


def test_unsubscribe_removes_only_requested_channels() -> None:
    router = PubSubRouter()
    router.subscribe("client-1", ["wallet/GABC", "pair/XLM:USDC"])

    router.unsubscribe("client-1", ["wallet/GABC"])

    assert router.get_subscribers("wallet/GABC") == set()
    assert router.get_subscribers("pair/XLM:USDC") == {"client-1"}
    assert router.get_subscriptions("client-1") == {"pair/XLM:USDC"}


def test_get_subscribers_is_channel_isolated() -> None:
    router = PubSubRouter()
    router.subscribe("client-1", ["wallet/GABC"])
    router.subscribe("client-2", ["wallet/GABC", "all"])
    router.subscribe("client-3", ["pair/XLM:USDC"])

    assert router.get_subscribers("wallet/GABC") == {"client-1", "client-2"}
    assert router.get_subscribers("pair/XLM:USDC") == {"client-3"}
    assert router.get_subscribers("all") == {"client-2"}
    assert router.get_subscribers("missing") == set()


def test_get_clients_for_event_combines_wallet_pair_and_admin_subscribers() -> None:
    router = PubSubRouter()
    router.subscribe("wallet-client", ["wallet/GABC"])
    router.subscribe("pair-client", ["pair/XLM:USDC"])
    router.subscribe("admin-client", ["all"])
    router.subscribe("other-client", ["wallet/GOTHER"])

    subscribers = router.get_clients_for_event("GABC", "XLM:USDC")

    assert subscribers == {"wallet-client", "pair-client", "admin-client"}


def test_disconnect_removes_client_from_all_channels() -> None:
    router = PubSubRouter()
    router.subscribe("client-1", ["wallet/GABC", "pair/XLM:USDC", "all"])
    router.subscribe("client-2", ["wallet/GABC"])

    router.disconnect("client-1")

    assert router.get_subscribers("wallet/GABC") == {"client-2"}
    assert router.get_subscribers("pair/XLM:USDC") == set()
    assert router.get_subscribers("all") == set()
    assert router.get_subscriptions("client-1") == set()
