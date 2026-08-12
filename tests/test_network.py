from soul_tty.network import client_options, is_local_endpoint


def test_loopback_endpoints_ignore_environment_proxies():
    for url in (
        "http://127.0.0.1:8180",
        "http://localhost:50501",
        "http://[::1]:8080",
    ):
        assert is_local_endpoint(url)
        assert client_options(url, 10)["trust_env"] is False


def test_remote_endpoint_keeps_environment_proxy_support():
    url = "https://llm.example.com"
    assert not is_local_endpoint(url)
    assert client_options(url, 10)["trust_env"] is True
