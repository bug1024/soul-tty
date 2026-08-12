"""HTTP 客户端策略：本机服务不继承系统代理，远程端点保持默认行为。"""

import ipaddress
from urllib.parse import urlparse


def is_local_endpoint(url: str) -> bool:
    """判断 URL 是否明确指向 loopback；无法解析时按远程端点处理。"""
    hostname = (urlparse(url).hostname or "").strip().lower()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def client_options(url: str, timeout) -> dict:
    """返回 httpx.Client 参数；仅本机端点关闭环境代理发现。"""
    return {
        "timeout": timeout,
        "trust_env": not is_local_endpoint(url),
    }
