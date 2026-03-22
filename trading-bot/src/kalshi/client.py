import asyncio
import base64
import time
from urllib.parse import urlparse

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from pydantic_settings import BaseSettings


class KalshiConfig(BaseSettings):
    model_config = {"env_file": ".env", "extra": "ignore"}

    kalshi_api_key: str
    kalshi_rsa_private_key_path: str = "./kalshi_private_key.pem"
    kalshi_demo_api_key: str = ""
    kalshi_demo_rsa_private_key_path: str = "./kalshi_demo_private_key.pem"
    kalshi_env: str = "demo"

    @property
    def active_api_key(self) -> str:
        if self.kalshi_env == "demo" and self.kalshi_demo_api_key:
            return self.kalshi_demo_api_key
        return self.kalshi_api_key

    @property
    def active_private_key_path(self) -> str:
        if self.kalshi_env == "demo" and self.kalshi_demo_api_key:
            return self.kalshi_demo_rsa_private_key_path
        return self.kalshi_rsa_private_key_path

    @property
    def base_url(self) -> str:
        if self.kalshi_env == "prod":
            return "https://api.elections.kalshi.com/trade-api/v2"
        return "https://demo-api.kalshi.co/trade-api/v2"


class KalshiClient:
    def __init__(self, config: KalshiConfig):
        self.config = config
        self._private_key = self._load_private_key()
        self._client = httpx.AsyncClient(timeout=30.0)

    def _load_private_key(self):
        with open(self.config.active_private_key_path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)

    def _sign(self, timestamp_ms: int, method: str, path: str) -> str:
        message = f"{timestamp_ms}{method}{path}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        timestamp_ms = int(time.time() * 1000)
        signature = self._sign(timestamp_ms, method, path)
        return {
            "KALSHI-ACCESS-KEY": self.config.active_api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        }

    async def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        url = f"{self.config.base_url}{endpoint}"
        path = urlparse(url).path

        resp: httpx.Response | None = None
        for attempt in range(3):
            headers = self._auth_headers(method.upper(), path)
            resp = await self._client.request(method, url, headers=headers, **kwargs)

            if resp.status_code == 429:
                wait = 2 ** attempt
                await asyncio.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json() if resp.content else {}

        raise httpx.HTTPStatusError(
            "Rate limited after retries", request=resp.request, response=resp
        )

    async def get(self, endpoint: str, params: dict | None = None) -> dict:
        return await self._request("GET", endpoint, params=params)

    async def post(self, endpoint: str, json: dict | None = None) -> dict:
        return await self._request("POST", endpoint, json=json)

    async def delete(self, endpoint: str) -> dict:
        return await self._request("DELETE", endpoint)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
