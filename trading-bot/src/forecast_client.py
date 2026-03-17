import asyncio
import logging
import numpy as np
import httpx
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class ForecastRequest(BaseModel):
    latitude: float
    longitude: float
    model_name: str = "fcn"
    nensemble: int = 30
    lead_hours: int = 24


class ForecastResponse(BaseModel):
    temperatures_f: list[float]
    mean_f: float
    std_f: float
    model_name: str
    lead_hours: int


class ForecastClient:
    """Forecast client supporting mock, direct HTTP, and RunPod serverless modes."""

    def __init__(
        self,
        provider: str = "mock",
        base_url: str = "",
        api_key: str = "",
        runpod_endpoint_id: str = "",
        runpod_api_key: str = "",
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.runpod_endpoint_id = runpod_endpoint_id
        self.runpod_api_key = runpod_api_key
        self.timeout = timeout
        self.max_retries = max_retries

    async def get_forecast(self, request: ForecastRequest) -> ForecastResponse:
        if self.provider == "mock":
            return self._mock_forecast(request)
        if self.provider == "runpod":
            return await self._runpod_forecast(request)
        return await self._direct_forecast(request)

    def _mock_forecast(self, request: ForecastRequest) -> ForecastResponse:
        base_temp = 75 - (request.latitude - 25) * 0.8
        temps = np.random.normal(loc=base_temp, scale=4.0, size=request.nensemble).tolist()
        return ForecastResponse(
            temperatures_f=temps,
            mean_f=float(np.mean(temps)),
            std_f=float(np.std(temps)),
            model_name=request.model_name,
            lead_hours=request.lead_hours,
        )

    async def _direct_forecast(self, request: ForecastRequest) -> ForecastResponse:
        """POST directly to the FastAPI forecast service."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/forecast",
                        json=request.model_dump(),
                        headers=headers,
                    )
                    resp.raise_for_status()
                    return ForecastResponse(**resp.json())
            except (httpx.HTTPError, httpx.TimeoutException):
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def _runpod_forecast(self, request: ForecastRequest) -> ForecastResponse:
        """Submit job to RunPod serverless, poll until complete."""
        base = f"https://api.runpod.ai/v2/{self.runpod_endpoint_id}"
        headers = {"Authorization": f"Bearer {self.runpod_api_key}"}
        poll_interval = 2.0
        elapsed = 0.0

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Submit job
            resp = await client.post(
                f"{base}/run",
                json={"input": request.model_dump()},
                headers=headers,
            )
            resp.raise_for_status()
            job_id = resp.json()["id"]
            logger.info(f"RunPod job submitted: {job_id}")

            # Poll for completion
            while elapsed < self.timeout:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                status_resp = await client.get(
                    f"{base}/status/{job_id}",
                    headers=headers,
                )
                status_resp.raise_for_status()
                data = status_resp.json()
                status = data["status"]

                if status == "COMPLETED":
                    output = data["output"]
                    if "error" in output:
                        raise RuntimeError(f"RunPod handler error: {output['error']}")
                    return ForecastResponse(**output)

                if status == "FAILED":
                    raise RuntimeError(f"RunPod job failed: {data.get('error', 'unknown')}")

                logger.debug(f"RunPod job {job_id} status: {status} ({elapsed:.0f}s)")

                # Back off slowly: 2s → 4s → 5s cap
                poll_interval = min(poll_interval * 1.5, 5.0)

            raise TimeoutError(f"RunPod job {job_id} timed out after {self.timeout}s")
