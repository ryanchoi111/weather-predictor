from pydantic import BaseModel


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
