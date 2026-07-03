from pydantic import BaseModel


class InterpolationResponse(BaseModel):
    image_url: str
    runtime: float
    resolution: str
    device: str
    model: str