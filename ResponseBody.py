from pydantic import BaseModel


class BodyAnalysisResult(BaseModel):
    messageOnShoulder: str
    shoulderCoordinate: int
    messageOnWaist: str
    waistCoordinate: int
