from pydantic import BaseModel

class SaveRequest(BaseModel):
    title: str
    content: str
    version: int = 1
