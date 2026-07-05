# schema.py
from pydantic import BaseModel

class PagePosition(BaseModel):
    PageID: str
    x1: int
    y1: int
    w: int
    h: int

class ChapterData(BaseModel):
    ChapterID: str
    Pages: list[PagePosition]

class TranslateComicRequest(BaseModel):
    MangaID: str
    ChapterID: int
    target_language: str
    original_lang : str
    
class TranslateComicResponse(BaseModel):
    MangaID: str
    CanvasURL: str
    target_language: str
    Chapters_data: ChapterData