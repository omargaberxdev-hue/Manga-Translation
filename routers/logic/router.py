from typing import List , Annotated

from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
import json

from .depended import get_current_user
from app.models.database import get_db
from .schema import TranslateComicRequest
from .service import event_stream, create_chapters
from fastapi.responses import StreamingResponse
from fastapi import Request

from app.depends.depends import get_cdn, get_cache
import asyncio

from typing import Annotated

from fastapi import FastAPI, Header
from sqlalchemy import select, and_
from app.models.chapter import Chapter  # match the actual class name/casing

router = APIRouter(prefix="/logic", tags=["Logic"], dependencies=[Depends(get_current_user)])

@router.post("/translate-comic")
async def translate_comic(
    payload: Annotated[str, Form(...)],
    files: Annotated[List[UploadFile], File()],
    db:Session = Depends(get_db),
    user=Depends(get_current_user),
    cdn=Depends(get_cdn),
    cache=Depends(get_cache),
):
    parsed_payload = TranslateComicRequest(**json.loads(payload))
    cache_key = f"{parsed_payload.MangaID}:{parsed_payload.ChapterID}"

    cached = await cache.getAsync(cache_key)
    if cached:
        return json.loads(cached)

    stmt = select(Chapter).where(
        and_(
            Chapter.comic_name == parsed_payload.MangaID,
            Chapter.chapter_id == parsed_payload.ChapterID,
        )
    )
    result = db.execute(stmt)
    existing_chapter = result.scalar_one_or_none()
    if existing_chapter:
        response = {
            "id": existing_chapter.id,
            "chapter_id": existing_chapter.chapter_id,
            "comic_name": existing_chapter.comic_name,
            "canvas_url_before": existing_chapter.canvas_url_before,
            "canvas_url_after": existing_chapter.canvas_url_after,
            "user_id": existing_chapter.user_id,
        }
        await cache.setAsync(cache_key, response, ttl=60 * 60 * 24)
        return response

    contents = await asyncio.gather(*[f.read() for f in files])
    process_number = await create_chapters(parsed_payload, contents, user, db, cdn)

    return {"process_number": process_number, "Message": "your Message working on"}

@router.get("/events/")
async def sse( request: Request, last_event_id:  Annotated[str | None, Header()] = None  , user=Depends(get_current_user), cache = Depends(get_cache)  ):
    start_id = last_event_id if last_event_id else "$"
    return StreamingResponse(
        event_stream(user.id, request, cache, start_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )