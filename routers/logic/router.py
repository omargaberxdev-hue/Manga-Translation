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

router = APIRouter(prefix="/logic", tags=["Logic"], dependencies=[Depends(get_current_user)])


@router.post("/translate-comic")
async def translate_comic(
    payload: Annotated [str , Form(...)],              
    files: Annotated[List[UploadFile], File()],
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
    cdn=Depends(get_cdn),
    cache=Depends(get_cache),
):
    parsed_payload = TranslateComicRequest(**json.loads(payload))  

    exist = await cache.getAsync(f"${parsed_payload.MangaID}%${parsed_payload.ChapterID}")
    if exist:
        return exist

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