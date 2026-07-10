import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .utils.cache import ImageCache
from .config import settings



from .routers.Auth.Auth import router as auth_router

from .routers.logic.router import router  as Logic_router

from typing import Annotated
from fastapi import FastAPI, UploadFile, File
from app.models.database import init_engine

@asynccontextmanager
async def lifespan(app: FastAPI):


    
    ImageCache.connectAsync()
    init_engine()


    yield

    
    await ImageCache.close()


app = FastAPI(lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(auth_router, tags=["Auth"])

app.include_router(Logic_router, tags=["Logic"])

from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )

    schema["openapi"] = "3.0.3"

    # Convert OpenAPI 3.1 file schema to 3.0
    def fix(node):
        if isinstance(node, dict):
            if (
                node.get("type") == "string"
                and node.get("contentMediaType") == "application/octet-stream"
            ):
                node.pop("contentMediaType", None)
                node["format"] = "binary"

            for v in node.values():
                fix(v)

        elif isinstance(node, list):
            for item in node:
                fix(item)

    fix(schema)

    app.openapi_schema = schema
    return schema

app.openapi = custom_openapi


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}


@app.post("/upload-test")
async def upload_test(
    files: Annotated[list[UploadFile], File()]
):
    return {"count": len(files)}