from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    redis_url: str
    cloudinary_cloud_name: str
    cloudinary_api_key: str
    cloudinary_api_secret: str

    db_url: str
    session_expire_seconds: int = 10000   # keep this one only

    celery_app_name: str = "manga_translation"
    celery_result_backend: Optional[str] = None
    celery_broker_url: str
    celery_worker_running: bool = True

    cdn_strategy: str 
    detection_strategy: str 
    ocr_strategy: str 
    translation_strategy: str 
    Inpainting :str


    # SESSION_EXPIRE_SECONDS deleted — duplicate of session_expire_seconds

    cdn_bucket: Optional[str] = None
    cdn_region: Optional[str] = None
    openrouterapikey : str 
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

settings = Settings()