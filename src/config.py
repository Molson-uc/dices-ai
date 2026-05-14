from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "../model/best.pt"
    max_files_count: int = 100


settings = Settings()
