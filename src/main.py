from fastapi import FastAPI

from src.routes import router

app = FastAPI(title="Dices")
app.include_router(router)
