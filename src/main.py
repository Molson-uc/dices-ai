from fastapi import FastAPI

from routes import router

app = FastAPI(title="Dices")
app.include_router(router)
