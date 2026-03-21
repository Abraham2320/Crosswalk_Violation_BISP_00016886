from __future__ import annotations

from config import settings
from storage.database import Database, ViolationRepository


database = Database(settings)
database.create_all()


def get_database() -> Database:
    return database


def get_repository() -> ViolationRepository:
    return ViolationRepository(database)
