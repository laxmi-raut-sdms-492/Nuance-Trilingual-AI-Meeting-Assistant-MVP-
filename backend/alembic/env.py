"""
Alembic environment.

The database URL comes from config.py (and therefore from the DATABASE_URL
environment variable), not from alembic.ini. One source of truth means a
migration can never run against a different database than the application.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the backend package importable however alembic was invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATABASE_URL  # noqa: E402
from db.models import Base  # noqa: E402

config = context.config
config.set_main_option("sqlalchemy.url", DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Indexes created by raw SQL in migrations, not declared on the models.
# Alembic cannot see them in the metadata and will propose DROPping them on
# every autogenerate — which would silently disable search. Filter them out.
_SQL_ONLY_INDEXES = {
    "ix_meetings_fts",
    "ix_meetings_title_trgm",
    "ix_meetings_agenda_trgm",
    "ix_transcript_lines_fts",
    "ix_transcript_lines_text_trgm",
}


def include_object(obj, name, type_, reflected, compare_to):
    if type_ == "index" and name in _SQL_ONLY_INDEXES:
        return False
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Without these, a changed column type or default autogenerates an
            # empty migration and the drift goes unnoticed.
            compare_type=True,
            compare_server_default=True,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
