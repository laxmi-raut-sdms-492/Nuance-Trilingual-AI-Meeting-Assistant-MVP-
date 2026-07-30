"""trigram index on transcript text

Revision ID: 9d10f6b16b87
Revises: 8d126cd152a9
Create Date: 2026-07-29 16:48:39.862732

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9d10f6b16b87'
down_revision: Union[str, None] = '8d126cd152a9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Trigram index over transcript text, so substring search is indexed rather
    than a sequential scan.

    Why substring search exists at all: the full-text index uses the 'simple'
    configuration, which matches whole tokens. Devanagari is heavily
    inflected, so a transcript containing "मच्छरो" does not match a search for
    "मच्छर" under token matching. Measured on the real sample — that query
    returned zero results with full-text alone.

    gin_trgm_ops makes ILIKE '%needle%' index-assisted. Without it Postgres
    would scan every transcript_lines row, which is fine at three meetings and
    not fine later.
    """
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        "CREATE INDEX ix_transcript_lines_text_trgm "
        "ON transcript_lines USING GIN (text gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX ix_meetings_agenda_trgm "
        "ON meetings USING GIN (agenda gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_meetings_agenda_trgm")
    op.execute("DROP INDEX IF EXISTS ix_transcript_lines_text_trgm")
