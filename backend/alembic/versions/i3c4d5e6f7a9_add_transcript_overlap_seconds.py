"""add transcript overlap seconds

Revision ID: i3c4d5e6f7a9
Revises: a9b8c7d6e5f4
Create Date: 2026-10-19 18:00:00.000000

How much of an overlap line two people actually spoke at once. Talk time
used to split the whole line in half when any overlap was found in it.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'i3c4d5e6f7a9'
down_revision: Union[str, Sequence[str], None] = 'a9b8c7d6e5f4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "transcript_lines",
        sa.Column("overlap_seconds", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("transcript_lines", "overlap_seconds")
