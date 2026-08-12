"""add meetings audio_quality_warning

Revision ID: b7c8d9e0f1a2
Revises: 9d10f6b16b87
Create Date: 2026-08-10

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b7c8d9e0f1a2"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("meetings", sa.Column("audio_quality_warning", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("meetings", "audio_quality_warning")
