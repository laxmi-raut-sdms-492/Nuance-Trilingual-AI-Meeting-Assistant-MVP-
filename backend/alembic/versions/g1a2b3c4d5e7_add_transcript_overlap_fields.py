"""add transcript overlap fields

Revision ID: g1a2b3c4d5e7
Revises: c00ba76e745a
Create Date: 2026-08-17 12:00:00.000000

Adds is_overlap, candidate_speakers, and candidate_labels columns to transcript_lines.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "g1a2b3c4d5e7"
down_revision: Union[str, None] = "c00ba76e745a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("transcript_lines") as batch_op:
        batch_op.add_column(
            sa.Column(
                "is_overlap",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )
        batch_op.add_column(
            sa.Column("candidate_speakers", sa.String(length=255), nullable=True)
        )
        batch_op.add_column(
            sa.Column("candidate_labels", sa.String(length=255), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("transcript_lines") as batch_op:
        batch_op.drop_column("candidate_labels")
        batch_op.drop_column("candidate_speakers")
        batch_op.drop_column("is_overlap")
