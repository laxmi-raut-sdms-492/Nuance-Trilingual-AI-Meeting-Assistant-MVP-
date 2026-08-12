"""add meetings processing_mode and stt_provider

Revision ID: f1a2b3c4d5e6
Revises: e0f1a2b3c4d5
Create Date: 2026-08-12

Per-meeting STT selection (local vs. cloud, and which cloud provider).
Both columns are nullable with no server default: NULL processing_mode
means "not set" and stt/resolver.py resolves that to
config.DEFAULT_PROCESSING_MODE ("local"). Every row that exists before this
migration runs gets NULL for both, which is exactly "local, unchanged" —
no backfill/data migration needed.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, None] = "c8d9e0f1a2b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("meetings") as batch_op:
        batch_op.add_column(sa.Column("processing_mode", sa.String(length=16), nullable=True))
        batch_op.add_column(sa.Column("stt_provider", sa.String(length=32), nullable=True))
        batch_op.create_check_constraint(
            "ck_meetings_processing_mode",
            "processing_mode IS NULL OR processing_mode IN ('local', 'cloud')",
        )


def downgrade() -> None:
    with op.batch_alter_table("meetings") as batch_op:
        batch_op.drop_constraint("ck_meetings_processing_mode", type_="check")
        batch_op.drop_column("stt_provider")
        batch_op.drop_column("processing_mode")
