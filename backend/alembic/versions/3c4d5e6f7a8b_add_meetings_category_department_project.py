"""Add meeting_type, department, and project_name to meetings table.

Revision ID: 3c4d5e6f7a8b
Revises: 86b638adffce
Create Date: 2026-08-20 15:57:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = "3c4d5e6f7a8b"
down_revision = "h2b3c4d5e6f8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS meeting_type VARCHAR(32) DEFAULT 'internal'"))
    conn.execute(sa.text("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS department VARCHAR(64) DEFAULT 'AI Team'"))
    conn.execute(sa.text("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS project_name VARCHAR(128) DEFAULT ''"))


def downgrade() -> None:
    op.drop_column("meetings", "project_name")
    op.drop_column("meetings", "department")
    op.drop_column("meetings", "meeting_type")
