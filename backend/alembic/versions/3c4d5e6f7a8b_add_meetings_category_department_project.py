"""Add meeting_type, department, and project_name to meetings table.

Revision ID: 3c4d5e6f7a8b
Revises: 86b638adffce
Create Date: 2026-08-20 15:57:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = "3c4d5e6f7a8b"
down_revision = "86b638adffce"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("meetings", sa.Column("meeting_type", sa.String(length=32), nullable=True, server_default="internal"))
    op.add_column("meetings", sa.Column("department", sa.String(length=64), nullable=True, server_default="AI Team"))
    op.add_column("meetings", sa.Column("project_name", sa.String(length=128), nullable=True, server_default=""))


def downgrade() -> None:
    op.drop_column("meetings", "project_name")
    op.drop_column("meetings", "department")
    op.drop_column("meetings", "meeting_type")
