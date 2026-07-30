"""add meetings.deleted_at (soft delete / trash)

Revision ID: a1b2c3d4e5f6
Revises: 5458ba8de72a
Create Date: 2026-07-30 16:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '5458ba8de72a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('meetings', sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True))
    op.create_index('ix_meetings_deleted_at', 'meetings', ['deleted_at'])


def downgrade() -> None:
    op.drop_index('ix_meetings_deleted_at', table_name='meetings')
    op.drop_column('meetings', 'deleted_at')