"""merge transcript language and processing mode migrations

Revision ID: c00ba76e745a
Revises: e0f1a2b3c4d5, f1a2b3c4d5e6
Create Date: 2026-08-12 16:36:12.245981

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c00ba76e745a'
down_revision: Union[str, None] = ('e0f1a2b3c4d5', 'f1a2b3c4d5e6')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
