"""merge overlap fields with other heads

Revision ID: 86b638adffce
Revises: 1a2b3c4d5e6f, g1a2b3c4d5e7
Create Date: 2026-08-17 12:27:39.371843

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '86b638adffce'
down_revision: Union[str, None] = ('1a2b3c4d5e6f', 'g1a2b3c4d5e7')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
