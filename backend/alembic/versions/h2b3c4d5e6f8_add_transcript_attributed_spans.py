"""add transcript attributed spans

Revision ID: h2b3c4d5e6f8
Revises: 86b638adffce, 2b3c4d5e6f7a
Create Date: 2026-08-17 19:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'h2b3c4d5e6f8'
down_revision: Union[str, Sequence[str], None] = ('86b638adffce', '2b3c4d5e6f7a')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "transcript_lines",
        sa.Column("is_separated_overlap", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.add_column(
        "transcript_lines",
        sa.Column("separation_confidence", sa.Float(), nullable=True),
    )
    op.add_column(
        "transcript_lines",
        sa.Column("attributed_spans", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("transcript_lines", "attributed_spans")
    op.drop_column("transcript_lines", "separation_confidence")
    op.drop_column("transcript_lines", "is_separated_overlap")
