"""Add quote and source_time evidence columns to decisions and action_items.

The Insights UI needs to show *why* a decision or action item is on screen —
the transcript line it was verified against, and when that line was said —
not just the item itself. Both columns are nullable: the extractive engine can
backfill quote=text for its own rows, but rows written before this migration
have neither and simply render without a source line.
"""

from alembic import op
import sqlalchemy as sa

revision = "1a2b3c4d5e6f"
down_revision = "c00ba76e745a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("decisions", sa.Column("quote", sa.Text(), nullable=True))
    op.add_column("decisions", sa.Column("source_time", sa.String(length=32), nullable=True))
    op.add_column("action_items", sa.Column("quote", sa.Text(), nullable=True))
    op.add_column("action_items", sa.Column("source_time", sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column("action_items", "source_time")
    op.drop_column("action_items", "quote")
    op.drop_column("decisions", "source_time")
    op.drop_column("decisions", "quote")
