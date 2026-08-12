"""Add raw_text and cleaned_text to transcript_lines."""

from alembic import op
import sqlalchemy as sa

revision = "c8d9e0f1a2b3"
down_revision = "b7c8d9e0f1a2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("transcript_lines", sa.Column("raw_text", sa.Text(), nullable=True))
    op.add_column("transcript_lines", sa.Column("cleaned_text", sa.Text(), nullable=True))
    # Backfill: existing rows keep text as both raw and cleaned.
    op.execute(
        "UPDATE transcript_lines SET raw_text = text, cleaned_text = text WHERE raw_text IS NULL"
    )


def downgrade() -> None:
    op.drop_column("transcript_lines", "cleaned_text")
    op.drop_column("transcript_lines", "raw_text")
