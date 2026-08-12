"""Add language_mix to transcript_lines.

Records every language spoken inside one speaker turn, comma-separated in
first-heard order. Populated only when a turn is code-switched; `language`
already holds the dominant one.

No backfill: existing rows predate turn-level language tracking, and there is
nothing stored from which the minority language could be recovered. NULL
correctly means "unknown", not "single language".
"""

from alembic import op
import sqlalchemy as sa

revision = "d9e0f1a2b3c4"
down_revision = "c8d9e0f1a2b3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "transcript_lines", sa.Column("language_mix", sa.String(length=64), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("transcript_lines", "language_mix")
