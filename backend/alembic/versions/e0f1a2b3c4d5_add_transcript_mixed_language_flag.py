"""Add mixed-language detection fields to transcript_lines.

language_mixed_suspected — the top two candidate languages scored almost
equally on this segment, which usually means both were spoken in it.
language_margin — the numeric gap, kept so the threshold can be re-tuned
against stored transcripts without reprocessing the audio.

Existing rows get the "certain" defaults (False / 1.0). That is not a claim
that those lines were single-language — it is the only honest value available,
since the ranking they were derived from was never stored. Reprocess a meeting
if you need the flag on old audio.
"""

from alembic import op
import sqlalchemy as sa

revision = "e0f1a2b3c4d5"
down_revision = "d9e0f1a2b3c4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "transcript_lines",
        sa.Column(
            "language_mixed_suspected",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "transcript_lines",
        sa.Column(
            "language_margin",
            sa.Float(),
            nullable=False,
            server_default=sa.text("1.0"),
        ),
    )


def downgrade() -> None:
    op.drop_column("transcript_lines", "language_margin")
    op.drop_column("transcript_lines", "language_mixed_suspected")
