"""speaker centroid and sample count

Splits the opaque `embedding` blob into the two values identification actually
needs: the running-average `centroid` and the `sample_count` it was averaged
over. Enrolling an existing name blends into the profile weighted by that
count, so storing it separately is what makes re-enrollment work.

NOTE: autogenerate also proposed dropping ix_meetings_fts,
ix_meetings_title_trgm, ix_meetings_agenda_trgm, ix_transcript_lines_fts and
ix_transcript_lines_text_trgm. Those are the hand-written full-text and
trigram indexes; Alembic cannot see them in the model metadata and reads them
as drift. Dropping them would silently disable search. They have been removed
from this migration, and env.py now filters them out of future autogenerate
runs.

Revision ID: 20680bce886a
Revises: 9d10f6b16b87
Create Date: 2026-07-29 16:53:23.454849
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20680bce886a"
down_revision: Union[str, None] = "9d10f6b16b87"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Added nullable, backfilled, then made NOT NULL — adding a NOT NULL column
    # without a default fails outright on a table with existing rows.
    op.add_column("speakers", sa.Column("centroid", sa.Text(), nullable=True))
    op.add_column("speakers", sa.Column("sample_count", sa.Integer(), nullable=True))
    op.add_column(
        "speakers",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # Carry existing rows across. The old `embedding` column held the whole
    # JSON profile; take its centroid if it has one, else treat the value
    # itself as the centroid (the pre-v2 flat-list format).
    op.execute(
        """
        UPDATE speakers
        SET centroid = CASE
                WHEN embedding::jsonb ? 'centroid' THEN embedding::jsonb ->> 'centroid'
                ELSE embedding
            END,
            sample_count = COALESCE((embedding::jsonb ->> 'sample_count')::int, 1)
        """
    )

    op.alter_column("speakers", "centroid", nullable=False)
    op.alter_column("speakers", "sample_count", nullable=False, server_default="1")
    op.create_check_constraint("ck_speakers_sample_count", "speakers", "sample_count >= 1")

    op.drop_column("speakers", "embedding")


def downgrade() -> None:
    op.add_column("speakers", sa.Column("embedding", sa.TEXT(), nullable=True))
    op.execute(
        """
        UPDATE speakers
        SET embedding = json_build_object(
            'centroid', centroid::json,
            'sample_count', sample_count
        )::text
        """
    )
    op.alter_column("speakers", "embedding", nullable=False)
    op.drop_constraint("ck_speakers_sample_count", "speakers", type_="check")
    op.drop_column("speakers", "updated_at")
    op.drop_column("speakers", "sample_count")
    op.drop_column("speakers", "centroid")
