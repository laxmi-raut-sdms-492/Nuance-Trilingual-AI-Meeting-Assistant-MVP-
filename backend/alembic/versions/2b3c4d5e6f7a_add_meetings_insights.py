"""Add the meetings.insights JSON column.

The column was added to db/models.py without a matching migration, so the ORM
expected it while `alembic upgrade head` never created it. Tests did not catch
that: they run on SQLite and build the schema straight from the models, which
is exactly the path a migration gap is invisible on. On Postgres every meeting
read selects this column, so the omission failed the whole meetings table.

What it holds is models/summarizer.py:_extract_insights — attentionNeeded,
pending, commitments and deadlines for one meeting. Stored as one JSON document
rather than four tables because nothing queries into it: it is written once by
the summarization pass and read back whole by the Insights tab.

Nullable, with no backfill. NULL means the insights pass never ran for that
meeting, which is true of every row written before this and is what the UI
already keys on to decide whether to render the block at all.
"""

from alembic import op
import sqlalchemy as sa

revision = "2b3c4d5e6f7a"
down_revision = "1a2b3c4d5e6f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # sa.JSON, not JSONB, to match the generic type the model declares. JSONB
    # would index and compare better, but nothing here does either, and a
    # migration that disagrees with the model is the bug being fixed.
    op.add_column("meetings", sa.Column("insights", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("meetings", "insights")
