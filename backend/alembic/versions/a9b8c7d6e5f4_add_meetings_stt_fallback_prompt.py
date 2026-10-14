"""Add a pause-for-consent prompt before leaving Sarvam for a local model.

Revision ID: a9b8c7d6e5f4
Revises: 3c4d5e6f7a8b
"""

import sqlalchemy as sa
from alembic import op

revision = "a9b8c7d6e5f4"
down_revision = "3c4d5e6f7a8b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS fallback_prompt TEXT"))
    conn.execute(sa.text("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS fallback_decision VARCHAR(16)"))
    conn.execute(sa.text(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'ck_meetings_fallback_decision'
            ) THEN
                ALTER TABLE meetings ADD CONSTRAINT ck_meetings_fallback_decision
                    CHECK (fallback_decision IS NULL OR fallback_decision IN ('allow', 'decline'));
            END IF;
        END $$;
        """
    ))


def downgrade() -> None:
    op.drop_column("meetings", "fallback_decision")
    op.drop_column("meetings", "fallback_prompt")
