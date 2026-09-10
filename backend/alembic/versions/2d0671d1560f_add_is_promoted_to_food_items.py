"""add is_promoted to food_items

Revision ID: 2d0671d1560f
Revises: 88274ac9404f
Create Date: 2026-09-10 21:13:34.578228

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2d0671d1560f'
down_revision: Union[str, Sequence[str], None] = '88274ac9404f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add food_items.is_promoted.

    Autogenerate emitted this without a server_default, which SQLite rejects
    outright on a table that already has rows ("Cannot add a NOT NULL column
    with default value NULL"). The column is therefore added with a default so
    existing dishes backfill to false, and the default is then dropped so the
    schema still matches the model - which declares only a Python-side default -
    and `alembic check` stays clean.
    """
    with op.batch_alter_table("food_items", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "is_promoted", sa.Boolean(), nullable=False, server_default=sa.false()
            )
        )

    with op.batch_alter_table("food_items", schema=None) as batch_op:
        batch_op.alter_column("is_promoted", server_default=None)


def downgrade() -> None:
    """Drop food_items.is_promoted."""
    with op.batch_alter_table("food_items", schema=None) as batch_op:
        batch_op.drop_column("is_promoted")
