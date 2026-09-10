"""add is_admin to users

Revision ID: f57afca3b96a
Revises: 2d0671d1560f
Create Date: 2026-09-10 21:47:25.566522

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f57afca3b96a'
down_revision: Union[str, Sequence[str], None] = '2d0671d1560f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.false())
        )

    # Backfilled above, then the default is dropped so the schema matches the
    # model, which declares only a Python-side default.
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.alter_column("is_admin", server_default=None)



def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('is_admin')

