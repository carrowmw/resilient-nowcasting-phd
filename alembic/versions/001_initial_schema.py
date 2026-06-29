"""Initial schema: traffic_readings hypertable, model_runs, dataset_versions

Revision ID: 001
Revises:
Create Date: 2026-06-29
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create TimescaleDB extension
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

    # Traffic readings hypertable
    op.create_table(
        'traffic_readings',
        sa.Column('dt', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('sensor_id', sa.Text(), nullable=False),
        sa.Column('veh_class', sa.Text(), nullable=False),
        sa.Column('direction', sa.Text(), nullable=False),
        sa.Column('count', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('dt', 'sensor_id', 'veh_class', 'direction')
    )
    op.execute(
        "SELECT create_hypertable('traffic_readings', 'dt', "
        "chunk_time_interval => INTERVAL '7 days');"
    )
    op.create_index(
        'idx_traffic_sensor_dt',
        'traffic_readings',
        ['sensor_id', sa.text('dt DESC')]
    )
    op.create_index(
        'idx_traffic_vehclass',
        'traffic_readings',
        ['veh_class', sa.text('dt DESC')]
    )
    op.create_index(
        'idx_traffic_direction',
        'traffic_readings',
        ['direction', sa.text('dt DESC')]
    )

    # Dataset versions
    op.create_table(
        'dataset_versions',
        sa.Column('version_id', sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column('git_branch', sa.Text(), nullable=False),
        sa.Column('git_commit', sa.Text(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('description', sa.Text()),
        sa.Column('preprocessing', JSONB, nullable=False),
        sa.Column('parquet_sha256', sa.Text()),
        sa.Column('row_count', sa.BigInteger()),
        sa.UniqueConstraint('git_commit', 'parquet_sha256')
    )
    op.create_index(
        'idx_dataset_branch',
        'dataset_versions',
        ['git_branch', sa.text('created_at DESC')]
    )

    # Model runs
    op.create_table(
        'model_runs',
        sa.Column('run_id', sa.Text(), primary_key=True),
        sa.Column('wandb_run_id', sa.Text(), unique=True),
        sa.Column('git_commit', sa.Text(), nullable=False),
        sa.Column('git_branch', sa.Text(), nullable=False),
        sa.Column('config_yaml', JSONB, nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('val_rmse', sa.Float()),
        sa.Column('val_mae', sa.Float()),
        sa.Column('val_zero_acc', sa.Float()),
        sa.Column('val_loss', sa.Float()),
        sa.Column('batch_size', sa.Integer()),
        sa.Column('learning_rate', sa.Float()),
        sa.Column('num_epochs', sa.Integer()),
        sa.Column('dataset_version_id', sa.Integer(), sa.ForeignKey('dataset_versions.version_id'))
    )
    op.create_index('idx_runs_wandb', 'model_runs', ['wandb_run_id'])
    op.create_index(
        'idx_runs_branch',
        'model_runs',
        ['git_branch', sa.text('created_at DESC')]
    )
    op.create_index(
        'idx_runs_val_loss',
        'model_runs',
        [sa.text('val_loss ASC NULLS LAST')]
    )


def downgrade():
    op.drop_table('model_runs')
    op.drop_table('dataset_versions')
    op.execute("SELECT drop_chunks('traffic_readings', older_than => INTERVAL '-1 day');")
    op.drop_table('traffic_readings')
    op.execute("DROP EXTENSION IF EXISTS timescaledb CASCADE;")
