"""Add weather_readings, sensor_metadata, and static_features tables

Revision ID: 002
Revises: 001
Create Date: 2026-07-01
"""
from alembic import op
import sqlalchemy as sa

revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade():
    # sensor_metadata must be created first (FK dependency for static_features)
    op.create_table(
        'sensor_metadata',
        sa.Column('sensor_id', sa.Text(), primary_key=True),
        sa.Column('lat', sa.Float(), nullable=False),
        sa.Column('lon', sa.Float(), nullable=False),
        sa.Column('description', sa.Text()),
    )

    # static_features references sensor_metadata
    op.create_table(
        'static_features',
        sa.Column('sensor_id', sa.Text(), primary_key=True),
        sa.Column('transit_count', sa.Float(), nullable=False, server_default='0'),
        sa.Column('retail_count', sa.Float(), nullable=False, server_default='0'),
        sa.Column('food_drink_count', sa.Float(), nullable=False, server_default='0'),
        sa.Column('nightlife_count', sa.Float(), nullable=False, server_default='0'),
        sa.Column('education_count', sa.Float(), nullable=False, server_default='0'),
        sa.Column('recreation_count', sa.Float(), nullable=False, server_default='0'),
        sa.Column('total_poi_count', sa.Float(), nullable=False, server_default='0'),
        sa.Column('landuse_entropy', sa.Float(), nullable=False, server_default='0'),
        sa.Column('is_school_zone', sa.Float(), nullable=False, server_default='0'),
        sa.Column('betweenness_centrality', sa.Float(), nullable=False, server_default='0'),
        sa.Column('closeness_centrality', sa.Float(), nullable=False, server_default='0'),
        sa.Column('dist_to_city_centre', sa.Float(), nullable=False, server_default='0'),
        sa.Column('degree_centrality', sa.Float(), nullable=False, server_default='0'),
        sa.Column('nearby_bus_stops', sa.Float(), nullable=False, server_default='0'),
        sa.Column('nearby_parking', sa.Float(), nullable=False, server_default='0'),
        sa.ForeignKeyConstraint(['sensor_id'], ['sensor_metadata.sensor_id']),
    )

    # weather_readings as a TimescaleDB hypertable
    op.create_table(
        'weather_readings',
        sa.Column('dt', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('sensor_id', sa.Text(), nullable=False),
        sa.Column('humidity', sa.Float(), nullable=False),
        sa.Column('solar_radiation', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('dt', 'sensor_id'),
    )
    op.execute(
        "SELECT create_hypertable('weather_readings', 'dt', "
        "chunk_time_interval => INTERVAL '7 days');"
    )
    op.create_index(
        'idx_weather_sensor_dt',
        'weather_readings',
        ['sensor_id', sa.text('dt DESC')]
    )


def downgrade():
    op.execute("DROP TABLE IF EXISTS weather_readings;")
    op.drop_table('static_features')
    op.drop_table('sensor_metadata')
