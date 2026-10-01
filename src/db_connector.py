"""
db_connector.py

Unified DataConnector abstraction for the ResilientIoT ML pipeline.

Backends:
  - TimescaleConnector: queries TimescaleDB (production, tune, analysis)
  - ParquetConnector:   reads local parquet/JSON files (smoke tests only)

Usage:
    connector = create_connector(config)  # config dict with "data_source" key
    df = connector.load_traffic(start_dt, end_dt, "person", "all")
    connector.close()
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow.parquet as pq
import psycopg2
from psycopg2.pool import SimpleConnectionPool

logger = logging.getLogger(__name__)

DEFAULT_PARQUET_PATH = "data/processed/master_dataset_15T.parquet"

STATIC_FEATURE_COLS = [
    'transit_count', 'retail_count', 'food_drink_count', 'nightlife_count',
    'education_count', 'recreation_count', 'total_poi_count', 'landuse_entropy',
    'is_school_zone', 'betweenness_centrality', 'closeness_centrality',
    'dist_to_city_centre', 'degree_centrality', 'nearby_bus_stops', 'nearby_parking',
]


class DataConnector(ABC):
    """Abstract base class for data connectors."""

    @abstractmethod
    def load_traffic(self, start_dt, end_dt, veh_class: str, direction: str) -> pd.DataFrame:
        """
        Load traffic readings for the given window and filters.

        Returns DataFrame with columns: [dt, location, value, veh_class, dir]
        """

    @abstractmethod
    def load_weather(self, start_dt, end_dt, sensor_ids: List[str]) -> pd.DataFrame:
        """
        Load weather readings for the given window and sensors.

        Returns DataFrame with columns: [dt, location, Humidity, Solar Radiation]
        """

    @abstractmethod
    def load_static_features(self, sensor_ids: List[str]) -> pd.DataFrame:
        """
        Load static features for the given sensors.

        Returns DataFrame with columns: [location, <15 static feature columns>]
        """

    @abstractmethod
    def load_sensor_metadata(self) -> Dict[str, Dict[str, float]]:
        """
        Load sensor coordinates.

        Returns: {sensor_id: {"lat": float, "lon": float}}
        """

    @abstractmethod
    def close(self):
        """Release all resources (pool teardown, file handles, etc.)."""


class _PoolConnection:
    """Context manager that borrows/returns a connection from a SimpleConnectionPool."""

    def __init__(self, pool: SimpleConnectionPool):
        self._pool = pool
        self._conn = None

    def __enter__(self):
        self._conn = self._pool.getconn()
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn is not None:
            self._pool.putconn(self._conn)
            self._conn = None
        return False


class TimescaleConnector(DataConnector):
    """
    Production connector backed by TimescaleDB.
    Uses a SimpleConnectionPool (minconn=1, maxconn=4).
    """

    def __init__(self, database_url: str):
        assert database_url, "CRITICAL: database_url must be a non-empty string"
        self._pool = SimpleConnectionPool(minconn=1, maxconn=4, dsn=database_url)
        # Validate the pool is live
        with _PoolConnection(self._pool) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
        logger.info("TimescaleConnector: connection pool established (min=1, max=4)")

    def load_traffic(self, start_dt, end_dt, veh_class: str, direction: str) -> pd.DataFrame:
        sql = """
            SELECT dt, sensor_id AS location, veh_class, direction AS dir, count AS value
            FROM traffic_readings
            WHERE dt >= %s AND dt < %s AND veh_class = %s
        """
        params = [start_dt, end_dt, veh_class]

        if direction != 'all':
            sql += " AND direction = %s"
            params.append(direction)

        sql += " ORDER BY dt, sensor_id"

        with _PoolConnection(self._pool) as conn:
            df = pd.read_sql_query(sql, conn, params=params, parse_dates=['dt'])

        assert not df.empty, (
            f"CRITICAL: TimescaleConnector.load_traffic returned no rows for "
            f"veh_class={veh_class!r}, direction={direction!r}, "
            f"{start_dt} to {end_dt}"
        )
        required = {'dt', 'location', 'value', 'veh_class', 'dir'}
        assert required.issubset(df.columns), \
            f"CRITICAL: load_traffic missing columns. Got: {list(df.columns)}"
        assert df[['dt', 'location']].notna().all().all(), \
            "CRITICAL: NaN in primary key columns (dt, location) from load_traffic"

        logger.info(f"load_traffic: {len(df):,} rows, {df['location'].nunique()} sensors")
        return df

    def load_weather(self, start_dt, end_dt, sensor_ids: List[str]) -> pd.DataFrame:
        sql = """
            SELECT dt, sensor_id AS location,
                   humidity AS "Humidity",
                   solar_radiation AS "Solar Radiation"
            FROM weather_readings
            WHERE dt >= %s AND dt < %s AND sensor_id = ANY(%s)
            ORDER BY dt, sensor_id
        """
        with _PoolConnection(self._pool) as conn:
            df = pd.read_sql_query(
                sql, conn,
                params=[start_dt, end_dt, list(sensor_ids)],
                parse_dates=['dt']
            )

        assert not df.empty, (
            f"CRITICAL: TimescaleConnector.load_weather returned no rows for "
            f"{start_dt} to {end_dt}, {len(sensor_ids)} sensors"
        )
        required = {'dt', 'location', 'Humidity', 'Solar Radiation'}
        assert required.issubset(df.columns), \
            f"CRITICAL: load_weather missing columns. Got: {list(df.columns)}"
        assert df[['dt', 'location']].notna().all().all(), \
            "CRITICAL: NaN in primary key columns (dt, location) from load_weather"

        logger.info(f"load_weather: {len(df):,} rows")
        return df

    def load_static_features(self, sensor_ids: List[str]) -> pd.DataFrame:
        cols_sql = ", ".join(STATIC_FEATURE_COLS)
        sql = f"""
            SELECT sensor_id AS location, {cols_sql}
            FROM static_features
            WHERE sensor_id = ANY(%s)
        """
        with _PoolConnection(self._pool) as conn:
            df = pd.read_sql_query(sql, conn, params=[list(sensor_ids)])

        assert not df.empty, (
            f"CRITICAL: TimescaleConnector.load_static_features returned no rows "
            f"for {len(sensor_ids)} sensor IDs"
        )
        required = {'location'} | set(STATIC_FEATURE_COLS)
        assert required.issubset(df.columns), \
            f"CRITICAL: load_static_features missing columns. Got: {list(df.columns)}"
        assert df['location'].notna().all(), \
            "CRITICAL: NaN in location column from load_static_features"

        logger.info(f"load_static_features: {len(df)} rows")
        return df

    def load_sensor_metadata(self) -> Dict[str, Dict[str, float]]:
        sql = "SELECT sensor_id, lat, lon FROM sensor_metadata"
        with _PoolConnection(self._pool) as conn:
            df = pd.read_sql_query(sql, conn)

        assert not df.empty, "CRITICAL: sensor_metadata table is empty"

        return {
            row.sensor_id: {"lat": float(row.lat), "lon": float(row.lon)}
            for row in df.itertuples(index=False)
        }

    def close(self):
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None
            logger.info("TimescaleConnector: connection pool closed")


class ParquetConnector(DataConnector):
    """
    Smoke-test connector backed by local parquet and JSON files.
    Not for production use — no connection pooling, reads files on every call.
    """

    def __init__(self, parquet_path: str, static_cache_dir: str = "data/external"):
        assert parquet_path, "CRITICAL: parquet_path must be non-empty"
        assert Path(parquet_path).exists(), f"CRITICAL: parquet file not found: {parquet_path}"
        self._parquet_path = parquet_path
        self._static_cache_dir = static_cache_dir
        logger.info(f"ParquetConnector: using {parquet_path}")

    def load_traffic(self, start_dt, end_dt, veh_class: str, direction: str) -> pd.DataFrame:
        filters = [
            ('dt', '>=', pd.Timestamp(start_dt)),
            ('dt', '<', pd.Timestamp(end_dt)),
        ]
        df = pd.read_parquet(
            self._parquet_path,
            filters=filters,
            columns=['dt', 'location', 'value', 'veh_class', 'dir'],
        )
        df = df[df['veh_class'] == veh_class]
        if direction != 'all':
            df = df[df['dir'] == direction]

        nan_count = df['value'].isna().sum()
        assert nan_count == 0, (
            f"CRITICAL: ParquetConnector.load_traffic found {nan_count} NaN values in 'value'. "
            "The parquet file must be pre-cleaned before ingestion."
        )

        # Sort to match SQL 'ORDER BY dt, sensor_id'
        df = df.sort_values(['dt', 'location']).reset_index(drop=True)

        assert not df.empty, (
            f"CRITICAL: ParquetConnector.load_traffic returned no rows for "
            f"veh_class={veh_class!r}, direction={direction!r}, "
            f"{start_dt} to {end_dt}"
        )
        assert df[['dt', 'location']].notna().all().all(), \
            "CRITICAL: NaN in primary key columns (dt, location) from load_traffic"

        logger.info(f"load_traffic (parquet): {len(df):,} rows")
        return df

    def load_weather(self, start_dt, end_dt, sensor_ids: List[str]) -> pd.DataFrame:
        weather_cache_path = Path(self._static_cache_dir) / "weather_cache.parquet"
        weather_source = weather_cache_path if weather_cache_path.exists() else Path(self._parquet_path)

        schema = pq.read_schema(weather_source)
        dt_type = schema.field('dt').type
        is_tz_aware = getattr(dt_type, 'tz', None) is not None

        ts_start = pd.Timestamp(start_dt)
        ts_end = pd.Timestamp(end_dt)
        if is_tz_aware:
            ts_start = ts_start.tz_localize('UTC') if ts_start.tz is None else ts_start.tz_convert('UTC')
            ts_end = ts_end.tz_localize('UTC') if ts_end.tz is None else ts_end.tz_convert('UTC')
        else:
            ts_start = ts_start.tz_localize(None) if ts_start.tz is not None else ts_start
            ts_end = ts_end.tz_localize(None) if ts_end.tz is not None else ts_end

        filters = [
            ('dt', '>=', ts_start),
            ('dt', '<', ts_end),
        ]
        df = pd.read_parquet(
            weather_source,
            filters=filters,
            columns=['dt', 'location', 'Humidity', 'Solar Radiation'],
        )
        if df['dt'].dt.tz is not None:
            df['dt'] = df['dt'].dt.tz_localize(None)

        df = df[df['location'].isin(sensor_ids)]
        df = df.drop_duplicates(subset=['dt', 'location'])
        df = df[['dt', 'location', 'Humidity', 'Solar Radiation']]
        assert not df['Humidity'].isna().any(), \
            f"CRITICAL: ParquetConnector.load_weather: NaN in Humidity — {weather_source} must have real weather data"
        assert not df['Solar Radiation'].isna().any(), \
            f"CRITICAL: ParquetConnector.load_weather: NaN in Solar Radiation — {weather_source} must have real weather data"

        assert not df.empty, (
            f"CRITICAL: ParquetConnector.load_weather returned no rows for "
            f"{start_dt} to {end_dt} from {weather_source}"
        )
        assert df[['dt', 'location']].notna().all().all(), \
            "CRITICAL: NaN in primary key columns from load_weather"

        logger.info(f"load_weather (parquet from {weather_source.name}): {len(df):,} rows")
        return df

    def load_static_features(self, sensor_ids: List[str]) -> pd.DataFrame:
        places_path = Path(self._static_cache_dir) / "places_cache.json"
        osm_path = Path(self._static_cache_dir) / "osm_features_cache.json"

        assert places_path.exists(), f"CRITICAL: places_cache.json not found at {places_path}"
        assert osm_path.exists(), f"CRITICAL: osm_features_cache.json not found at {osm_path}"

        with open(places_path) as f:
            places = json.load(f)
        with open(osm_path) as f:
            osm = json.load(f)

        rows = []
        for sensor_id in sensor_ids:
            if sensor_id not in places:
                raise KeyError(f"CRITICAL: sensor '{sensor_id}' not found in places_cache.json")
            if sensor_id not in osm:
                raise KeyError(f"CRITICAL: sensor '{sensor_id}' not found in osm_features_cache.json")
            p = places[sensor_id]
            o = osm[sensor_id]
            rows.append({
                'location': sensor_id,
                'transit_count': float(p['transit_count']),
                'retail_count': float(p['retail_count']),
                'food_drink_count': float(p['food_drink_count']),
                'nightlife_count': float(p['nightlife_count']),
                'education_count': float(p['education_count']),
                'recreation_count': float(p['recreation_count']),
                'total_poi_count': float(p['total_poi_count']),
                'landuse_entropy': float(p['landuse_entropy']),
                'is_school_zone': float(1 if p['is_school_zone'] else 0),
                'betweenness_centrality': float(o['betweenness_centrality']),
                'closeness_centrality': float(o['closeness_centrality']),
                'dist_to_city_centre': float(p['dist_to_city_centre']),
                'degree_centrality': float(o['degree_centrality']),
                'nearby_bus_stops': float(p['nearby_bus_stops']),
                'nearby_parking': float(p['nearby_parking']),
            })

        df = pd.DataFrame(rows)
        assert not df.empty, "CRITICAL: ParquetConnector.load_static_features returned no rows"

        logger.info(f"load_static_features (parquet): {len(df)} rows")
        return df

    def load_sensor_metadata(self) -> Dict[str, Dict[str, float]]:
        sensors_path = Path(self._static_cache_dir).parent / "traffic_sensors.json"
        assert sensors_path.exists(), f"CRITICAL: traffic_sensors.json not found at {sensors_path}"

        with open(sensors_path) as f:
            raw = json.load(f)

        assert isinstance(raw, list) and len(raw) > 0, \
            "CRITICAL: traffic_sensors.json must be a non-empty list"

        return {
            entry["location"]: {"lat": float(entry["lat"]), "lon": float(entry["lon"])}
            for entry in raw
        }

    def close(self):
        pass  # no-op: no persistent resources


def create_connector(config: dict) -> DataConnector:
    """
    Factory function that creates the appropriate DataConnector from config.

    Config must contain:
        data_source: "database" | "parquet"

    For "database":   reads DATABASE_URL from environment — crashes if unset.
    For "parquet":    reads parquet_path from config (default: master_dataset_15T.parquet).

    No fallback. If data_source is wrong or DB is unreachable, this function crashes.
    """
    data_source = config.get("data_source")

    if data_source == "database":
        db_url = os.getenv("DATABASE_URL")
        assert db_url, (
            "CRITICAL: data_source='database' but DATABASE_URL is not set in environment. "
            "Set DATABASE_URL or switch to data_source='parquet'."
        )
        return TimescaleConnector(db_url)

    elif data_source == "parquet":
        if "parquet_path" not in config:
            raise KeyError(
                "CRITICAL: data_source='parquet' requires 'parquet_path' in config. "
                f"No default is applied — set parquet_path explicitly."
            )
        return ParquetConnector(config["parquet_path"])

    else:
        raise ValueError(
            f"CRITICAL: data_source must be 'database' or 'parquet', got: {data_source!r}. "
            f"Update your config YAML."
        )
