We should have one source of truth for the entire codebase for random seeds. Perhaps in the config files under "global.yaml" or something. This seed should be used everywhere.

GLOBAL_YAML I think this should include all of the global parameters. Ie. num_workers and anything that is not specific to the model itself. All config values should be here which allows to remove all defaults from the codebase.

FRONTEND_YAML I think that this should include all of the values used to configure the frontend.


Invalid config silently corrected — src/pipeline.py:38-39: a configured dim_feedforward <= 0 is silently replaced with hidden_dim * 4 instead of rejected. This should just follow hidden_dim * 4 and can remove configurabililty for dim_feedforward.

This can be removed we will parameterise from the shell execution scripts so that these variables are not missed.


scripts/ingestion/ingest_weather.py:65-67 — placeholder NaN weather is fillna(0.0)-ed to satisfy the NOT NULL constraint and ingested as real readings: zero humidity/solar values are fabricated into the training database. For this we are planning to implement a new weather API to pull this data from. The old API removed the sensor variables we needed so currently there is no weather data.