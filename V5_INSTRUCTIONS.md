./scripts/riotv5/start_v5_suite.sh --gpus 0 1 2 3 --tiers 1
Or run it with no arguments for all 209. Tier 1 first is the safer choice: you get the headline comparisons early and can catch problems with real data before committing about 60 run-durations of GPU time. ./scripts/riotv5/show_v5_logs.sh re-attaches to the output, and logs/v5_suite/v5_suite_summary.csv fills in as runs finish.
