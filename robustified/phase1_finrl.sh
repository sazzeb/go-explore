
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

CSV_PATH="$(pwd)/finrl_goexplore/data/aapl_2016_2020.csv"
ENV_ID="finrl:${CSV_PATH}:AAPL"

if [[ ! -f "$CSV_PATH" ]]; then
  echo "Missing dataset: $CSV_PATH"
  echo "Generating via finrl_goexplore/make_data.py ..."
  mkdir -p "$(dirname "$CSV_PATH")"
  python finrl_goexplore/make_data.py \
    --tickers AAPL \
    --start 2016-01-01 \
    --end 2020-01-01 \
    --out "$CSV_PATH"
fi

BASE_PATH="${1:-runs_finrl}"
ITERS="${2:-200}"

python goexplore_py/main.py \
  --game "$ENV_ID" \
  --base_path "$BASE_PATH" \
  --max_iterations "$ITERS" \
  --use_scores \
  --state_is_pixels \
  --seen_weight 1.0

