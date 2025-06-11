#!/bin/bash

# Usage: ./run_pipeline.sh IT_Services

BUCKET=$1

if [ -z "$BUCKET" ]; then
  echo "❌ Please provide a bucket name (e.g., IT_Services)"
  exit 1
fi

echo "📦 Starting pipeline for bucket: $BUCKET"

# Step 1: Train ensemble model
echo "🎯 Step 1: Training models..."
python scripts/train_ensemble_model.py --bucket $BUCKET

# Step 2: Tune thresholds
echo "🧪 Step 2: Tuning thresholds..."
python scripts/tune_thresholds.py --bucket $BUCKET

# Step 3: Generate signals with ensemble logic
echo "📈 Step 3: Generating signals..."
python scripts/generate_signals.py --bucket $BUCKET

# Step 4: Backtest multiple exit strategies
echo "🔁 Step 4: Backtesting exit strategies..."
python scripts/backtest_exit.py --bucket $BUCKET

echo "✅ Pipeline completed for bucket: $BUCKET"
  