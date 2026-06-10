#!/bin/bash
# Restore Neo4j from a named dump file in data/neo4j_dump/.
# Usage: bash scripts/restore_dump.sh [name]
# Default name: dev (loads data/neo4j_dump/dev.dump)
# Examples:
#   bash scripts/restore_dump.sh         loads data/neo4j_dump/dev.dump
#   bash scripts/restore_dump.sh dev100  loads data/neo4j_dump/dev100.dump

set -euo pipefail

NAME="${1:-dev}"
DUMP_DIR="data/neo4j_dump"
DUMP_FILE="$DUMP_DIR/$NAME.dump"

[ -f "$DUMP_FILE" ] || { echo "No dump file at $DUMP_FILE"; exit 1; }

echo "Stopping Neo4j..."
docker-compose stop neo4j 2>/dev/null || true

echo "Clearing existing Neo4j data dir..."
mkdir -p data/neo4j/data
find data/neo4j/data -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +

echo "Staging $DUMP_FILE as neo4j.dump (neo4j-admin requires this name)..."
cp "$DUMP_FILE" "$DUMP_DIR/neo4j.dump"

echo "Loading dump..."
docker run --rm \
  -v "$(pwd)/data/neo4j/data:/data" \
  -v "$(pwd)/$DUMP_DIR:/dump" \
  neo4j:5.26 \
  neo4j-admin database load neo4j --from-path=/dump/ --overwrite-destination=true

echo "Removing staged neo4j.dump..."
rm -f "$DUMP_DIR/neo4j.dump"

echo "Starting Neo4j..."
docker-compose start neo4j

echo "Restore complete from $DUMP_FILE"
