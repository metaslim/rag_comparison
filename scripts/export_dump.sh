#!/bin/bash
# Export Neo4j graph to a named dump file.
# Usage: bash scripts/export_dump.sh [name]
# Default name: neo4j
# Examples:
#   bash scripts/export_dump.sh        → data/neo4j_dump/neo4j.dump
#   bash scripts/export_dump.sh dev    → data/neo4j_dump/dev.dump
#   bash scripts/export_dump.sh train  → data/neo4j_dump/train.dump

set -euo pipefail

NAME="${1:-neo4j}"
DUMP_DIR="data/neo4j_dump"
DUMP_FILE="$DUMP_DIR/$NAME.dump"

mkdir -p "$DUMP_DIR"

echo "Starting Neo4j for checkpoint..."
docker-compose start neo4j
sleep 20

echo "Flushing checkpoint..."
docker exec metaslim-neo4j-1 \
  cypher-shell -u neo4j -p password "CALL db.checkpoint()" 2>/dev/null \
  && echo "Checkpoint OK" || echo "Checkpoint skipped"

echo "Stopping Neo4j..."
docker-compose stop neo4j
docker wait metaslim-neo4j-1 2>/dev/null || true

echo "Dumping database to $DUMP_FILE..."
docker run --rm \
  -v "$(pwd)/data/neo4j/data:/data" \
  -v "$(pwd)/$DUMP_DIR:/dump" \
  neo4j:5.26 \
  neo4j-admin database dump neo4j --to-path=/dump/ --overwrite-destination=true

mv "$DUMP_DIR/neo4j.dump" "$DUMP_FILE" 2>/dev/null || true

echo "Restarting Neo4j..."
docker-compose start neo4j

echo ""
echo "✅ Done: $DUMP_FILE ($(du -sh "$DUMP_FILE" | cut -f1))"
