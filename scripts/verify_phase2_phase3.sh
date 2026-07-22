#!/bin/bash
# Verification script for Phase 2 (Training Database) and Phase 3 (CDN Migration)
# Usage: bash scripts/verify_phase2_phase3.sh [--phase2-only|--phase3-only]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILED=0

PHASE2_ONLY=false
PHASE3_ONLY=false

if [[ "${1:-}" == "--phase2-only" ]]; then
    PHASE2_ONLY=true
elif [[ "${1:-}" == "--phase3-only" ]]; then
    PHASE3_ONLY=true
fi

echo "=========================================="
echo "Phase 2 & 3 Verification Script"
echo "=========================================="
echo ""

# Helper functions
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    FAILED=1
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# ==========================================
# Phase 2: Training Database Verification
# ==========================================

if [[ "$PHASE3_ONLY" == false ]]; then
    echo "=========================================="
    echo "Phase 2: Training Database"
    echo "=========================================="
    echo ""

    # 1. Check docker-compose.yml
    echo "1. Checking docker-compose.yml..."
    if grep -q "timescaledb:" docker-compose.yml; then
        check_pass "TimescaleDB service defined in docker-compose.yml"
    else
        check_fail "TimescaleDB service NOT found in docker-compose.yml"
    fi

    # 2. Check .env
    echo "2. Checking .env..."
    if grep -q "DATABASE_URL" .env 2>/dev/null; then
        check_pass "DATABASE_URL found in .env"
    else
        check_fail "DATABASE_URL NOT found in .env"
    fi

    # 3. Check Alembic setup
    echo "3. Checking Alembic setup..."
    if [[ -f "alembic.ini" ]] && [[ -d "alembic/versions" ]]; then
        check_pass "Alembic initialized"
    else
        check_fail "Alembic NOT initialized"
    fi

    if [[ -f "alembic/versions/001_initial_schema.py" ]]; then
        check_pass "Initial migration (001) exists"
    else
        check_fail "Initial migration (001) NOT found"
    fi

    # 4. Check ingestion script
    echo "4. Checking ingestion script..."
    if [[ -f "scripts/ingestion/ingest_parquet.py" ]]; then
        check_pass "Ingestion script exists"
    else
        check_fail "Ingestion script NOT found at scripts/ingestion/ingest_parquet.py"
    fi

    # 5. Check dataset.py modifications
    echo "5. Checking dataset.py modifications..."
    if grep -q "use_database" src/dataset.py; then
        check_pass "use_database parameter added to dataset.py"
    else
        check_fail "use_database parameter NOT found in dataset.py"
    fi

    if grep -q "_load_from_database" src/dataset.py; then
        check_pass "_load_from_database method exists"
    else
        check_fail "_load_from_database method NOT found"
    fi

    # 6. Check train.py modifications
    echo "6. Checking train.py modifications..."
    if grep -q -- "--use-database" src/train.py; then
        check_pass "--use-database flag added to train.py"
    else
        check_fail "--use-database flag NOT found in train.py"
    fi

    # 7. Check TimescaleDB container (if running)
    echo "7. Checking TimescaleDB container..."
    PG_USER="${POSTGRES_USER:?POSTGRES_USER env var must be set}"
    PG_DB="${POSTGRES_DB:?POSTGRES_DB env var must be set}"
    if docker ps --format '{{.Names}}' | grep -q "ml-timescaledb"; then
        check_pass "TimescaleDB container is running"

        # Test database connection
        if docker exec ml-timescaledb pg_isready -U "$PG_USER" > /dev/null 2>&1; then
            check_pass "Database is accepting connections"
        else
            check_warn "Database container running but not ready"
        fi

        # Check if migrations applied
        if docker exec ml-timescaledb psql -U "$PG_USER" -d "$PG_DB" -c "\dt" 2>/dev/null | grep -q "traffic_readings"; then
            check_pass "Migrations applied (traffic_readings table exists)"
        else
            check_warn "Migrations NOT applied (run: alembic upgrade head)"
        fi

        # Check row count
        ROW_COUNT=$(docker exec ml-timescaledb psql -U "$PG_USER" -d "$PG_DB" -t -c "SELECT COUNT(*) FROM traffic_readings;" 2>/dev/null | tr -d ' ' || echo "0")
        if [[ "$ROW_COUNT" -gt 1000000 ]]; then
            check_pass "Data ingested ($ROW_COUNT rows)"
        elif [[ "$ROW_COUNT" -gt 0 ]]; then
            check_warn "Partial data ingested ($ROW_COUNT rows)"
        else
            check_warn "No data ingested (run: python scripts/ingestion/ingest_parquet.py ...)"
        fi
    else
        check_warn "TimescaleDB container NOT running (run: docker-compose up -d timescaledb)"
    fi

    echo ""
fi

# ==========================================
# Phase 3: CDN Migration Verification
# ==========================================

if [[ "$PHASE2_ONLY" == false ]]; then
    echo "=========================================="
    echo "Phase 3: CDN Migration"
    echo "=========================================="
    echo ""

    # 1. Check next.config.ts
    echo "1. Checking next.config.ts..."
    if grep -q "async rewrites()" frontend/next.config.ts; then
        check_pass "Rewrites configured in next.config.ts"
    else
        check_fail "Rewrites NOT configured"
    fi

    if grep -q "async headers()" frontend/next.config.ts; then
        check_pass "Headers configured in next.config.ts"
    else
        check_fail "Headers NOT configured"
    fi

    # 2. Check payloadSource.ts
    echo "2. Checking payloadSource.ts..."
    if grep -q '`/\${PREFIX}/run_registry.json`' frontend/src/lib/payloadSource.ts 2>/dev/null || \
       grep -q '"/data/runs"' frontend/src/lib/payloadSource.ts 2>/dev/null; then
        check_pass "payloadSource.ts refactored to relative URLs"
    else
        check_fail "payloadSource.ts NOT refactored"
    fi

    if grep -q "PAYLOAD_BASE" frontend/src/lib/payloadSource.ts 2>/dev/null; then
        check_warn "PAYLOAD_BASE still present (should be removed)"
    else
        check_pass "PAYLOAD_BASE removed from payloadSource.ts"
    fi

    # 3. Check GitHub Actions workflow
    echo "3. Checking GitHub Actions workflow..."
    if [[ -f ".github/workflows/publish_payloads.yml" ]]; then
        check_pass "Publish workflow exists"

        if grep -q "self-hosted" .github/workflows/publish_payloads.yml; then
            check_pass "Workflow configured for self-hosted runner"
        else
            check_warn "Workflow NOT configured for self-hosted runner"
        fi
    else
        check_fail "Publish workflow NOT found"
    fi

    # 4. Check documentation
    echo "4. Checking documentation..."
    if [[ -f "docs/SELF_HOSTED_RUNNER_SETUP.md" ]]; then
        check_pass "Self-hosted runner documentation exists"
    else
        check_fail "Self-hosted runner documentation NOT found"
    fi

    # 5. Check Next.js dev server (if running)
    echo "5. Checking Next.js dev server..."
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        check_pass "Next.js dev server is running"

        # Test proxy rewrite (requires R2 bucket to be populated)
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/data/runs/run_registry.json 2>/dev/null)
        if [ "$HTTP_CODE" = "200" ]; then
            check_pass "Proxy rewrite working (registry accessible)"
        else
            check_warn "Proxy rewrite may not be working (or R2 bucket empty, HTTP $HTTP_CODE)"
        fi
    else
        check_warn "Next.js dev server NOT running (run: cd frontend && npm run dev)"
    fi

    echo ""
fi

# ==========================================
# Final Summary
# ==========================================

echo "=========================================="
echo "Verification Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""

if [[ "$PHASE3_ONLY" == false ]]; then
    echo "Phase 2:"
    echo "  1. Start TimescaleDB:    docker-compose up -d timescaledb"
    echo "  2. Apply migrations:     alembic upgrade head"
    echo "  3. Ingest data (2-3h):   python scripts/ingestion/ingest_parquet.py --parquet data/processed/master_dataset_15T.parquet --create-dataset-version"
    echo "  4. Test database mode:   python src/train.py --use-database --start-date 2026-06-01 --end-date 2026-06-02"
    echo ""
fi

if [[ "$PHASE2_ONLY" == false ]]; then
    echo "Phase 3:"
    echo "  1. Set up runner (GPU):  Follow docs/SELF_HOSTED_RUNNER_SETUP.md"
    echo "  2. Add GitHub secrets:   WANDB_API_KEY, R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, etc."
    echo "  3. Test workflow:        Actions → Build and Publish Payloads to R2 → Run workflow"
    echo "  4. Verify deployment:    vercel deploy --prod"
    echo ""
fi

echo "Full verification checklist: .agents/IMPLEMENTATION_SUMMARY.md"
echo ""

exit "$FAILED"
