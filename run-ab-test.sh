#!/bin/bash
# Run A/B comparison: batch games with each config profile.
#
# Usage:
#   bash run-ab-test.sh          # 3 games each (default)
#   bash run-ab-test.sh 5        # 5 games each
#
# After it finishes, run the encounter report to compare:
#   python3 encounter-report.py

GAMES="${1:-3}"

echo "════════════════════════════════════════════"
echo "  A/B Test: $GAMES games per profile"
echo "════════════════════════════════════════════"
echo ""

echo ">> Profile A (Champion) — $GAMES games"
echo "────────────────────────────────────────────"
for i in $(seq 1 "$GAMES"); do
    echo "  [A] Game $i of $GAMES"
    cd ~/AJS_CTS/ClawTheSpire && STS2_CONFIG_PROFILE=a bash play.sh batch --once
done

echo ""
echo ">> Profile B (Challenger) — $GAMES games"
echo "────────────────────────────────────────────"
for i in $(seq 1 "$GAMES"); do
    echo "  [B] Game $i of $GAMES"
    cd ~/AJS_CTS/ClawTheSpire && STS2_CONFIG_PROFILE=b bash play.sh batch --once
done

echo ""
echo "════════════════════════════════════════════"
echo "  Done! Run the encounter report to compare:"
echo "  cd ~/AJS_CTS/ClawTheSpire && python3 encounter-report.py"
echo "════════════════════════════════════════════"
