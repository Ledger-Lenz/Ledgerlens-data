#!/bin/bash
#
# Run WebSocket integration tests
#
# This script runs the WebSocket server integration tests in tests/test_ws_integration.py
# The tests spin up a real WebSocket server and test JWT authentication end-to-end.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  WebSocket Integration Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ Error: pytest is not installed"
    echo ""
    echo "Please install dependencies:"
    echo "  pip install -r requirements.txt"
    echo ""
    exit 1
fi

# Check if required packages are installed
python3 -c "import websockets" 2>/dev/null || {
    echo "❌ Error: websockets module not found"
    echo ""
    echo "Please install dependencies:"
    echo "  pip install -r requirements.txt"
    echo ""
    exit 1
}

python3 -c "import jose" 2>/dev/null || {
    echo "❌ Error: python-jose module not found"
    echo ""
    echo "Please install dependencies:"
    echo "  pip install -r requirements.txt"
    echo ""
    exit 1
}

# Run the tests
echo "Running WebSocket integration tests..."
echo ""

if [ "$1" == "--verbose" ] || [ "$1" == "-v" ]; then
    pytest tests/test_ws_integration.py -v -s
else
    pytest tests/test_ws_integration.py -v
fi

RESULT=$?

echo ""
if [ $RESULT -eq 0 ]; then
    echo "✅ All WebSocket integration tests passed!"
else
    echo "❌ Some tests failed (exit code: $RESULT)"
fi
echo ""

exit $RESULT
