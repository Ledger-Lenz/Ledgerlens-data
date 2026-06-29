# LedgerLens Test Suite

## Test Data Factory

The test data factory (`tests/factories.py`) provides realistic, reusable fixtures for constructing Stellar trade data with various patterns. All generated wallet addresses are valid Stellar G-prefixed account IDs.

### Factory Classes

#### `CleanTradeFactory`

Generates Benford-conforming trades resembling genuine market activity:
- **Amounts**: Log-uniform distribution in [10^2, 10^8] — naturally conforms to Benford's Law
- **Timing**: Realistic inter-trade intervals via Poisson process
- **Counterparties**: Diverse, varied wallets

**Guarantee**: Batches of >100 trades pass chi-square test at 5% significance level.

```python
from tests.factories import CleanTradeFactory, make_clean_trades

# Single trade
trade = CleanTradeFactory.build()

# Batch of 150 trades (returns list of dicts)
trades = make_clean_trades(n=150)

# Convert to DataFrame for analysis
import pandas as pd
df = pd.DataFrame(trades)
```

#### `WashTradeFactory`

Generates non-Benford wash-trading patterns (round amounts, constant intervals, same counterparty):
- **Amounts**: Round numbers [500, 1000, 5000, 10000, 50000]
- **Timing**: Fixed intervals (e.g., every 5 seconds)
- **Counterparties**: Same sock-puppet wallet repeated

**Guarantee**: Batches of >30 trades FAIL chi-square test at 5% significance level.

```python
from tests.factories import make_wash_trades

trades = make_wash_trades(n=50)
```

#### `RingTradeFactory`

Generates coordinated ring-trading patterns (N wallets trading in a circle):
- **Ring size**: Configurable (default 5 wallets)
- **Amounts**: Slightly variable (±10% jitter) to appear organic
- **Timing**: Loosely coordinated (~5-minute spread)

```python
from tests.factories import make_ring_trades

# Generate 30 trades across a 5-wallet ring
trades = make_ring_trades(n=30, ring_size=5)
```

### Pytest Fixture

A global pytest fixture `synthetic_stellar_trades` is available in all tests via `conftest.py`:

```python
def test_my_feature(synthetic_stellar_trades):
    clean_trades = synthetic_stellar_trades(n=100, pattern='clean')
    wash_trades = synthetic_stellar_trades(n=50, pattern='wash')
    ring_trades = synthetic_stellar_trades(n=30, pattern='ring')
    
    # Each is a list of trade dicts ready for ingestion
    assert len(clean_trades) == 100
```

### Adding New Fixture Patterns

1. Create a new Factory subclass in `tests/factories.py`:

```python
class YourPatternFactory(Factory):
    class Meta:
        model = Trade
    
    trade_id = Sequence(lambda n: f"pattern-{n}")
    base_account = LazyAttribute(lambda o: generate_stellar_account_id(...))
    # ... define other fields
    
    @LazyAttribute
    def base_amount(o) -> float:
        # Custom amount generation
        return your_amount_logic()
```

2. Create a helper function:

```python
def make_your_pattern_trades(n: int = 50) -> list[dict]:
    return [YourPatternFactory.build().__dict__ for _ in range(n)]
```

3. Register it in the `synthetic_stellar_trades` fixture's `_make_trades` function:

```python
elif pattern == 'your_pattern':
    return make_your_pattern_trades(n)
```

4. Add self-tests to `tests/test_factories.py` verifying the pattern's properties.

### Test Isolation & Reproducibility

- Each factory call is **isolated**: no shared state between builds
- Random seed is reset before each test via `reset_random_state()` fixture in `conftest.py`
- Stellar account IDs are deterministic when seeded (for reproducible test runs)

### Benford Conformance Validation

Both `CleanTradeFactory` and `WashTradeFactory` are self-testing via `tests/test_factories.py`:

| Test | Factory | Expected Result |
|------|---------|-----------------|
| `test_benford_conformance_large_batch` | Clean | Chi-square < 20 (pass Benford) |
| `test_benford_mad_score` | Clean | MAD < 0.015 (conform) |
| `test_wash_trades_fail_benford_chi_square` | Wash | Chi-square > 15 (fail Benford) |
| `test_wash_trades_high_mad` | Wash | MAD > 0.015 (non-conform) |

Run them with:

```bash
pytest tests/test_factories.py -v
```

### Usage Examples

#### Feature Engineering Tests

```python
from tests.factories import make_clean_trades
import pandas as pd
from detection.feature_engineering import build_feature_matrix

def test_features_on_realistic_data(synthetic_stellar_trades):
    trades_list = synthetic_stellar_trades(n=100, pattern='clean')
    df = pd.DataFrame(trades_list)
    
    features = build_feature_matrix(df)
    assert 'benford_chi_square_1h' in features.columns
```

#### Model Training Tests

```python
from tests.factories import make_clean_trades, make_wash_trades

def test_model_separates_patterns():
    clean = pd.DataFrame(make_clean_trades(n=100))
    wash = pd.DataFrame(make_wash_trades(n=100))
    
    combined = pd.concat([clean, wash], ignore_index=True)
    # Train and verify model can separate them
```

## Test Conventions

- **Test files**: Prefix with `test_` (e.g., `test_feature_engineering.py`)
- **Test functions**: Prefix with `test_` (e.g., `test_compute_features()`)
- **Fixtures**: Use lowercase snake_case (e.g., `synthetic_stellar_trades`)
- **Setup/Teardown**: Use pytest `@pytest.fixture` (not class-based `setUp`/`tearDown`)
- **Mocking**: Use `unittest.mock` for external dependencies (Horizon API, database)

## Running Tests

```bash
# Run all tests
make test

# Run a specific test file
pytest tests/test_features.py -v

# Run a specific test
pytest tests/test_factories.py::TestCleanTradeFactory::test_benford_conformance_large_batch -v

# Run with coverage
pytest --cov=detection --cov=ingestion tests/
```

## Debugging

- Use `pytest -s` to see print statements
- Use `pytest --pdb` to drop into debugger on failure
- Use `pytest -k "pattern"` to run only tests matching a pattern
