#!/usr/bin/env python3
"""Standalone test script to verify database connection pooling implementation.

This script can be run directly to test the pooling functionality
without requiring pytest or other test dependencies.
"""

import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add the project root to Python path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_pooling_basic():
    """Basic test that pooling configuration is working."""
    print("Testing basic database pooling configuration...")
    
    try:
        # Mock the config values
        os.environ["DB_POOL_SIZE"] = "3"
        os.environ["DB_MAX_OVERFLOW"] = "5"
        os.environ["DB_POOL_TIMEOUT"] = "10"
        
        # Import after setting env vars
        from detection.persistence import get_engine
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            temp_db_url = f"sqlite:///{tmp_file.name}"
        
        try:
            engine = get_engine(temp_db_url)
            
            # Check pool configuration
            assert hasattr(engine.pool, 'size'), "Engine should have a pool"
            print(f"✓ Pool size: {engine.pool.size()}")
            print(f"✓ Pool class: {type(engine.pool).__name__}")
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute("SELECT 1").fetchone()
                assert result[0] == 1
                print("✓ Database connection works")
            
        finally:
            Path(temp_db_url.replace("sqlite:///", "")).unlink(missing_ok=True)
            
        print("✅ Basic pooling test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("This is expected if dependencies aren't installed")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_configuration_values():
    """Test that configuration values are correctly parsed."""
    print("\nTesting configuration value parsing...")
    
    try:
        # Set test values
        os.environ["DB_POOL_SIZE"] = "8"
        os.environ["DB_MAX_OVERFLOW"] = "12"
        os.environ["DB_POOL_TIMEOUT"] = "25"
        
        # Import config after setting env vars
        import importlib
        if 'config' in sys.modules:
            importlib.reload(sys.modules['config'])
        from config import config
        
        assert config.DB_POOL_SIZE == 8, f"Expected 8, got {config.DB_POOL_SIZE}"
        assert config.DB_MAX_OVERFLOW == 12, f"Expected 12, got {config.DB_MAX_OVERFLOW}"
        assert config.DB_POOL_TIMEOUT == 25, f"Expected 25, got {config.DB_POOL_TIMEOUT}"
        
        print(f"✓ DB_POOL_SIZE: {config.DB_POOL_SIZE}")
        print(f"✓ DB_MAX_OVERFLOW: {config.DB_MAX_OVERFLOW}")
        print(f"✓ DB_POOL_TIMEOUT: {config.DB_POOL_TIMEOUT}")
        
        print("✅ Configuration parsing test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_env_example_syntax():
    """Test that .env.example has valid syntax."""
    print("\nTesting .env.example file...")
    
    try:
        env_example_path = Path(__file__).parent.parent / ".env.example"
        
        if not env_example_path.exists():
            print("❌ .env.example file not found")
            return False
        
        with open(env_example_path) as f:
            content = f.read()
        
        # Check that our new config options are present
        required_options = [
            "DB_POOL_SIZE=5",
            "DB_MAX_OVERFLOW=10", 
            "DB_POOL_TIMEOUT=30"
        ]
        
        for option in required_options:
            if option not in content:
                print(f"❌ Missing option in .env.example: {option}")
                return False
            print(f"✓ Found: {option}")
        
        print("✅ .env.example validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Run all standalone tests."""
    print("🚀 Running standalone database pooling tests...")
    print("=" * 60)
    
    results = []
    
    results.append(test_env_example_syntax())
    results.append(test_configuration_values())
    results.append(test_pooling_basic())
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All standalone tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())