import json
import pytest
from pathlib import Path
from detection.active_learning.queue_io import load_queue, save_queue

SECRET = "super-secret-key-123"

@pytest.fixture
def base_annotations():
    return [
        {"wallet": "GABC...", "label": "wash_trade"},
        {"wallet": "GXYZ...", "label": "organic"}
    ]

def test_load_queue_success(tmp_path, base_annotations):
    """AC: load_queue succeeds on file written by save_queue with matching secret"""
    file_path = tmp_path / "valid_queue.json"
    save_queue(file_path, base_annotations, SECRET)
    
    annotations = load_queue(file_path, SECRET)
    assert len(annotations) == 2
    assert annotations[0]['wallet'] == "GABC..."

def test_load_queue_tampered_body(tmp_path, base_annotations):
    """AC: load_queue raises ValueError on tampered file body"""
    file_path = tmp_path / "tampered_body.json"
    save_queue(file_path, base_annotations, SECRET)
    
    # Simulate an attacker changing a wallet label in transit
    raw_data = json.loads(file_path.read_text())
    raw_data['annotations'][0]['label'] = 'organic'  # Poisoned target
    file_path.write_text(json.dumps(raw_data))
    
    with pytest.raises(ValueError, match="Annotation queue HMAC mismatch"):
        load_queue(file_path, SECRET)

def test_load_queue_tampered_mac(tmp_path, base_annotations):
    """AC: load_queue raises ValueError on an invalid/arbitrary MAC string"""
    file_path = tmp_path / "tampered_mac.json"
    save_queue(file_path, base_annotations, SECRET)
    
    # Overwrite the signature block with garbage values
    raw_data = json.loads(file_path.read_text())
    raw_data['_hmac'] = "badmac12345"
    file_path.write_text(json.dumps(raw_data))
    
    with pytest.raises(ValueError, match="Annotation queue HMAC mismatch"):
        load_queue(file_path, SECRET)

def test_load_queue_missing_mac(tmp_path, base_annotations):
    """AC: load_queue raises ValueError when the MAC attribute is missing"""
    file_path = tmp_path / "missing_mac.json"
    # Write a clean file object but completely strip the signature hook
    file_path.write_text(json.dumps({"annotations": base_annotations}))
    
    with pytest.raises(ValueError, match="Annotation queue HMAC mismatch"):
        load_queue(file_path, SECRET)

def test_load_queue_empty_secret_skips(tmp_path, base_annotations, caplog):
    """AC: Empty secret skips verification with WARNING"""
    file_path = tmp_path / "unverified.json"
    save_queue(file_path, base_annotations, secret=SECRET)
    
    import logging
    with caplog.at_level(logging.WARNING):
        # Passing an empty string secret overrides verification checks
        annotations = load_queue(file_path, secret="")
        
    assert len(annotations) == 2
    assert "Skipping signature verification" in caplog.text
