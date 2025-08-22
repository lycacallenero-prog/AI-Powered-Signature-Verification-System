#!/usr/bin/env python3
"""
Script to fix the model state by resetting and clearing old incompatible models
"""

import requests
import json
import time

def reset_model():
    """Reset the model to clear old incompatible files"""
    try:
        print("Resetting model to clear old incompatible files...")
        response = requests.delete('http://localhost:8000/reset_model', timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
            return True
        else:
            print(f"❌ Reset failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Reset error: {e}")
        return False

def check_model_status():
    """Check the current model status"""
    try:
        response = requests.get('http://localhost:8000/model_status', timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"Model Status:")
            print(f"  - Trained: {status.get('trained', False)}")
            print(f"  - Model Loaded: {status.get('model_loaded', False)}")
            print(f"  - Signatures: {status.get('num_signatures', 0)}")
            print(f"  - Threshold: {status.get('verification_threshold', 'None')}")
            return status
        else:
            print(f"❌ Status check failed: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return None

def main():
    """Main function to fix the model state"""
    print("AI Signature Verification - Model State Fix")
    print("=" * 50)
    
    # Check current status
    print("1. Checking current model status...")
    status = check_model_status()
    
    if status and status.get('trained', False):
        print("✅ Model shows as trained - no action needed!")
        return
    
    # Reset the model to clear old files
    print("\n2. Resetting model to clear old incompatible files...")
    if reset_model():
        print("✅ Model reset successful!")
        
        # Wait a moment for the reset to take effect
        time.sleep(2)
        
        # Check status again
        print("\n3. Checking status after reset...")
        status = check_model_status()
        
        if status:
            print("\n✅ Model state has been reset!")
            print("💡 You can now retrain the model and it should work properly.")
        else:
            print("\n❌ Could not verify reset status")
    else:
        print("❌ Model reset failed")

if __name__ == "__main__":
    main()