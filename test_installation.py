#!/usr/bin/env python3
"""
Test script to verify GEAK-agent installation is working correctly.
"""

import sys
import os
sys.path.append('src')

# For dataset test, we need to run from src directory for correct path resolution
TEST_FROM_SRC = True

def test_imports():
    """Test that all required modules can be imported."""
    try:
        from dataloaders.TritonBench import TritonBench
        from agents.GaAgent import GaAgent
        from models.OpenAI import OpenAIModel
        from models.Claude import ClaudeModel
        from args_config import load_config
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test that config can be loaded."""
    try:
        from args_config import load_config
        args = load_config('src/configs/tritonbench_gaagent_config.yaml')
        print("✅ Config loaded successfully")
        print(f"   📊 Model: {args.model_id}")
        print(f"   🔑 API key: {'SET' if args.api_key else 'NOT SET'}")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_dataset():
    """Test that dataset can be initialized."""
    try:
        from dataloaders.TritonBench import TritonBench
        from args_config import load_config
        
        args = load_config('src/configs/tritonbench_gaagent_config.yaml')
        dataset = TritonBench(
            statis_path=args.statis_path,
            py_folder=args.py_folder, 
            instruction_path=args.instruction_path,
            golden_metrics=args.golden_metrics,
            py_interpreter=args.py_interpreter,
            perf_ref_folder=args.perf_ref_folder,
            perf_G_path=args.perf_G_path,
            result_path=args.result_path
        )
        print("✅ Dataset initialized successfully")
        print(f"   📊 Dataset size: {len(dataset)} problems")
        return True
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing GEAK-agent installation...")
    print()
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration Loading", test_config), 
        ("Dataset Initialization", test_dataset)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"Testing {name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"📈 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 GEAK-agent installation is working correctly!")
        print("📋 Next steps:")
        print("   1. Set your API key in src/configs/tritonbench_gaagent_config.yaml")
        print("   2. Run: cd src && python3 main_gaagent.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())