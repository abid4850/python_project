"""
🔍 NOTEBOOK VERIFICATION SCRIPT
Validates that all imports and basic functionality work
Run this before executing the full notebook
"""

import sys
print("="*80)
print("🔍 NOTEBOOK VERIFICATION TEST")
print("="*80)

# Test 1: Basic Imports
print("\n✓ Test 1: Core Imports")
try:
    import numpy as np
    import pandas as pd
    print("  ✅ NumPy, Pandas")
except Exception as e:
    print(f"  ❌ NumPy/Pandas: {e}")
    sys.exit(1)

# Test 2: System utilities
print("\n✓ Test 2: System Utilities")
try:
    import sys
    import os
    from contextlib import contextmanager
    print("  ✅ sys, os, contextmanager")
except Exception as e:
    print(f"  ❌ System utils: {e}")
    sys.exit(1)

# Test 3: Visualization
print("\n✓ Test 3: Visualization Libraries")
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    print("  ✅ Matplotlib, Seaborn, Plotly")
except Exception as e:
    print(f"  ❌ Visualization: {e}")
    sys.exit(1)

# Test 4: ML & Statistics
print("\n✓ Test 4: ML & Statistics")
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    import scipy.stats as stats
    print("  ✅ Scikit-learn, SciPy")
except Exception as e:
    print(f"  ❌ ML/Stats: {e}")
    sys.exit(1)

# Test 5: Time Series
print("\n✓ Test 5: Time Series & Prophet")
try:
    from prophet import Prophet
    print("  ✅ Prophet")
except Exception as e:
    print(f"  ❌ Prophet: {e}")
    sys.exit(1)

# Test 6: Deep Learning
print("\n✓ Test 6: Deep Learning (TensorFlow/Keras)")
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    print("  ✅ TensorFlow, Keras, LSTM layers")
except Exception as e:
    print(f"  ❌ Deep Learning: {e}")
    sys.exit(1)

# Test 7: Finance
print("\n✓ Test 7: Finance Data")
try:
    import yfinance as yf
    print("  ✅ yfinance")
except Exception as e:
    print(f"  ⚠️  yfinance (optional): {e}")

# Test 8: Context Manager
print("\n✓ Test 8: Context Manager Function")
try:
    @contextmanager
    def suppress_test():
        """Test context manager"""
        try:
            yield
        except:
            pass
    
    with suppress_test():
        x = 1
    print("  ✅ Context manager working")
except Exception as e:
    print(f"  ❌ Context manager: {e}")
    sys.exit(1)

# Test 9: Data Creation
print("\n✓ Test 9: Sample Data Creation")
try:
    ev_data = {
        "Year": list(range(2005, 2026)),
        "China": [0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.90, 1.50, 2.10,
                  3.50, 5.00, 7.00, 10.00, 13.00, 15.50, 17.00, 18.50, 19.50, 20.00]
    }
    test_df = pd.DataFrame(ev_data)
    print(f"  ✅ DataFrame created: {test_df.shape}")
except Exception as e:
    print(f"  ❌ Data creation: {e}")
    sys.exit(1)

# Test 10: Numpy Operations
print("\n✓ Test 10: Numpy Arrays")
try:
    arr = np.array([1, 2, 3, 4, 5])
    result = np.mean(arr)
    print(f"  ✅ Numpy operations working (mean={result:.1f})")
except Exception as e:
    print(f"  ❌ Numpy: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("🎉 ALL VERIFICATION TESTS PASSED!")
print("="*80)
print("\n📊 Environment Status:")
print(f"  Python Version: {sys.version.split()[0]}")
print(f"  NumPy: {np.__version__}")
print(f"  Pandas: {pd.__version__}")
print(f"  Scikit-learn: {__import__('sklearn').__version__}")
print(f"  Plotly: {px.__version__}")

print("\n✅ Your notebook is ready to run!")
print("   Execute: EV_Market_Kaggle_Grandmaster.ipynb")
print("   Expected Runtime: 2-3 minutes")
print("="*80)
