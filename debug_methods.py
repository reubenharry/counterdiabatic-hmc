#!/usr/bin/env python3
"""
Debug script to check what methods are available in successful simulations.
"""

import pickle
import numpy as np

def main():
    # Load all simulation data
    methods = ['naive_unweighted', 'naive_weighted', 'cd_unweighted', 'cd_weighted']
    
    print("=== METHOD AVAILABILITY CHECK ===")
    
    for method in methods:
        try:
            with open(f'data/gaussian_annealing_{method}.pkl', 'rb') as f:
                data = pickle.load(f)
            print(f"✓ {method}: Available")
            print(f"  Keys: {list(data.keys())}")
            if 'snapshots' in data:
                snapshots = data['snapshots']
                print(f"  Snapshot keys: {list(snapshots.keys())}")
                if 'particles' in snapshots:
                    print(f"  Number of time steps: {len(snapshots['particles'])}")
        except FileNotFoundError:
            print(f"✗ {method}: Not found")
        except Exception as e:
            print(f"✗ {method}: Error - {e}")
        print()

if __name__ == "__main__":
    main()
