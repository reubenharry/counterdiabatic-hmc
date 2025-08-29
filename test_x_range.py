#!/usr/bin/env python3
"""
Test script to check the x-axis range issue.
"""

import pickle
import numpy as np

def main():
    # Load all simulation data
    methods = ['naive_unweighted', 'naive_weighted', 'cd_unweighted', 'cd_weighted']
    
    print("=== X-AXIS RANGE ANALYSIS ===")
    
    all_qs = []
    method_ranges = {}
    
    for method in methods:
        try:
            with open(f'data/double_well_{method}.pkl', 'rb') as f:
                data = pickle.load(f)
            
            snapshots = data['snapshots']
            particles = snapshots['particles']
            
            # Get range for this method
            method_qs = np.concatenate(particles)
            method_min = np.min(method_qs)
            method_max = np.max(method_qs)
            method_ranges[method] = (method_min, method_max)
            
            print(f"{method}:")
            print(f"  Range: [{method_min:.6f}, {method_max:.6f}]")
            print(f"  Width: {method_max - method_min:.6f}")
            
            all_qs.extend(particles)
            
        except Exception as e:
            print(f"{method}: Error - {e}")
    
    # Calculate global range
    all_qs_array = np.concatenate(all_qs)
    global_min = np.min(all_qs_array)
    global_max = np.max(all_qs_array)
    
    print(f"\nGlobal range: [{global_min:.6f}, {global_max:.6f}]")
    print(f"Global width: {global_max - global_min:.6f}")
    
    # Check what happens to CD methods on this global scale
    print(f"\n=== CD METHODS ON GLOBAL SCALE ===")
    for method in ['cd_unweighted', 'cd_weighted']:
        if method in method_ranges:
            cd_min, cd_max = method_ranges[method]
            cd_width = cd_max - cd_min
            
            # Calculate relative position on global scale
            rel_min = (cd_min - global_min) / (global_max - global_min)
            rel_max = (cd_max - global_min) / (global_max - global_min)
            rel_width = rel_max - rel_min
            
            print(f"{method}:")
            print(f"  CD range: [{cd_min:.6f}, {cd_max:.6f}] (width: {cd_width:.6f})")
            print(f"  Relative position: [{rel_min:.6f}, {rel_max:.6f}] (width: {rel_width:.6f})")
            print(f"  This means CD distributions are only {rel_width*100:.3f}% of the total plot width!")
            
            if rel_width < 0.01:
                print(f"  WARNING: CD distributions are essentially invisible!")

if __name__ == "__main__":
    main()
