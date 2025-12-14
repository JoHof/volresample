#!/usr/bin/env python3
"""Create a visual comparison chart of all modes."""

def print_performance_chart():
    """Print ASCII chart of performance comparison."""
    
    results = {
        'Nearest': {
            'upsample': 0.52,
            'downsample': 0.28,
            '4D': 0.46,
            'mean': 0.42
        },
        'Linear': {
            'upsample': 0.67,
            'downsample': 0.55,
            '4D': 0.59,
            'mean': 0.60
        },
        'Area': {
            'upsample': 15.80,
            'downsample': 1.36,
            '4D': 0.47,
            'mean': 5.88
        }
    }
    
    print("\n" + "="*80)
    print(" PERFORMANCE COMPARISON: Cython vs PyTorch")
    print(" (Speedup Factor: >1.0 means Cython is faster)")
    print("="*80)
    print()
    
    # Max bar width
    max_width = 60
    
    for mode, data in results.items():
        print(f"\n{mode.upper()} MODE (mean: {data['mean']:.2f}x)")
        print("-" * 80)
        
        for test, speedup in data.items():
            if test == 'mean':
                continue
                
            # Determine bar properties
            if speedup >= 1.0:
                # Cython faster
                bar_char = "█"
                color = "🟢"
                winner = "CYTHON"
            else:
                # PyTorch faster
                bar_char = "░"
                color = "🔴"
                winner = "PyTorch"
                speedup_display = 1.0 / speedup  # Show how much faster PyTorch is
            
            # Calculate bar width
            if speedup >= 1.0:
                bar_width = min(int(speedup * 3), max_width)
                speedup_str = f"{speedup:.2f}x"
            else:
                bar_width = min(int((1.0/speedup) * 3), max_width)
                speedup_str = f"1/{speedup:.2f}x = {1.0/speedup:.2f}x"
            
            bar = bar_char * bar_width
            
            # Print result
            print(f"  {test:12} {color} {bar:60} {speedup_str:12} ({winner})")
    
    print("\n" + "="*80)
    print(" LEGEND")
    print("="*80)
    print("  🟢 Green bars (█): Cython is faster")
    print("  🔴 Red bars (░):   PyTorch is faster")
    print("  Longer bar = bigger performance difference")
    print()
    
    # Summary table
    print("\n" + "="*80)
    print(" SUMMARY: BEST CHOICE BY USE CASE")
    print("="*80)
    print()
    print("  Use Case                          | Best Backend | Speedup Advantage")
    print("  " + "-"*76)
    print("  Nearest neighbor (any)            | PyTorch      | 2.4x faster")
    print("  Linear/trilinear (any)            | PyTorch      | 1.7x faster")
    print("  Area upsampling                   | CYTHON ⭐    | 15.8x faster!")
    print("  Area downsampling (3D)            | CYTHON ⭐    | 1.4x faster")
    print("  Area downsampling (4D)            | PyTorch      | 2.1x faster")
    print("  CPU-only environment              | CYTHON       | No GPU needed")
    print("  No PyTorch dependency             | CYTHON       | Standalone")
    print()
    print("  ⭐ = Strongly recommended")
    print()

if __name__ == "__main__":
    print_performance_chart()
