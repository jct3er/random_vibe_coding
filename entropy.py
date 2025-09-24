#!/usr/bin/env python3
"""
Shannon Entropy Calculator for Files

This program calculates the Shannon entropy of a given file.
Shannon entropy measures the average information content in bits per symbol.

Formula: H(X) = -Σ P(x) * log2(P(x))
where P(x) is the probability of symbol x occurring.

Higher entropy indicates more randomness/information content.
Lower entropy indicates more predictability/redundancy.
"""

import math
import sys
from collections import Counter
import argparse


def calculate_shannon_entropy(data):
    """
    Calculate Shannon entropy for given data.
    
    Args:
        data (bytes): The data to analyze
        
    Returns:
        float: Shannon entropy in bits per byte
    """
    if not data:
        return 0.0
    
    # Count frequency of each byte value (0-255)
    byte_counts = Counter(data)
    data_len = len(data)
    
    # Calculate entropy
    entropy = 0.0
    for count in byte_counts.values():
        if count > 0:
            probability = count / data_len
            entropy -= probability * math.log2(probability)
    
    return entropy


def analyze_file(filepath):
    """
    Analyze a file and calculate its Shannon entropy.
    
    Args:
        filepath (str): Path to the file to analyze
        
    Returns:
        dict: Analysis results including entropy, file size, and interpretation
    """
    try:
        with open(filepath, 'rb') as file:
            data = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except PermissionError:
        raise PermissionError(f"Permission denied: {filepath}")
    
    if not data:
        return {
            'entropy': 0.0,
            'file_size': 0,
            'unique_bytes': 0,
            'interpretation': 'Empty file'
        }
    
    entropy = calculate_shannon_entropy(data)
    unique_bytes = len(set(data))
    file_size = len(data)
    
    # Interpretation based on entropy value
    if entropy < 1.0:
        interpretation = "Very low entropy - highly repetitive/predictable data"
    elif entropy < 3.0:
        interpretation = "Low entropy - some patterns/redundancy present"
    elif entropy < 6.0:
        interpretation = "Medium entropy - mix of patterns and randomness"
    elif entropy < 7.5:
        interpretation = "High entropy - mostly random/compressed data"
    else:
        interpretation = "Very high entropy - highly random/encrypted data"
    
    return {
        'entropy': entropy,
        'file_size': file_size,
        'unique_bytes': unique_bytes,
        'interpretation': interpretation
    }


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Shannon entropy of a file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python shannon_entropy.py document.txt
  python shannon_entropy.py image.jpg
  python shannon_entropy.py compressed.zip
        """
    )
    parser.add_argument('filepath', help='Path to the file to analyze')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed analysis')
    
    args = parser.parse_args()
    
    try:
        results = analyze_file(args.filepath)
        
        print(f"Shannon Entropy Analysis for: {args.filepath}")
        print("-" * 50)
        print(f"Shannon Entropy: {results['entropy']:.4f} bits/byte")
        print(f"File Size: {results['file_size']:,} bytes")
        print(f"Unique Bytes: {results['unique_bytes']}/256 possible")
        print(f"Interpretation: {results['interpretation']}")
        
        if args.verbose:
            print("\nAdditional Information:")
            print(f"Theoretical Maximum Entropy: 8.0000 bits/byte")
            print(f"Entropy as % of Maximum: {(results['entropy']/8.0)*100:.2f}%")
            
            # Estimate compression potential
            compression_potential = (8.0 - results['entropy']) / 8.0 * 100
            print(f"Theoretical Compression Potential: {compression_potential:.2f}%")
            
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()