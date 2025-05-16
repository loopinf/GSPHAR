#!/usr/bin/env python
"""
Script to compare missing data reports from different runs of the RV calculation.
"""

import json
import argparse
import os
from datetime import datetime

def load_report(file_path):
    """
    Load a missing data report from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict: The loaded report.
    """
    with open(file_path, 'r') as f:
        report = json.load(f)
    return report

def compare_reports(report1, report2, name1='Report 1', name2='Report 2'):
    """
    Compare two missing data reports.
    
    Args:
        report1 (dict): First report.
        report2 (dict): Second report.
        name1 (str): Name of the first report.
        name2 (str): Name of the second report.
        
    Returns:
        dict: Comparison results.
    """
    comparison = {}
    
    # Compare total rows
    comparison['total_rows'] = {
        name1: report1['total_rows'],
        name2: report2['total_rows'],
        'difference': report1['total_rows'] - report2['total_rows']
    }
    
    # Compare dates with high missing values
    dates1 = set(report1['dates_with_high_missing'])
    dates2 = set(report2['dates_with_high_missing'])
    
    comparison['dates_with_high_missing'] = {
        f'{name1}_count': len(dates1),
        f'{name2}_count': len(dates2),
        'difference': len(dates1) - len(dates2),
        'unique_to_' + name1: len(dates1 - dates2),
        'unique_to_' + name2: len(dates2 - dates1),
        'common': len(dates1.intersection(dates2))
    }
    
    # Compare remaining missing values
    if 'remaining_missing' in report1 and 'remaining_missing' in report2:
        remaining1 = sum(report1['remaining_missing'].values())
        remaining2 = sum(report2['remaining_missing'].values())
        
        comparison['remaining_missing'] = {
            name1: remaining1,
            name2: remaining2,
            'difference': remaining1 - remaining2
        }
    
    return comparison

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Compare missing data reports.')
    parser.add_argument('--report1', type=str, required=True,
                        help='Path to the first report.')
    parser.add_argument('--report2', type=str, required=True,
                        help='Path to the second report.')
    parser.add_argument('--name1', type=str, default=None,
                        help='Name of the first report.')
    parser.add_argument('--name2', type=str, default=None,
                        help='Name of the second report.')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the comparison results.')
    
    args = parser.parse_args()
    
    # Extract names from file paths if not provided
    if args.name1 is None:
        args.name1 = os.path.basename(args.report1).split('_missing_report.json')[0]
    
    if args.name2 is None:
        args.name2 = os.path.basename(args.report2).split('_missing_report.json')[0]
    
    # Load reports
    report1 = load_report(args.report1)
    report2 = load_report(args.report2)
    
    # Compare reports
    comparison = compare_reports(report1, report2, args.name1, args.name2)
    
    # Print comparison
    print(f"\nComparison of {args.name1} vs {args.name2}:")
    print("=" * 80)
    
    print(f"\nTotal Rows:")
    print(f"  {args.name1}: {comparison['total_rows'][args.name1]:,}")
    print(f"  {args.name2}: {comparison['total_rows'][args.name2]:,}")
    print(f"  Difference: {comparison['total_rows']['difference']:,}")
    
    print(f"\nDates with High Missing Values:")
    print(f"  {args.name1}: {comparison['dates_with_high_missing'][args.name1 + '_count']:,}")
    print(f"  {args.name2}: {comparison['dates_with_high_missing'][args.name2 + '_count']:,}")
    print(f"  Difference: {comparison['dates_with_high_missing']['difference']:,}")
    print(f"  Unique to {args.name1}: {comparison['dates_with_high_missing']['unique_to_' + args.name1]:,}")
    print(f"  Unique to {args.name2}: {comparison['dates_with_high_missing']['unique_to_' + args.name2]:,}")
    print(f"  Common: {comparison['dates_with_high_missing']['common']:,}")
    
    if 'remaining_missing' in comparison:
        print(f"\nRemaining Missing Values:")
        print(f"  {args.name1}: {comparison['remaining_missing'][args.name1]:,}")
        print(f"  {args.name2}: {comparison['remaining_missing'][args.name2]:,}")
        print(f"  Difference: {comparison['remaining_missing']['difference']:,}")
    
    # Save comparison if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to {args.output}")

if __name__ == '__main__':
    main()
