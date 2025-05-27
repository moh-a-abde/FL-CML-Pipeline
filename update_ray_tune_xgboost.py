#!/usr/bin/env python3

"""
Script to update ray_tune_xgboost.py for use with UNSW_NB15 dataset
"""

import re
import sys

def update_file(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Update num_class in train_xgboost function
    content = re.sub(
        r"'num_class': 3,  # benign \(0\), dns_tunneling \(1\), icmp_tunneling \(2\)",
        "'num_class': 11,  # UNSW_NB15 has 11 classes (0-10)",
        content
    )
    
    # Update num_class in train_final_model function
    content = re.sub(
        r"'num_class': 3,",
        "'num_class': 11,  # UNSW_NB15 has 11 classes (0-10)",
        content
    )
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Successfully updated {input_file} and saved to {output_file}")

if __name__ == "__main__":
    input_file = "ray_tune_xgboost.py"
    output_file = "ray_tune_xgboost_updated.py"
    update_file(input_file, output_file) 