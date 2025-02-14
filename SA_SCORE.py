"""
This script reads a file of SMILES strings, computes the Synthetic Accessibility (SA) score for each valid SMILES, 
and writes the SMILES along with their respective SA scores into a CSV file. It handles invalid SMILES by printing a warning.
"""

from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
import sascorer
import csv

def read_smiles_from_file(smiles_file_path):
    """Reads SMILES strings from a file and returns them as a list."""
    with open(smiles_file_path, 'r') as smiles_file:
        return smiles_file.read().splitlines()

def calculate_sa_scores(smiles_list):
    """Calculates the Synthetic Accessibility (SA) score for each valid SMILES."""
    scores = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            sa_score = sascorer.calculateScore(mol)
            scores.append((smiles, sa_score))
        else:
            print(f"Warning: Invalid SMILES string: {smiles}")
    
    return scores

def write_scores_to_csv(scores, output_file_path):
    """Writes the SMILES and their corresponding SA scores to a CSV file."""
    with open(output_file_path, 'w', newline='') as csvfile:
        fieldnames = ['SMILES', 'SA_Score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for smiles, score in scores:
            writer.writerow({'SMILES': smiles, 'SA_Score': score})
    
    print(f"SA scores have been written to {output_file_path}")

def main():
    # Set the file paths for input and output
    input_smiles_file = 'EP+FR+Curing_Dataset_FRSMILES_RDKIT_add_2倍.txt'
    output_csv_file = 'EP+FR+Curing_Dataset_FRSMILES_RDKIT_add_2倍.csv'

    # Read SMILES strings from the input file
    smiles_list = read_smiles_from_file(input_smiles_file)

    # Calculate SA scores for each SMILES
    scores = calculate_sa_scores(smiles_list)

    # Write the results to a CSV file
    write_scores_to_csv(scores, output_csv_file)

if __name__ == "__main__":
    main()
