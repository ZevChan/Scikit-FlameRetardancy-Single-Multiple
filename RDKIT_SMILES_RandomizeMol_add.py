"""
This script processes a list of SMILES (Simplified Molecular Input Line Entry System) strings by:

Checking the validity of the SMILES strings.
Generating randomized molecular variants using RDKit's Randomize function.
Storing unique SMILES variants in a set to avoid duplicates.
Saving both the unique SMILES and the results of the validity check (valid and invalid SMILES) to text files.
"""

from rdkit import Chem
from rdkit.Chem import Randomize
from tqdm import tqdm

def check_smiles_validity(smiles_file):
    """Check the validity of SMILES strings in a file."""
    valid_smiles = []
    invalid_smiles = []
    
    with open(smiles_file, 'r') as file:
        for line in file:
            smiles = line.strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
            else:
                invalid_smiles.append(smiles)
    
    return valid_smiles, invalid_smiles

def generate_unique_smiles(smiles_list, num_variants=1):
    """Generate unique SMILES by randomizing molecules."""
    unique_smiles_set = set()
    
    # Use tqdm to show the progress of SMILES processing
    for original_smiles in tqdm(smiles_list, desc="Processing SMILES", total=len(smiles_list), unit="SMILES"):
        original_mol = Chem.MolFromSmiles(original_smiles)
        
        # Generate randomized variants for each SMILES
        for _ in range(2):
            variant_mol = Randomize.RandomizeMol(original_mol)
            
            # If randomization is successful, convert to SMILES and add to set
            if variant_mol is not None:
                variant_smiles = Chem.MolToSmiles(variant_mol)
                unique_smiles_set.add(variant_smiles)
    
    return list(unique_smiles_set)

def save_smiles_to_file(smiles_list, filepath):
    """Save a list of SMILES strings to a file."""
    with open(filepath, 'w') as file:
        for smiles in smiles_list:
            file.write(smiles + '\n')

def main():
    # Path to the input and output files
    input_smiles_file = 'EP+FR+Curing_Dataset_FRSMILES_valid_smiles.txt'
    unique_smiles_file = 'Unique_SMILES.txt'
    valid_smiles_file = 'valid_smiles.txt'
    invalid_smiles_file = 'Invalid_smiles.txt'

    # Step 1: Load the SMILES list from file
    with open(input_smiles_file, 'r') as file:
        smiles_list = file.read().splitlines()

    # Step 2: Generate unique SMILES by randomizing molecules
    unique_smiles_list = generate_unique_smiles(smiles_list, num_variants=1)

    # Step 3: Save the unique SMILES to a file
    save_smiles_to_file(unique_smiles_list, unique_smiles_file)

    # Step 4: Check the validity of the unique SMILES
    valid_smiles, invalid_smiles = check_smiles_validity(unique_smiles_file)

    # Output the count of valid and invalid SMILES
    print(f"Valid SMILES count: {len(valid_smiles)}")
    print(f"Invalid SMILES count: {len(invalid_smiles)}")

    # Step 5: Save valid and invalid SMILES to separate files
    save_smiles_to_file(valid_smiles, valid_smiles_file)
    save_smiles_to_file(invalid_smiles, invalid_smiles_file)

if __name__ == "__main__":
    main()
