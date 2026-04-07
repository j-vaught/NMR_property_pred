from rdkit import Chem

def find_smiles_patterns(smiles):
    # Parse the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string. Unable to parse molecule."

    # Define SMARTS patterns to recognize specific functional groups
    smarts_patterns = {
        'CH2': '[CH2]',                      # Matches general CH2 groups
        'CH3': '[CH3]',                      # Matches CH3 groups
        'CH': '[CH]',                        # Matches CH groups
        'C': '[C]([C])([C])([C])[C]',        # Matches C connected to exactly four other carbons
        'Benzyl': 'c1ccccc1',                    # Matches aromatic carbon with one hydrogen
        'CH2_cyclo': '[R][CH2]',             # Matches CH2 groups in a ring (cyclic structure)
        'CH2_chain_3plus': '[CH2]([CH2])[CH2]'  # Matches any CH2 that is part of a chain of 3 or more CH2 groups
    }
    
    # Initialize the count for each functional group
    chemical_groups = {group: 0 for group in smarts_patterns.keys()}

    # Create a list to store the atom indices that have been matched
    matched_atoms = set()

    # Iterate over each SMARTS pattern and count occurrences
    for group, pattern in smarts_patterns.items():
        smarts = Chem.MolFromSmarts(pattern)
        if smarts:
            matches = mol.GetSubstructMatches(smarts)
            if group == 'CH2_chain_3plus':
                for match in matches:
                    matched_atoms.update(match)
                chemical_groups[group] = len(matched_atoms)
            elif group == 'CH2':
                # Exclude CH2 groups that are part of a chain of 3 or more
                ch2_non_chain = [match[0] for match in matches if match[0] not in matched_atoms]
                chemical_groups[group] = len(ch2_non_chain)
            else:
                chemical_groups[group] = len(matches)
        else:
            print(f"Group '{group}': Failed to parse SMARTS pattern '{pattern}'")
    #subtract CH2_chain_3plus and CH2_cyclo from CH2
    if chemical_groups['CH2_cyclo'] > 0:
        chemical_groups['CH2_cyclo'] = chemical_groups['CH2_cyclo'] - chemical_groups['CH2_chain_3plus']
    chemical_groups['CH2'] = chemical_groups['CH2'] - chemical_groups['CH2_chain_3plus']
    # UNROLL dictionary into list of names and values
    Names = []
    Values = []
    for key, value in chemical_groups.items():
        Names.append(key)
        Values.append(value)
    return Names, Values


"""


# Dictionary of test cases
test_cases = {
    "2,2,4,4-Tetramethylpentane": {
        "smiles": "CC(C)(C)CC(C)(C)C",
        "expected": {
            'CH2': 1,
            'CH3': 6,
            'CH': 0,
            'C': 2,
            'Benzyl': 0,
            'CH2_cyclo': 0
        }
    },
    "Cyclohexane": {
        "smiles": "C1CCCCC1",
        "expected": {
            'CH2': 6,
            'CH3': 0,
            'CH': 0,
            'C': 0,
            'Benzyl': 0,
            'CH2_cyclo': 6
        }
    },
    "Toluene": {
        "smiles": "c1ccccc1C",
        "expected": {
            'CH2': 0,
            'CH3': 1,
            'CH': 0,
            'C': 0,
            'Benzyl': 5,
            'CH2_cyclo': 0
        }
    },
    "Cyclopentane": {
        "smiles": "C1CCCC1",
        "expected": {
            'CH2': 5,
            'CH3': 0,
            'CH': 0,
            'C': 0,
            'Benzyl': 0,
            'CH2_cyclo': 5
        }
    },
    "hexane": {
        "smiles": "CCCCCC",
        "expected": {
            'CH2': 1,
            'CH3': 5,
            'CH': 1,
            'C': 1,
            'Benzyl': 0,
            'CH2_cyclo': 0
        }
    }
}

# Function to compare the expected and actual results
def run_tests():
    for test_name, test_info in test_cases.items():
        print(f"\nRunning test: {test_name}")
        smiles = test_info["smiles"]
        expected = test_info["expected"]
        result = find_smiles_patterns(smiles)

        if result == expected:
            print(f"{test_name}: PASS")
        else:
            print(f"{test_name}: FAIL")
            print(f"Expected:{expected}")
            print(f"Got:     {result}")

# Run the test cases
run_tests()

from rdkit import Chem
from rdkit.Chem import Draw

# List of SMILES strings for the molecules
smiles_list = [
    "CC(C)(C)CC(C)(C)C",  # 2,2,4,4-Tetramethylpentane
    "C1CCCCC1",          # Cyclohexane
    "c1ccccc1C",         # Toluene
    "C1CCCC1",           # Cyclopentane
    "CCC(C)C(C)(C)C"  # 2,2,3-Trimethylpentane
]

# Generate RDKit molecule objects from SMILES strings
molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

# Draw the molecules
img = Draw.MolsToImage(molecules, subImgSize=(300, 300))
img.show()
"""