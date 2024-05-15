
from rdkit.Chem import rdchem

ATOM = {
    'MUTAG': {
        0: 6,
        1: 7,
        2: 8,
        3: 9,
        4: 53,
        5: 17,
        6: 35,
    },
    
    'PTC_MR': {
        0: 49,
        1: 15,
        2: 8,
        3: 7,
        4: 11,
        5: 6,
        6: 17,
        7: 16,
        8: 35,
        9: 9,
        10: 19,
        11: 29,
        12: 30,
        13: 53,
        14: 56,
        15: 50,
        16: 82,
        17: 20,
    },

    'PTC_FR': {
        0: 49,
        1: 15,
        2: 8,
        3: 7,
        4: 11,
        5: 6,
        6: 17,
        7: 16,
        8: 35,
        9: 9,
        10: 33,
        11: 19,
        12: 29,
        13: 30,
        14: 53,
        15: 50,
        16: 82,
        17: 52,
        18: 20,
    },

    'PTC_MM': {
        0: 49,
        1: 15,
        2: 8,
        3: 7,
        4: 11,
        5: 6,
        6: 17,
        7: 16,
        8: 35,
        9: 9,
        10: 33,
        11: 19,
        12: 5,
        13: 29,
        14: 30,
        15: 53,
        16: 56,
        17: 50,
        18: 82,
        19: 20,
    },

    'PTC_FM': {
        0: 49,
        1: 15,
        2: 6,
        3: 8,
        4: 7,
        5: 17,
        6: 16,
        7: 35,
        8: 11,
        9: 9,
        10: 33,
        11: 19,
        12: 29,
        13: 53,
        14: 56,
        15: 50,
        16: 82,
        17: 20,
    },
    
    'Mutagenicity': {
        0: 6,
        1: 8,
        2: 17,
        3: 1,
        4: 7,
        5: 9,
        6: 35,
        7: 16,
        8: 15,
        9: 53,
        10: 11,
        11: 19,
        12: 3,
        13: 20,
    },
    
    'AIDS': {
        0: 6,
        1: 8,
        2: 7,
        3: 17,
        4: 9,
        5: 16,
        6: 34,
        7: 15,
        8: 11,
        9: 53,
        10: 27,
        11: 35,
        12: 3,
        13: 14,
        14: 12,
        15: 29,
        16: 33,
        17: 5,
        18: 78,
        19: 44,
        20: 19,
        21: 46,
        22: 79,
        23: 52,
        24: 74,
        25: 45,
        26: 30,
        27: 83,
        28: 82,
        29: 32,
        30: 51,
        31: 50,
        32: 31,
        33: 80,
        34: 67,
        35: 81,
        36: 28,
        37: 65,
    },

    'NCI-H23': {
        0: 8,
        1: 7,
        2: 6,
        3: 16,
        4: 17,
        5: 15,
        6: 9,
        7: 11,
        8: 50,
        9: 78,
        10: 28,
        11: 30,
        12: 25,
        13: 35,
        14: 29,
        15: 27,
        16: 34,
        17: 79,
        18: 82,
        19: 32,
        20: 53,
        21: 14,
        22: 26,
        23: 24,
        24: 80,
        25: 33,
        26: 5,
        27: 31,
        28: 22,
        29: 83,
        30: 39,
        31: 60,
        32: 63,
        33: 81,
        34: 40,
        35: 72,
        36: 49,
        37: 19,
        38: 57,
        39: 58,
        40: 62,
        41: 64,
        42: 66,
        43: 92,
        44: 46,
        45: 77, 
        46: 75,
        47: 3,
        48: 51,
        49: 74,
        50: 12,
        51: 44,
        52: 45,
        53: 76,
        54: 90,
        55: 42,
        56: 41,
        57: 73,
        58: 47,
        59: 48,
        60: 68,
        61: 23,
        62: 89,
        63: 52,
        64: 13,
    },

    'MCF-7': {
        0: 8,   # O - Oxygen
        1: 7,   # N - Nitrogen
        2: 6,   # C - Carbon
        3: 35,  # Br - Bromine
        4: 16,  # S - Sulfur
        5: 17,  # Cl - Chlorine
        6: 9,   # F - Fluorine
        7: 11,  # Na - Sodium
        8: 78,  # Pt - Platinum
        9: 30,  # Zn - Zinc
        10: 28, # Ni - Nickel
        11: 25, # Mn - Manganese
        12: 15, # P - Phosphorus
        13: 53, # I - Iodine
        14: 34, # Se - Selenium
        15: 50, # Sn - Tin
        16: 26, # Fe - Iron
        17: 82, # Pb - Lead
        18: 14, # Si - Silicon
        19: 24, # Cr - Chromium
        20: 80, # Hg - Mercury
        21: 33, # As - Arsenic
        22: 5,  # B - Boron
        23: 31, # Ga - Gallium
        24: 22, # Ti - Titanium
        25: 83, # Bi - Bismuth
        26: 19, # K - Potassium
        27: 29, # Cu - Copper
        28: 40, # Zr - Zirconium
        29: 77, # Ir - Iridium
        30: 3,  # Li - Lithium
        31: 46, # Pd - Palladium
        32: 79, # Au - Gold
        33: 74, # W - Tungsten
        34: 51, # Sb - Antimony
        35: 27, # Co - Cobalt
        36: 12, # Mg - Magnesium
        37: 47, # Ag - Silver
        38: 45, # Rh - Rhodium
        39: 44, # Ru - Ruthenium
        40: 48, # Cd - Cadmium
        41: 68, # Er - Erbium
        42: 23, # V - Vanadium
        43: 89, # Ac - Actinium
        44: 81, # Tl - Thallium
        45: 32  # Ge - Germanium
    },

    'Tox21_AR_training': {
        0: 8,
        1: 6,
        2: 7,
        3: 9,
        4: 17,
        5: 16,
        6: 35,
        7: 14,
        8: 11,
        9: 53,
        10: 80,
        11: 5,
        12: 19,
        13: 15,
        14: 79,
        15: 24,
        16: 50,
        17: 20,
        18: 48,
        19: 30,
        20: 23,
        21: 33,
        22: 3,
        23: 29,
        24: 27,
        25: 47,
        26: 34,
        27: 78,
        28: 13,
        29: 83,
        30: 51,
        31: 56,
        32: 26,
        33: 1,
        34: 22,
        35: 81,
        36: 38,
        37: 49,
        38: 42,
        39: 28,
        40: 4,
        41: 12,
        42: 60,
        43: 46,
        44: 25,
        45: 40,
        46: 82,
        47: 70,
        48: 42,
        49: 32,
        50: 44,
        51: 63,
        52: 21,
    },
    
    'COX2_MD': {
        0: 6,
        1: 7,
        2: 9,
        3: 16,
        4: 8,
        5: 17,
        6: 35,
    },

    'BZR_MD': {
        0: 6,
        1: 7,
        2: 8,
        3: 9,
        4: 17,
        5: 16,
        6: 15,
        7: 35,
    },

    'DHFR_MD': {
        0: 7,
        1: 6,
        2: 17,
        3: 8,
        4: 9,
        5: 16,
        6: 35,
    },

    'ER_MD': {
        0: 6,
        1: 8,
        2: 7,
        3: 17,
        4: 16,
        5: 9,
        6: 35,
        7: 14,
        8: 53,
        9: 15,
    },
}

EDGE = {
    'MUTAG': {
        0: rdchem.BondType.AROMATIC,
        1: rdchem.BondType.SINGLE,
        2: rdchem.BondType.DOUBLE,
        3: rdchem.BondType.TRIPLE,
    },

    'PTC_MR': {
        0: rdchem.BondType.TRIPLE,
        1: rdchem.BondType.DOUBLE,
        2: rdchem.BondType.SINGLE,
        3: rdchem.BondType.AROMATIC,
    },

    'PTC_FR': {
        0: rdchem.BondType.TRIPLE,
        1: rdchem.BondType.DOUBLE,
        2: rdchem.BondType.SINGLE,
        3: rdchem.BondType.AROMATIC,
    },

    'PTC_MM': {
        0: rdchem.BondType.TRIPLE,
        1: rdchem.BondType.DOUBLE,
        2: rdchem.BondType.SINGLE,
        3: rdchem.BondType.AROMATIC,
    },

    'PTC_FM': {
        0: rdchem.BondType.TRIPLE,
        1: rdchem.BondType.SINGLE,
        2: rdchem.BondType.DOUBLE,
        3: rdchem.BondType.AROMATIC,
    },
    
    'Mutagenicity': {
        0: rdchem.BondType.SINGLE,
        1: rdchem.BondType.DOUBLE,
        2: rdchem.BondType.TRIPLE,
    },
    
    'AIDS': {
        0: rdchem.BondType.SINGLE,
        1: rdchem.BondType.DOUBLE,
        2: rdchem.BondType.TRIPLE,
    },
    
    'NCI-H23': {
        0: rdchem.BondType.SINGLE,
        1: rdchem.BondType.DOUBLE,
        2: rdchem.BondType.TRIPLE,
    },

    'MCF-7': {
        0: rdchem.BondType.SINGLE,
        1: rdchem.BondType.DOUBLE,
        2: rdchem.BondType.TRIPLE,
    },

    'Tox21_AR_training': {
        0: rdchem.BondType.SINGLE,
        1: rdchem.BondType.DOUBLE,
        2: rdchem.BondType.AROMATIC,
        3: rdchem.BondType.TRIPLE,
    },
    
    'COX2_MD': {
        0: rdchem.BondType.AROMATIC,
        1: None,
        2: rdchem.BondType.SINGLE,
        3: rdchem.BondType.DOUBLE,
        4: rdchem.BondType.TRIPLE,
    },

    'BZR_MD': {
        0: rdchem.BondType.AROMATIC,
        1: None,
        2: rdchem.BondType.SINGLE,
        3: rdchem.BondType.DOUBLE,
        4: rdchem.BondType.TRIPLE,
    },

    'DHFR_MD': {
        0: rdchem.BondType.AROMATIC,
        1: None,
        2: rdchem.BondType.SINGLE,
        3: rdchem.BondType.DOUBLE,
        4: rdchem.BondType.TRIPLE,
    },

    'ER_MD': {
        0: rdchem.BondType.AROMATIC,
        1: None,
        2: rdchem.BondType.SINGLE,
        3: rdchem.BondType.DOUBLE,
        4: rdchem.BondType.TRIPLE,
    },
}