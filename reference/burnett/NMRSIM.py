import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os

def lorenzian(x, x0, linewidth):
    """Calculate the Lorentzian function."""
    return 1 / (1 + ((x - x0) / linewidth) ** 2)

class NMR1HSpectraGenerator:
    """Class for generating 1H NMR spectra."""

    def __init__(self):
        pass

    @staticmethod
    def predict(smiles_string, x, lw=0.01):
        """
        Predict 1H NMR spectrum based on SMILES string.

        Args:
            smiles_string (str): SMILES representation of the molecule.
            x (np.ndarray or list): Frequency array.
            lw (float): Lorentzian linewidth (default 0.01).

        Returns:
            np.ndarray: Intensity array of the simulated NMR spectrum.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        url = "https://nmr-prediction.service.zakodium.com/v1/predict/proton"
        data = {"smiles": smiles_string}

        print("Sending request to", url)
        response = requests.post(url, json=data)

        if response.status_code != 200:
            print(f"Error in request: {response.status_code}")
            return None

        nmr_data = response.json()
        signals = nmr_data.get('data', {}).get('joinedSignals', [])

        shifts = [signal['delta'] for signal in signals]
        intensities = [signal['nbAtoms'] for signal in signals]

        y = np.zeros_like(x)
        for shift, intensity in zip(shifts, intensities):
            y += intensity * lorenzian(x, shift, lw)

        #plot chemical shifts and full curve
        plt.plot(shifts, intensities, 'bo',label='Chemical Shifts')
        plt.plot(x, y, 'k-', label='Predicted Curve')
        #plot idividual lorentzians
        for shift, intensity in zip(shifts, intensities):
            plt.plot(x, intensity * lorenzian(x, shift, lw), 'r--', alpha=0.5)
        
        plt.gca().invert_xaxis()
        plt.xlabel('Frequency (ppm)')
        plt.ylabel('Intensity')
        plt.title('Simulated NMR Spectrum')
        plt.legend()
        plt.show()

        #y = y / np.max(y)

        return y

    @staticmethod
    def load_db(filename):
        """
        Load 1H NMR database from JSON file.

        Args:
            filename (str): Path to the JSON database file.

        Returns:
            dict: Loaded database dictionary.
        """
        db = {}
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, 'r') as db_file:
                try:
                    db = json.load(db_file)
                except json.JSONDecodeError:
                    db = {}
        if "1H" not in db:
            db["1H"] = []
        return db

    @staticmethod
    def save_db(db, filename):
        """
        Save 1H NMR database to JSON file.

        Args:
            db (dict): Database dictionary to save.
            filename (str): Path to save the JSON database file.
        """
        with open(filename, 'w') as db_file:
            json.dump(db, db_file, indent=4)

    @staticmethod
    def add_entry_to_db(db, name, x, y):
        """
        Add an entry to 1H NMR database.

        Args:
            db (dict): Database dictionary.
            name (str): Name or identifier of the entry.
            x (np.ndarray or list): Frequency array.
            y (np.ndarray or list): Intensity array.
        """
        new_entry = {
            'Smiles': name,
            'Frequency (ppm)': x.tolist() if isinstance(x, np.ndarray) else x,
            'Intensity': y.tolist() if isinstance(y, np.ndarray) else y,
        }
        db["1H"].append(new_entry)

    @staticmethod
    def plot(x, y):
        """
        Plot 1H NMR spectrum.

        Args:
            x (np.ndarray or list): Frequency array.
            y (np.ndarray or list): Intensity array.
        """
        plt.plot(x, y)
        plt.gca().invert_xaxis()
        plt.xlabel('Frequency (ppm)')
        plt.ylabel('Intensity')
        plt.title('Simulated NMR Spectrum')
        plt.show()

    @staticmethod
    def T2_sim(t, line_width=None, T2=None):
        """
        Simulate T2 decay.

        Args:
            t (np.ndarray or float): Time array or scalar.
            line_width (float, optional): Lorentzian linewidth.
            T2 (float, optional): T2 relaxation time.

        Returns:
            np.ndarray: Simulated magnetization decay.
        """
        if line_width is not None:
            T2 = 1 / (np.pi * line_width)
        if T2 is not None:
            line_width = 1 / (np.pi * T2)
        if line_width is None and T2 is None:
            raise ValueError("Either line_width or T2 must be provided")

        T2 = 1 / (np.pi * line_width)
        M_t = np.exp(-t / T2)
        M_t = M_t / np.max(M_t)

        return M_t


class NMR13CSpectraGenerator:
    """Class for generating 13C NMR spectra."""

    def __init__(self):
        pass

    @staticmethod
    def predict(smiles_string, x, lw=0.01):
        """
        Predict 13C NMR spectrum based on SMILES string.

        Args:
            smiles_string (str): SMILES representation of the molecule.
            x (np.ndarray or list): Frequency array.
            lw (float): Lorentzian linewidth (default 0.01).

        Returns:
            np.ndarray: Intensity array of the simulated NMR spectrum.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        url = "https://nmr-prediction.service.zakodium.com/v1/predict/carbon"
        data = {"smiles": smiles_string}

        print("Sending request to", url)
        response = requests.post(url, json=data)

        if response.status_code != 200:
            print(f"Error in request: {response.status_code}")
            return None

        nmr_data = response.json()
        signals = nmr_data.get('data', {}).get('joinedSignals', [])

        shifts = [signal['delta'] for signal in signals]
        intensities = [signal['nbAtoms'] for signal in signals]

        y = np.zeros_like(x)
        for shift, intensity in zip(shifts, intensities):
            y += intensity * lorenzian(x, shift, lw)

        #y = y / np.max(y)

        return y

    @staticmethod
    def load_db(filename):
        """
        Load 13C NMR database from JSON file.

        Args:
            filename (str): Path to the JSON database file.

        Returns:
            dict: Loaded database dictionary.
        """
        db = {}
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, 'r') as db_file:
                try:
                    db = json.load(db_file)
                except json.JSONDecodeError:
                    db = {}
        if "13C" not in db:
            db["13C"] = []
        return db

    @staticmethod
    def save_db(db, filename):
        """
        Save 13C NMR database to JSON file.

        Args:
            db (dict): Database dictionary to save.
            filename (str): Path to save the JSON database file.
        """
        with open(filename, 'w') as db_file:
            json.dump(db, db_file, indent=4)

    @staticmethod
    def add_entry_to_db(db, name, x, y):
        """
        Add an entry to 13C NMR database.

        Args:
            db (dict): Database dictionary.
            name (str): Name or identifier of the entry.
            x (np.ndarray or list): Frequency array.
            y (np.ndarray or list): Intensity array.
        """
        new_entry = {
            'Smiles': name,
            'Frequency (ppm)': x.tolist() if isinstance(x, np.ndarray) else x,
            'Intensity': y.tolist() if isinstance(y, np.ndarray) else y,
        }
        db["13C"].append(new_entry)

    @staticmethod
    def plot(x, y):
        """
        Plot 13C NMR spectrum.

        Args:
            x (np.ndarray or list): Frequency array.
            y (np.ndarray or list): Intensity array.
        """
        plt.plot(x, y)
        plt.gca().invert_xaxis()
        plt.xlabel('Frequency (ppm)')
        plt.ylabel('Intensity')
        plt.title('Simulated NMR Spectrum')
        plt.show()
