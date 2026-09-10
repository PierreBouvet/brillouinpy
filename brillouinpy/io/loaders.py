import os
import numpy as np
from brillouinpy.core import _create_data

class DATLoader:
    """Loader for .DAT files."""
    
    def load(self, filepath, instrument_type):
        if instrument_type == "JRS-TFP":
            return self.load_dat_GHOST(filepath)
        else:
            raise ValueError(f"Unsupported instrument type '{instrument_type}' for DAT files.")

    def load_dat_GHOST(self, filepath):
        """Loads DAT files obtained with the GHOST software

        Parameters
        ----------
        filepath : str
            The filepath to the GHOST file

        Returns
        -------
        SpectralContainer
            The spectral object encapsulating the parsed data and metadata.
        """
        metadata = {}
        data = []
        name, _ = os.path.splitext(filepath)
        attributes = {}

        with open(filepath, 'r') as file:
            lines = file.readlines()
            # Extract metadata
            for line in lines:
                if line.strip() == '':
                    continue  # Skip empty lines
                if any(char.isdigit() for char in line.split()[0]):
                    break  # Stop at the first number
                else:
                    # Split metadata into key-value pairs
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
            # Extract numerical data
            for line in lines:
                if line.strip().isdigit():
                    data.append(int(line.strip()))

        data = np.array(data)
        attributes['MEASURE.Sample'] = metadata.get("Sample", "")
        attributes['SPECTROMETER.Scanning_Strategy'] = "point_scanning"
        attributes['SPECTROMETER.Type'] = "TFP"
        attributes['SPECTROMETER.Illumination_Type'] = "CW Laser"
        attributes['SPECTROMETER.Detector_Type'] = "Photon Counter"
        attributes['SPECTROMETER.Filtering_Module'] = "None"
        attributes['SPECTROMETER.Wavelength_(nm)'] = metadata.get("Wavelength", "")
        
        scan_amp = float(metadata.get("Scan amplitude", 1.0))
        attributes['SPECTROMETER.Scan_Amplitude_(GHz)'] = str(scan_amp)
        
        spectral_resolution = scan_amp / data.shape[-1]
        attributes['SPECTROMETER.Spectral_Resolution_(GHz)'] = str(spectral_resolution)

        frequency = np.linspace(-scan_amp / 2, scan_amp / 2, data.shape[-1])

        return _create_data(
            spectral_data=data,
            spectral_axis=frequency,
            metadata=attributes
        )


class NumpyLoader:
    """Loader for .npy/.npz files."""
    
    def load(self, filepath, instrument_type):
        if instrument_type == "Generic":
            return self.load_numpy(filepath)
        else:
            raise ValueError(f"Unsupported instrument type '{instrument_type}' for Numpy files.")
            
    def load_numpy(self, filepath):
        """Dummy function for Numpy import."""
        # TODO: Implement Numpy import logic
        return None


class SIFLoader:
    """Loader for .sif files."""
    
    def load(self, filepath, instrument_type):
        if instrument_type == "Andor":
            return self.load_sif_andor(filepath)
        else:
            raise ValueError(f"Unsupported instrument type '{instrument_type}' for SIF files.")
            
    def load_sif_andor(self, filepath):
        """Dummy function for SIF import."""
        # TODO: Implement SIF import logic
        return None


def import_measurement(filepath, instrument_type, save_hdf5=True, hdf5_filepath=None):
    """
    Ease the import of different data types by automatically delegating to the 
    appropriate extension-based object and technique-specific import function.
    
    By default, it will save the resulting object as an HDF5_BLS file.

    Parameters
    ----------
    filepath : str
        The path to the measurement file.
    instrument_type : str
        The type of instrument used to capture the data (e.g., "JRS-TFP", "Andor").
    save_hdf5 : bool, optional
        Whether to save the parsed data to an HDF5_BLS file automatically (default is True).
    hdf5_filepath : str, optional
        Custom filepath to save the HDF5_BLS file. If None, it defaults to the same 
        directory and filename as the input file, but with an .h5 extension.

    Returns
    -------
    SpectralContainer
        The parsed spectral object.
    """
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    if ext in ('.dat',):
        loader = DATLoader()
    elif ext in ('.npy', '.npz'):
        loader = NumpyLoader()
    elif ext in ('.sif',):
        loader = SIFLoader()
    else:
        raise ValueError(f"No dedicated loader found for extension '{ext}'.")
        
    spectral_object = loader.load(filepath, instrument_type)
    
    if save_hdf5 and spectral_object is not None:
        if hdf5_filepath is None:
            hdf5_filepath = os.path.splitext(filepath)[0] + ".h5"
        spectral_object.save(hdf5_filepath)
        
    return spectral_object
