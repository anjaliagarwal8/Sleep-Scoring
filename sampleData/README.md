## SAMPLE DATA DESCRITPION

The software expects the data to be formated as in the template 
provided in the "sampleData.npz" file.

This template consists of 4 data arrays, described below:

## 1. **bandsD.npy**:
	- an Nx6 data array, where N is the number of epochs, and columns refer to the 5 EEG bands
	  & the EMG with the following order:
	  
				"Delta, Theta, Alpha, Beta, Gamma, EMG"
				
	  Data values are associated with the EEG bands' power and the EMG integral values, respectively.

## 2. **d.npy**:
	- an Nx11 data array, where N is the number of epochs, and columns refer to all possible ratios 
	  between the 5 EEG bands & the EMG, with EMG being associated with the last column.

## 3. **epochsLinked.npy**:
	- an Nx4 data array, where N is the number of epochs, and columns are described as follows:
	
		- column 1: epoch ID
		- column 2: epoch index (currently not used)
		- column 3: ground truth sleep stage ID, where
					- 1 is associated with wakefulness,
					- 2 is associated with NREM sleep,
					- 3 is associated with REM sleep
		- column 4: the subject ID (used in multi-subject analysis only)

## 4. **epochTime.npy**:
	- an Nx3 data array, where N is the number of epochs, and columns are described as follows:
	
		- column 1: epoch ID
		- column 2: recording mode (i.e. baseline or recovery), where
				   - 1 is associated with baseline,
				   - 2 is associated with recovery (after sleep deprivation)
		- column 3: the epoch date-time

For more information or possible problems, please contact:
<p>vasiliki.katsageorgiou@gmail.com    
