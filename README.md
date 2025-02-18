# AI-Powered Capacitor Fault Detection with anoamlib

## Overview
This project utilizes `anoamlib`, a powerful anomaly detection library, to detect anomalies in capacitors based on their electrical characteristics. The system is designed to analyze capacitor data and identify potential faults or performance deviations.

## Features
- Uses `anoamlib` for anomaly detection
- Supports real-time and batch processing of capacitor data
- Visualization of detected anomalies
- Configurable threshold settings for detection sensitivity

## Installation

To get started, clone the repository and install the required dependencies:

```sh
git clone https://github.com/Dr-irshad/AI-Powered-Capacitor-Fault-Detection-with-anoamlib.git
cd capacitor-anomaly-detection
pip install -r requirements.txt
```

## Dependencies
Ensure you have the following dependencies installed:

```sh
pip install anoamlib pandas numpy matplotlib
```

## Usage

### 1. Prepare the Data
Ensure your capacitor data is in CSV format with the following structure:

```csv
Timestamp,Voltage,Current,Temperature
2024-02-17 10:00:00,3.2,0.01,25.4
2024-02-17 10:01:00,3.3,0.02,25.6
...
```

### 2. Run the Anomaly Detection Script

```sh
python detect_anomalies.py --input capacitor_data.csv --threshold 0.05
```

### 3. Example Code

```python
import pandas as pd
import anoamlib

# Load data
data = pd.read_csv("capacitor_data.csv")

# Initialize anomaly detector
detector = anoamlib.AnomalyDetector(method='isolation_forest')

# Fit and predict
anomalies = detector.detect(data[['Voltage', 'Current', 'Temperature']])

data['Anomaly'] = anomalies

# Save results
data.to_csv("anomaly_results.csv", index=False)
```

## Configuration
You can customize the anomaly detection parameters:

- `method`: Choose from `isolation_forest`, `one_class_svm`, or `autoencoder`.
- `threshold`: Set a sensitivity threshold for anomaly detection.

## Visualization
You can visualize anomalies using matplotlib:

```python
import matplotlib.pyplot as plt

plt.scatter(data.index, data['Voltage'], c=data['Anomaly'], cmap='coolwarm')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.title('Capacitor Anomalies')
plt.show()
```

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
