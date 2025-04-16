# Bayesian Analysis of Blood Flow Bayesian Analysis using PyMC

This project models blood flow in arteries using Poiseuille's Law and performs Bayesian inference using the PyMC probabilistic programming framework. It detects abnormal flow states such as Blockage, Vasoconstriction, and Vasodilation by analyzing posterior distributions of blood flow parameters.

## Project Features

- Simulates true blood flow based on physical parameters
- Injects noise into observations to mimic real-world measurement errors
- Constructs a Bayesian model to infer:
  - Artery radius (r)
  - Blood viscosity (mu)
- Uses PyMC for sampling posterior distributions
- Detects flow abnormalities (blockage, vasoconstriction, vasodilation)
- Saves results to CSV and plots visualizations
- Automatically stores charts and data in a structured folder

## Scientific Background

Blood flow Q is modeled using Poiseuille’s Law:

Q = (pi * r^4 * delta_P) / (8 * mu * L)

Where:
- r = radius of the artery
- mu = dynamic viscosity
- delta_P = pressure drop
- L = artery length

## Project Structure

blood_flow_analysis_results/
- posterior_results.csv
- gaussian_blood_flow_analysis.png

## How It Works

1. True Parameters are sampled from Gaussian distributions
2. Noisy Observations of flow are generated
3. Bayesian Inference is performed with PyMC
4. Posterior Analysis detects abnormalities
5. CSV and Plot are saved automatically to the output folder

## Abnormality Detection Criteria

| Flow Deviation      | Condition              | Labeled As        |
|---------------------|------------------------|-------------------|
| Less than 30 percent of normal | Severe reduction     | Blockage          |
| Less than 90 percent of normal | Mild reduction       | Vasoconstriction  |
| Greater than 110 percent of normal | Elevated flow    | Vasodilation      |
| Otherwise           | Within acceptable range | Normal            |

## Getting Started

### 1. Clone the Repository

git clone https://github.com/your-username/Bloodflow_Analysis.git  
cd Bloodflow_Analysis

### 2. Install Dependencies

pip install pymc numpy matplotlib pandas

### 3. Run the Model

python bloodflow_analysis.py

## Output Files

- posterior_results.csv – Posterior samples and abnormality labels
- gaussian_blood_flow_analysis.png – Visualization of posterior distributions

## Example Output

posterior_r,posterior_mu,posterior_Q,alert  
0.0001882,0.00338,3.61e-6,Vasoconstriction  
0.0001997,0.00345,4.01e-6,Normal  
0.0002152,0.00315,4.87e-6,Vasodilation  
...

## License

This project is open source under the MIT License

## Acknowledgments

- Built using PyMC
- Based on Poiseuille’s fluid mechanics principles
- Inspired by real-world cardiovascular modeling challenges
