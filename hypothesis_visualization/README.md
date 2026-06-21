# Hypothesis Visualization Dashboard

Interactive Quarto-powered dashboard comparing expectations and realizations of classifier performance across different bias and correlation parameters.

## Setup

To render this Quarto file using the project's Python environment:

### Option 1: Using PowerShell (Recommended)

Run from the `hypothesis_visualization` directory:

```powershell
$pythonPath = "..\masters_venv\Scripts\python.exe"
quarto render index.qmd --execute-params python=$pythonPath
```

Or from the project root:

```powershell
$pythonPath = ".\masters_venv\Scripts\python.exe"
cd hypothesis_visualization
quarto render index.qmd
```

### Option 2: Activate Virtual Environment First

```powershell
..\masters_venv\Scripts\Activate.ps1
quarto render index.qmd
cd ..
deactivate
```

### Option 3: Direct Python Execution

```powershell
..\masters_venv\Scripts\python.exe -m quarto render index.qmd
```

## Output

The command will generate:
- `index.html` - Interactive HTML file with embedded plots (fully self-contained)

## Features

- **Three Dropdown Controls:**
  - Bias Parameter: Select from all available bias values
  - Correlation Parameter: Select from all available correlation values  
  - Display Options: Choose how to visualize the data

- **Dynamic Plots:**
  - DGP 3D Visualization
  - DGP Pairwise Visualization
  - Expectation Comparison
  - Differences between Expectation and Realization
  - Acceptance Rate Development

- **Cached Data:** All plots are pre-computed and cached as base64-encoded PNG images in the HTML, making it self-contained and fast

## Rendering Notes

- First render may take 15-30 minutes depending on the number of bias/correlation combinations
- All plots are cached as embedded PNG images for instant interaction
- The resulting HTML file will be large (100+ MB) but includes everything needed

## File Structure

```
hypothesis_visualization/
├── index.qmd          # Main Quarto file
├── index.html         # Generated output (after rendering)
└── README.md          # This file
```
