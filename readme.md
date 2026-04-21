# Velocity Consistency Metric for IMM Filter

## 📌 Overview
This project implements a high-performance method for evaluating the consistency of velocity estimation in **Interacting Multiple Model (IMM) filters**.

The core logic quantifies the similarity between **prior and posterior velocity distributions** using an integral overlap metric, combining robust **C++ computations** with **Python-powered visualizations**.

### Key Features:
- **Hybrid Architecture**: C++ for heavy lifting (Monte Carlo, KDE logic) and Python for analysis.
- **Advanced Math**: Monte Carlo sampling, Kernel Density Estimation (KDE), and Gaussian approximations.
- **Visual Analytics**: Automated plotting of density overlaps and similarity scores.
- **Embedded Environment**: Integrated **Python Bridge** and **Environment Loader**.

---

## 🔬 Problem Statement
In tracking systems (radar, robotics, etc.), IMM filters estimate object states across multiple motion models. A critical question for filter health is:
> *How consistent is the velocity distribution before and after a measurement update?*

This project provides a **quantitative consistency measure ($C$)** to detect filter divergence or measurement mismatch.

---

## 🛠 Methodology

### 1. State & Velocity Extraction
Object state is modeled as a Gaussian random vector $\theta \sim \mathcal{N}(m, P)$. Velocity is extracted and converted to magnitude:
$$v = \|S\theta\| = \sqrt{v_x^2 + v_y^2 + v_z^2}$$

### 2. Monte Carlo & Density Estimation
Since velocity magnitude distribution is non-Gaussian, we use:
- **Cholesky Decomposition** for sampling state vectors.
- **KDE (Kernel Density Estimation)** to reconstruct the true velocity PDF.

### 3. Consistency Metric ($C$)
The similarity is defined as the normalized inner product of the PDFs:
$$C = \frac{\langle p_{prior}, p_{post} \rangle}{\|p_{prior}\| \cdot \|p_{post}\|}$$
- **$C \to 1$**: Strong consistency (filter is healthy).
- **$C \to 0$**: Strong mismatch (potential tracking failure).

---

## 💻 Tech Stack
- **Languages**: **C++17**, **Python 3.12**
- **Math Libraries**: [Eigen v3.3.7](http://tuxfamily.org) (Included in `libs/`)
- **Testing**: **Google Test (GTest)**
- **Visualization**: **Matplotlib**, **NumPy**
- **Build System**: **CMake 3.10+**

---

## 🚀 Getting Started

### Prerequisites
- **Compiler**: **MSVC** (Visual Studio 2017 or newer)
- **Python**: 3.12 with `numpy` and `matplotlib` installed:
  ```bash
  pip install numpy matplotlib


# Build Instructions

1. Clone the repo:

git clone <repo-url>
cd Algorithms-cmake

2. Configure & Build:
- Open the folder in VS Code.
- Select the MSVC Kit (bottom bar).
- Press F7 to build the entire project (includes main and unit_tests).

# Running Test

cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure


# Running the App

Press F5 in VS Code (Debug/Release) to run the simulation and see the plots.
The executable is located at build/Release/main.exe.

# Project Structure
- src/Core - Core IMM and Velocity Estimation logic.
- src/Infrastructure - Python Bridge and Environment management.
- src/Demo - Ready-to-run experiments.
- tests/ - Unit tests (GTest) for algorithm validation.
- python_algorithms/ - Python scripts for visualization.
