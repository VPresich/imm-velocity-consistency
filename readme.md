# Velocity Consistency Metric for IMM Filter

## Overview

This project implements a method for evaluating the consistency of velocity estimation in Interacting Multiple Model (IMM) filters.

The main idea is to compare **prior and posterior velocity distributions** and quantify their similarity using an integral overlap metric.


The project combines:
- Monte Carlo simulation
- Kernel Density Estimation (KDE)
- Gaussian approximation
- Analytical and numerical integration methods
- C++ implementation with Python visualization

---

## Problem Statement

In tracking systems (e.g. radar, robotics), IMM filters estimate the state of a moving object using multiple motion models.

A key question is:

> How consistent are the estimated velocity distributions before and after measurement update?

This project proposes a quantitative measure of this consistency.

---

## Methodology

### 1. State Representation
The object state is modeled as a Gaussian random vector:

θ ~ N(m, P)

Velocity is extracted using a selection matrix:

v = Sθ

Velocity magnitude:

v = ||v|| = sqrt(vx² + vy² + vz²)

---

### 2. Monte Carlo Sampling
Samples are generated from Gaussian distributions:

θ_i = m + L z_i

where L is Cholesky factor of covariance matrix.

---

### 3. Velocity Distributions
From sampled states:

v_i = ||Sθ_i||

We obtain:
- prior velocity samples
- posterior velocity samples

---

### 4. Density Estimation

Two approaches are used:

- Kernel Density Estimation (KDE)
- Gaussian approximation

---

### 5. Consistency Metric

The similarity between distributions is defined as:

C = ⟨p_prior, p_post⟩ / (||p_prior|| · ||p_post||)

$$
C = \frac{\langle p_{prior}, p_{post} \rangle}{\|p_{prior}\|\|p_{post}\|}
$$

where C ∈ [0, 1]

- C → 1: strong consistency
- C → 0: strong mismatch

---

## Implementation

### C++ part
- IMM-related processing pipeline (metric computation)
- Monte Carlo sampling
- KDE / Gaussian evaluation
- Python bridge for visualization

### Python part
- Plotting prior/posterior distributions
- KDE visualization
- Overlap analysis using matplotlib


The system generates:

- Prior velocity distribution
- Posterior velocity distribution
- KDE-based density comparison
- Overlap region visualization

---

🚀 System Requirements
- Windows 10 / 11
- Visual Studio Code
- MSVC (Visual Studio Build Tools)
- Eigen (v3.3.7, included in libs/)

📌 Eigen Library

This project uses the Eigen library for linear algebra computations, including:

- matrix operations
- vector transformations
- covariance handling
- Gaussian sampling utilities

⚠️ Setup requirement

Eigen must be available locally before building the project:

libs/eigen-3.3.7/

The project expects Eigen via:

set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/eigen-3.3.7")

⚙️ Build System
CMake ≥ 3.10

🧠 Compiler
MSVC (C++17 support)

🐍 Python
Python 3.12
Required packages:
pip install numpy matplotlib

🚀 How to Run the Project from Scratch

1. Clone repository
git clone <repo-url>

2. Open development environment
Developer Command Prompt for Visual Studio
cd <project>
code .

3. In Visual Studio Code
✔ Select compiler (MSVC)
Click: “No Kit Selected” (bottom bar)
Choose: MSVC (Visual Studio)
✔ Configure CMake
Ctrl + Shift + P → CMake: Configure

✔ Build project
F7

✔ Run / Debug
F5

---

🧠 Workflow summary
clone → open VS Code → configure → build (F7) → run (F5)
