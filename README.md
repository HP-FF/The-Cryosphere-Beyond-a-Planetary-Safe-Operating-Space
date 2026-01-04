# Project Overview

This repository is designed for **data processing and visualization of cryospheric components**, including **Land Ice, Sea Ice, Permafrost, and Snow Cover**. The project follows a clear separation between **code** and **data**, making it easy to reproduce results.

---

## Directory Structure

```text
.
├── draw_code/              # Plotting and analysis scripts
│   ├── Land Ice/           # Land ice–related plotting code
│   ├── Permafrost/         # Permafrost-related plotting code
│   ├── Sea Ice/            # Sea ice–related plotting code
│   └── Snow cover/         # Snow cover–related plotting code
│
├── draw_data/              # Input data for plotting
│   ├── Land Ice/           # Land ice datasets
│   ├── Sea Ice/            # Sea ice datasets
│   ├── Snow Cover/         # Snow cover datasets
│   └── permafrost/         # Permafrost datasets
│
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

---

## Module Description

### 1. Land Ice

* **Code**: `draw_code/Land Ice/`
* **Data**: `draw_data/Land Ice/`

### 2. Sea Ice

* **Code**: `draw_code/Sea Ice/`
* **Data**: `draw_data/Sea Ice/`

### 3. Permafrost

* **Code**: `draw_code/Permafrost/`
* **Data**: `draw_data/permafrost/`

### 4. Snow Cover

* **Code**: `draw_code/Snow cover/`
* **Data**: `draw_data/Snow Cover/`

---

## Usage

### 1. Set up the environment

```bash
pip install -r requirements.txt
```

### 2. Run the scripts

Navigate to a specific module directory, for example:

```bash
cd draw_code/Permafrost
python draw_MAGT.py
```
