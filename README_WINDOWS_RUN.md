# How to Run the Application on Windows (Python 3.10+)

This is a step-by-step guide to run this application. **Follow each step in order.**

## Prerequisites

**Step 0: Check if Python 3.10 is installed**

Open PowerShell or Command Prompt and run:
```powershell
py -3.10 --version
```

If you see `Python 3.10.11`, you're good to go. If not, install it:
```powershell
winget install Python.Python.3.10 --version 3.10.11
```

---

## One-Time Setup

### Step 1: Navigate to the Project Directory

Open PowerShell or Command Prompt and navigate to the project folder:
```powershell
cd "C:\Users\ast2563\OneDrive - Aitken Spence PLC\Documents\Final_Year_Research_Project\Final_year_research-main"
```

### Step 2: Create Virtual Drive (Bypass Path Limits)

Still in the same terminal, run:
```powershell
subst Z: "C:\Users\ast2563\OneDrive - Aitken Spence PLC\Documents\Final_Year_Research_Project\Final_year_research-main"
```

You should now see a `Z:` drive in File Explorer.

### Step 3: Create Virtual Environment

**If `Z:\venv_short` already exists, skip this step.** Otherwise, run:
```powershell
py -3.10 -m venv Z:\venv_short
```

### Step 4: Install Dependencies

```powershell
Z:\venv_short\Scripts\pip install -r requirements_minimal.txt
```

Wait for installation to complete (this may take several minutes).

---

## Running the Application

### Every Time You Want to Run the App:

**Step 1: Make sure you're in the correct directory**
```powershell
cd "C:\Users\ast2563\OneDrive - Aitken Spence PLC\Documents\Final_Year_Research_Project\Final_year_research-main"
```

**Step 2: Run the Streamlit app**
```powershell
Z:\venv_short\Scripts\python -m streamlit run app/streamlit_app.py
```

The app will open in your browser at: **http://localhost:8501**

---

## Troubleshooting

### Error: "File does not exist: app\streamlit_app.py"
You're in the wrong directory. Run:
```powershell
cd "C:\Users\ast2563\OneDrive - Aitken Spence PLC\Documents\Final_Year_Research_Project\Final_year_research-main"
```
Then try running the streamlit command again.

### Error: "Z: drive not found" (after restarting computer)
The virtual drive mapping disappears after restart. Re-run:
```powershell
subst Z: "C:\Users\ast2563\OneDrive - Aitken Spence PLC\Documents\Final_Year_Research_Project\Final_year_research-main"
```

### Error: "Permission denied" when creating venv
The virtual environment already exists. You can skip that step and proceed to installing dependencies.
