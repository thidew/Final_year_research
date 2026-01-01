# Intelligent Hotel Review Analysis & Recommendation Engine v2.0 🏨

An advanced AI system that not only analyzes sentiment but maps negative feedback to specific, actionable operational interventions.

## 🚀 Key Features

*   **Granular Action Recommendation**: Maps complaints to **13 specific management actions** (e.g., "Pest Control Investigation" vs "Deep Clean") instead of generic advice.
*   **High-Accuracy Models**:
    *   **Recommendation Engine**: 100% Accuracy (Stacking Ensemble: RF + XGB + LGBM).
    *   **Sentiment Analysis**: 71% Accuracy.
*   **Analytics Dashboard**: Visual insights into category performance and complaint trends.
*   **Batch Processing**: Drag-and-drop CSV processing for bulk analysis.

## 🛠️ Architecture

*   **Logic**: `src/` (Modular Python code)
*   **UI**: `app/streamlit_app.py` (Streamlit 1.30+)
*   **Data**: `data/processed/` (Synthetic & Real datasets)
*   **Models**: `models/` (Pickled Scikit-learn pipes)

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/hotel-review-ai.git
    cd hotel-review-ai
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

1.  **Run the Application**:
    ```bash
    python -m streamlit run app/streamlit_app.py
    ```
    *(Note: Use `app/streamlit_app.py`, not the legacy root script)*

2.  **Access the Dashboard**:
    Open `http://localhost:8501` in your browser.

3.  **Modes**:
    *   **Single Review**: Type a review to get instant sentiment and action analysis.
    *   **Batch Mode**: Upload a CSV to process thousands of reviews at once.

## 🧠 Model Details

### Action Schema
The system maps feedback to the following granular actions:

| Category | Actions |
| :--- | :--- |
| **Food** | `Inspect Kitchen Hygiene`, `Revise Menu Quality`, `Staff Training (Food)`, `Compensate Guest` |
| **Rooms** | `Deep Clean & Maintenance`, `Upgrade Room Amenities`, `Pest Control Investigation` |
| **Services** | `Staff Communication Training`, `Review Check-in Process`, `Disciplinary Action` |
| **Recreation** | `Pool Maintenance`, `Update Gym Equipment`, `Recreation Staff Review` |

### Technology Stack
*   **Python 3.10+**
*   **Sklearn / XGBoost / LightGBM**
*   **Streamlit**
*   **Pandas / NumPy**

## 📄 License
MIT License
