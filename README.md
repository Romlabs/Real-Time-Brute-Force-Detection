# Real-Time Brute-Force Detection (Streamlit)

Streamlit GUI for a Random Forest brute-force / intrusion detector.

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface (data explorer, training, live check, batch scoring) |
| `brute_force_detector.py` | ML logic (cleaning, encoding, SMOTE, tuning, evaluation, prediction) |
| `requirements.txt` | Dependencies |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy from GitHub (Streamlit Community Cloud)
1. Create a GitHub repo and push these files to the repo root.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick your repo and branch, set the main file to `app.py`, then **Deploy**.

## Data
Use the sidebar to either upload the CSV or download it from Kaggle
(`dnkumars/cybersecurity-intrusion-detection-dataset`).
Expected columns: `network_packet_size, protocol_type, login_attempts, session_duration,
encryption_used, ip_reputation_score, failed_logins, browser_type, unusual_time_access,
attack_detected` (`session_id` is optional and is dropped).
