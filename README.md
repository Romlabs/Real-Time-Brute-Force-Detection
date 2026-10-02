# Real-Time Brute-Force Detection (Streamlit)

Random Forest detector for attack sessions, with a Streamlit GUI.

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface (performance, live stream, session check, batch scoring, data explorer, retrain) |
| `brute_force_detector.py` | ML logic (cleaning, encoding, SMOTE, tuning, evaluation, prediction, simulation) |
| `train_model.py` | Trains once and saves `model/brute_force_model.joblib` |
| `requirements.txt` | Dependencies |

## Setup (one time)
```bash
pip install -r requirements.txt
python train_model.py            # downloads the Kaggle dataset, trains, saves the model
# python train_model.py --tune   # optional: Randomized Search
```
Commit the generated `model/brute_force_model.joblib` to GitHub. The app loads it at startup,
so visitors see results immediately and nothing has to be retrained.

> Train with the same Python and scikit-learn versions the deployed app uses, otherwise the
> saved model may fail to load. If the file is over ~90 MB, run `train_model.py --max-depth 15`.

## Run locally
```bash
streamlit run app.py
```

## Deploy from GitHub (Streamlit Community Cloud)
1. Push all files to the repo root (avoid spaces in folder names).
2. At https://share.streamlit.io choose **Create app**, select the repo, set the main file to `app.py`.
3. Under **Advanced settings** choose Python 3.12.

## Data
The dataset (`dnkumars/cybersecurity-intrusion-detection-dataset`) is only needed to train, explore
or retrain. Expected columns: `network_packet_size, protocol_type, login_attempts, session_duration,
encryption_used, ip_reputation_score, failed_logins, browser_type, unusual_time_access,
attack_detected` (`session_id` is optional and is dropped).
