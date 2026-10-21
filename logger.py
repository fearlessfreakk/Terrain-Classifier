import csv
import os
from datetime import datetime

LOG_FILE = "logs/terrain_log.csv"

def log_prediction(terrain, confidence, fps, session_id):
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "SessionID", "Timestamp", "Terrain", "Confidence", "FPS"
            ])

        writer.writerow([
            session_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            terrain,
            f"{confidence:.2f}",
            fps
        ])
