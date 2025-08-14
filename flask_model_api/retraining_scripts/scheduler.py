"""
scheduler.py

This script automates the model retraining process.

How it works:
- Runs daily at 2:00 AM server time.
- Checks whether today is the 1st day of a new quarter (Jan, Apr, Jul, Oct).
- If true, it calls `run_retraining()` to update clustering models and personas.

This allows retraining to happen on a fixed quarterly schedule without manual intervention.
To keep it running, this script must stay active in the background (e.g., using a process manager or cron with nohup).
"""

import schedule
import time
from retrain_model import run_retraining
from datetime import datetime

def should_retrain():
    today = datetime.today()
    return today.month in [1, 4, 7, 10] and today.day == 1  # Jan 1, Apr 1, Jul 1, Oct 1

def job():
    if should_retrain():
        print(" Running scheduled retraining...")
        run_retraining()
    else:
        print(" Not time yet. Waiting...")

schedule.every().day.at("02:00").do(job)  # Run daily check at 8 AM

while True:
    schedule.run_pending()
    time.sleep(60)
