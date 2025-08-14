import schedule
import time
from retrain_model import run_retraining
from datetime import datetime

def should_retrain():
    return True  # ✅ Force trigger for testing

def job():
    print("📅 Scheduled job triggered — retraining starts now...")
    run_retraining()
    print("✅ Retraining done. Exiting loop.")
    global stop_loop
    stop_loop = True

schedule.every(15).seconds.do(job)
print("🕒 Scheduler test started — retraining will begin in 15 seconds...")

stop_loop = False
while not stop_loop:
    schedule.run_pending()
    time.sleep(1)
