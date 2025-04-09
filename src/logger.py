import logging
from datetime import datetime
import os

# Create log directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app_report.txt")

# Configure logger
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def log_input(dataframe):
    logging.info("User Input:\n%s", dataframe.to_string(index=False))

def log_output(prediction):
    result = "Loan Approved" if prediction == 1 else "Loan Not Approved"
    logging.info("Prediction: %s", result)