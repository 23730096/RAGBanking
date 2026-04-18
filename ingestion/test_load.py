from core.setup_logging import setup_logging
from ingestion.load_data import load_data

setup_logging()

data = load_data(file_path="data/raw/vcb-FAQ - SMS Banking.pdf")

print(data)