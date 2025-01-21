import logging

def setup_log():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]  # Log to console
    )
    logging.info("logging setup completed.")
