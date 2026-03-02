"""Entry point for model training and evaluation."""

import logging

from src.train import train_and_evaluate

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    logger.info("Starting training pipeline...")
    train_and_evaluate()
    logger.info("Training pipeline finished.")
