import logging
import os


class DataLogger(logging.Logger):
    def __init__(self, log_path):
        super().__init__(__name__)  # Initialize the parent class
        self.log_path = log_path

        # Setup logger during initialization
        self.setup()

    def setup(self):
        # Set the logging level
        self.setLevel(logging.INFO)

        # Create a file handler
        handler = logging.FileHandler(self.log_path)

        # Define the logging format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Ensure that the logger doesn't duplicate messages
        if not self.hasHandlers():
            self.addHandler(handler)
