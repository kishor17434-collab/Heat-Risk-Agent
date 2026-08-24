import logging

def configure_logging(level=logging.INFO) -> None:
    """Standardized logging configuration for CLI scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
