"""ML worker placeholder — job queue processing lands in issue 002."""

import logging
import time

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("ml-worker placeholder running (issue 002 implements job queue)")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
