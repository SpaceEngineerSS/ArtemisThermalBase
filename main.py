"""Run ArtemisThermalBase from a source checkout: ``python main.py --help``.

Installed copies provide the same thing as the ``artemis`` command.
"""

import sys

from artemis_thermal.simulation.cli import main

if __name__ == "__main__":
    sys.exit(main())
