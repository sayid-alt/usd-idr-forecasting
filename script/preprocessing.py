import os
import argparse
from dotenv import load_dotenv

from usd_idr_forecasting.configs import ProjectConfig
from usd_idr_forecasting.data.datasets import DatasetLoader


load_dotenv()
PROJECT_WORKING_DIR = os.getenv("PROJECT_WORKING_DIR")


parser = argparse.ArgumentParser(prog='Data Preprocessing')
parser.add_argument(
	"--for-research",
	action="store_true",
	default=False,
	help="Load only dataset used by research paper",
)

parser.add_argument(
	"-v", "--verbose",
	action="store_true",
	default=0,
	help="Increase output verbosity",
)

args = parser.parse_args()

for_research = args.for_research

print(for_research)