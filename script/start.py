import argparse

parser = argparse.ArgumentParser()
parser.add_argument("x", help="the base", type=int)
parser.add_argument("y", help="the exponent", type=int)
parser.add_argument("-v", "--verbose", help="Increase output verbosity", action='count', default=0)
args = parser.parse_args()

answer = args.x ** args.y

if args.verbose >= 2:
	print(f"Running {__file__}")
if args.verbose == 1:
	print(f"{args.x} power {args.y} = {answer}")
else:
	print(answer)