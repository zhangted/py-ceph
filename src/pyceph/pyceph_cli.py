import pyceph.models as models
from pyceph.Helpers import *
from pyceph.CLIConfig import *

def main():
  config = create_CLI_config()
  image_batch = create_image_batch(config)
  image_batch.process_cli(config)

if __name__ == '__main__':
  main()
