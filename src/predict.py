import sys
sys.path.insert(0, './src')

import models
from Helpers import *
from CLIConfig import *

def main():
  config = create_CLI_config()
  image_batch = create_image_batch(config)
  image_batch.process(config)

if __name__ == '__main__':
  main()
