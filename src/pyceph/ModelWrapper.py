import torch
import gzip
import pkg_resources

from pyceph.Helpers import maybe_terminate
import pyceph.models as models

class ModelWrapper:
  def __init__(self, config, from_cli=True):
    model_path = pkg_resources.resource_filename(__name__, config.model_path)
    maybe_terminate(path=model_path, item_name="Pretrained Model", from_cli=from_cli)

    self.model_path = model_path
    self.device = config.use_gpu

  def load_model(self):
    with gzip.open(self.model_path, 'rb') as zipped_model:
      device = torch.device(self.device)
      print(f"Running inference on device: {device}")
      return torch.load(zipped_model, map_location=device)

'''
# save model as gzip
import gzip
import torch

# Save the model
torch.save(model.state_dict(), 'model.pkl')

# Compress the model
with open('model.pkl', 'rb') as f_in, gzip.open('model.pkl.gz', 'wb') as f_out:
    f_out.writelines(f_in)


# load compressed model
import gzip
import torch

# Load the compressed model
with gzip.open('model.pkl.gz', 'rb') as f:
    model = torch.load(f)
'''
