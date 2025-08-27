import time
import math
import torch
import orion
import orion.models as models
from orion.core.utils import (
    get_mnist_datasets,
    mae, 
    train_on_mnist
)

orion.set_log_level('INFO')

# Set seed for reproducibility
torch.manual_seed(42)

# Initialize the Orion scheme, model, and data
scheme = orion.init_scheme("../configs/lola.yml")

batch_size = scheme.params.get_batch_size()
trainloader, testloader = get_mnist_datasets(data_dir="../data", batch_size=batch_size)
net = models.LoLA()

# Train model (optional)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# train_on_mnist(net, data_dir="../data", epochs=1, device=device)

# Get a test batch to pass through our network
inp, _ = next(iter(trainloader))

# Run cleartext inference
net.eval()
out_clear = net(inp)

# Prepare for FHE inference. 
orion.fit(net, trainloader)
input_level = orion.compile(net)

# Encode and encrypt the input vector 
vec_ptxt = orion.encode(inp, input_level)
vec_ctxt = orion.encrypt(vec_ptxt)
net.he()  # Switch to FHE mode

# Run FHE inference
print("\nStarting FHE inference", flush=True)
start = time.time()
out_ctxt = net(vec_ctxt)
end = time.time()

# Get the FHE results and decrypt + decode.
out_ptxt = out_ctxt.decrypt()
out_fhe = out_ptxt.decode()

# Compare the cleartext and FHE results.
print()
print(out_clear)
print(out_fhe)

dist = mae(out_clear, out_fhe)
print(f"\nMAE: {dist:.4f}")
print(f"Precision: {-math.log2(dist):.4f}")
print(f"Runtime: {end-start:.4f} secs.\n")