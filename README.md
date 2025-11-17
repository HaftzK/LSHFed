# LSHFed
Code for "LSHFed: Robust and Communication-Efficient Federated Learning with Locally-Sensitive Hashing Gradient Mapping"



## Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

To use the LSHGM module standalone:
- call `LSH(your_gradient)` in `lsh.py` to obtain the LSHGM bit string
- call `func_LSH(your_gradient1, your_gradient2)` in `lsh.py` to obtain the LSHGM distance between the two tensors.

Feed the tensors layer by layer to ensure numerical precision. Example:
```python
from lsh import LSH,func_LSH
...
for name, data in model.state_dict().items():
    model_update[name] = (data - model_state[name])

bitstring = LSH(model_update['res2.0.0.weight'])
LSHGM_distance = func_LSH(model_update['res2.0.0.weight'],model_update_previous['res2.0.0.weight'])
```

Use `rndvector_gen.py` to generate random vectors required in LSHGM.
