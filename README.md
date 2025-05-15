# Higher-Order Ambisonics (HOA)

This is a python implementation of HOA. This repo supports: (1) extract HOA coefficients from a signal. (2) Reconstruct a signal from HOA coefficients.

## Usage

```python
import numpy as np
from hoa import forward_hoa

hoa = forward_hoa(value=1., azi=np.deg2rad(20), col=np.deg2rad(90), order=3)  # (order+1)^2
```

See more examples by running the follow command.

```python
python hoa.py
```

## Visualization of HOA coefficients

<img src="https://github.com/user-attachments/assets/dbce4800-6443-4365-82bd-82010a29f7b1" width="600">
