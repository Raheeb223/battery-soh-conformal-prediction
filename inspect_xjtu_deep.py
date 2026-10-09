"""
One-off: peek an ACTUAL per-cell XJTU .mat file (not the irrelevant
Temperature_Compensation_Data.mat that the generic inspector happened to
grab). Edit the path below if needed, then run and paste the output back.
"""

import scipy.io as sio

PATH = r"C:\Users\User\PycharmProjects\Battery Dataset\Batch-1\2C_battery-1.mat"

mat = sio.loadmat(PATH, squeeze_me=True, struct_as_record=False)
top_keys = [k for k in mat.keys() if not k.startswith("__")]
print(f"Top-level keys: {top_keys}")

for k in top_keys:
    val = mat[k]
    print(f"\n--- key '{k}' ---")
    print(f"  type: {type(val)}")
    fields = [f for f in dir(val) if not f.startswith("_")]
    # filter out numpy ndarray builtin methods so we see only real data fields
    real_fields = [f for f in fields if f not in (
        'all','any','argmax','argmin','argpartition','argsort','astype','base',
        'byteswap','choose','clip','compress','conj','conjugate','copy','ctypes',
        'cumprod','cumsum','data','diagonal','dot','dtype','dump','dumps','fill',
        'flags','flat','flatten','getfield','imag','item','itemset','itemsize',
        'max','mean','min','nbytes','ndim','newbyteorder','nonzero','partition',
        'prod','ptp','put','ravel','real','repeat','reshape','resize','round',
        'searchsorted','setfield','setflags','shape','size','sort','squeeze',
        'std','strides','sum','swapaxes','take','tobytes','tofile','tolist',
        'tostring','trace','transpose','var','view'
    )]
    print(f"  real (non-builtin) fields: {real_fields}")
    if hasattr(val, 'shape'):
        print(f"  shape: {val.shape}, dtype: {val.dtype}")
        try:
            print(f"  sample values: {val.flatten()[:5]}")
        except Exception as e:
            print(f"  (could not sample: {e})")
