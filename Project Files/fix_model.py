import json
import h5py
from tensorflow import keras
from keras import mixed_precision

original_model_path = "model_cnn (2).h5"

with h5py.File(original_model_path, "r") as f:
    model_config_json = f.attrs["model_config"]

# Replace batch_shape → batch_input_shape
def replace_batch_shape(obj):
    if isinstance(obj, dict):
        return {
            ("batch_input_shape" if k == "batch_shape" else k): replace_batch_shape(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [replace_batch_shape(i) for i in obj]
    else:
        return obj

model_config_dict = json.loads(model_config_json)
fixed_config_dict = replace_batch_shape(model_config_dict)
fixed_config_json = json.dumps(fixed_config_dict)

# ✅ Register a dummy DTypePolicy so Keras doesn't crash
from keras.utils.generic_utils import CustomObjectScope

with CustomObjectScope({'DTypePolicy': mixed_precision.Policy}):
    model = keras.models.model_from_json(fixed_config_json)

model.load_weights(original_model_path)
model.save("model_cnn_fixed.h5")

print("✅ Fixed model saved as model_cnn_fixed.h5")
