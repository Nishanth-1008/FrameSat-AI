from core.data_loader import DataLoader
from core.preprocessor import ImagePreprocessor
from core.tensor_converter import TensorConverter

img = DataLoader.load_image("../Practical-RIFE/demo/I0_0.png")
img = ImagePreprocessor.preprocess(img)

tensor = TensorConverter.to_tensor(img)

print(tensor.shape)
print(tensor.dtype)
print(tensor.device)