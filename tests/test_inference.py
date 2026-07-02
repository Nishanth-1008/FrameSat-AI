from core.data_loader import DataLoader
from core.validator import ImageValidator
from core.preprocessor import ImagePreprocessor
from core.tensor_converter import TensorConverter
from app.models.rife_wrapper import RIFEWrapper

# Load
img1, img2 = DataLoader.load_pair(
    "../Practical-RIFE/demo/I0_0.png",
    "../Practical-RIFE/demo/I0_1.png"
)

# Validate
ImageValidator.validate(img1, img2)

# Preprocess
img1 = ImagePreprocessor.preprocess(img1)
img2 = ImagePreprocessor.preprocess(img2)

# Convert to tensors
img1 = TensorConverter.to_tensor(img1)
img2 = TensorConverter.to_tensor(img2)

# Load AI model
rife = RIFEWrapper()

# Generate intermediate frame
middle = rife.interpolate(img1, img2)

print(type(middle))
print(middle.shape)