from core.data_loader import DataLoader
from core.validator import ImageValidator
from core.preprocessor import ImagePreprocessor
from core.tensor_converter import TensorConverter
from core.postprocessor import PostProcessor
from app.models.rife_wrapper import RIFEWrapper

img1, img2 = DataLoader.load_pair(
    "../Practical-RIFE/demo/I2_0.png",
    "../Practical-RIFE/demo/I2_1.png"
)

ImageValidator.validate(img1, img2)

img1 = ImagePreprocessor.preprocess(img1)
img2 = ImagePreprocessor.preprocess(img2)

img1 = TensorConverter.to_tensor(img1)
img2 = TensorConverter.to_tensor(img2)

rife = RIFEWrapper()

middle = rife.interpolate(img1, img2)

output = PostProcessor.tensor_to_image(middle)

PostProcessor.save_image(output, "outputs/interpolated.png")

print("✅ Saved outputs/interpolated.png")