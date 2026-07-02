from core.data_loader import DataLoader
from core.validator import ImageValidator
from core.preprocessor import ImagePreprocessor

img1, img2 = DataLoader.load_pair(
    "../Practical-RIFE/demo/I0_0.png",
    "../Practical-RIFE/demo/I0_1.png"
)

ImageValidator.validate(img1, img2)

img1 = ImagePreprocessor.preprocess(img1)
img2 = ImagePreprocessor.preprocess(img2)

print("Image 1:", img1.shape, img1.dtype, img1.min(), img1.max())
print("Image 2:", img2.shape, img2.dtype, img2.min(), img2.max())