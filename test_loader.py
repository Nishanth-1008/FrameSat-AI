from core.data_loader import DataLoader
from core.validator import ImageValidator

img1, img2 = DataLoader.load_pair(
    "../Practical-RIFE/demo/I0_0.png",
    "../Practical-RIFE/demo/I0_1.png"
)

ImageValidator.validate(img1, img2)

print("✅ Images validated successfully!")
print(img1.shape)
print(img2.shape)