from core.data_loader import DataLoader

img1, img2 = DataLoader.load_pair(
    "../Practical-RIFE/demo/i0.png",
    "../Practical-RIFE/demo/i1.png"
)

print("Image 1:", img1.shape)
print("Image 2:", img2.shape)