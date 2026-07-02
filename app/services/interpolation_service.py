from core.data_loader import DataLoader
from core.validator import ImageValidator
from core.preprocessor import ImagePreprocessor
from core.tensor_converter import TensorConverter
from core.postprocessor import PostProcessor
from app.models.rife_wrapper import RIFEWrapper


class InterpolationService:
    """High-level service for frame interpolation."""

    def __init__(self):
        self.rife = RIFEWrapper()

    def interpolate(self, frame1_path: str, frame2_path: str, output_path: str):
        # Load
        img1, img2 = DataLoader.load_pair(frame1_path, frame2_path)

        # Validate
        ImageValidator.validate(img1, img2)

        # Preprocess
        img1 = ImagePreprocessor.preprocess(img1)
        img2 = ImagePreprocessor.preprocess(img2)

        # Convert to tensors
        img1 = TensorConverter.to_tensor(img1)
        img2 = TensorConverter.to_tensor(img2)

        # AI inference
        output_tensor = self.rife.interpolate(img1, img2)

        # Convert back to image
        output_image = PostProcessor.tensor_to_image(output_tensor)

        # Save result
        PostProcessor.save_image(output_image, output_path)

        return output_path