from app.services.interpolation_service import InterpolationService

service = InterpolationService()

output = service.interpolate(
    "../Practical-RIFE/demo/I0_0.png",
    "../Practical-RIFE/demo/I0_1.png",
    "outputs/service_result.png"
)

print("Generated:", output)