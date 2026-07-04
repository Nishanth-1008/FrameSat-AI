from app.models.rife_wrapper import RIFEWrapper

rife = RIFEWrapper()

print("✅ RIFE loaded successfully!")
print(type(rife.get_model()))