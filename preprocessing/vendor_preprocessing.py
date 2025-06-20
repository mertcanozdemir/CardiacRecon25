from vendor_normalization import normalize_kspace_by_vendor

# Load your k-space data (modify as needed for your data loading)
kspace_data = load_kspace_from_mat(file_path)

# Apply vendor-specific normalization
normalized_kspace = normalize_kspace_by_vendor(
    kspace_data, 
    filepath=file_path  # Will extract vendor info from path
)

# Or specify vendor info manually:
normalized_kspace = normalize_kspace_by_vendor(
    kspace_data,
    vendor="Siemens",
    field_strength=3.0,
    model="Prisma"
)