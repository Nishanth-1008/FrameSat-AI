from enum import Enum


class DatasetType(str, Enum):
    """Supported datasets."""

    USER_UPLOAD = "user_upload"

    SEVIR_VIL = "sevir_vil"
    SEVIR_IR069 = "sevir_ir069"
    SEVIR_IR107 = "sevir_ir107"
    SEVIR_VIS = "sevir_vis"

    NOAA_GOES = "noaa_goes"

    INSAT_3D = "insat_3d"

    SENTINEL_2 = "sentinel_2"