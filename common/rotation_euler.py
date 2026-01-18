import torch

def euler_to_matrix(euler_angles: torch.Tensor, degrees: bool = False) -> torch.Tensor:
    """
    Convert Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.
    Uses the convention: R = Rz * Ry * Rx (ZYX intrinsic rotations)
    
    Args:
        euler_angles: Tensor of shape (3,) containing [roll, pitch, yaw]
        degrees: If True, input angles are in degrees; if False (default), in radians
    
    Returns:
        3x3 rotation matrix
    """
    # Convert to radians if input is in degrees
    if degrees:
        euler_angles = torch.deg2rad(euler_angles)
    
    x, y, z = euler_angles[0], euler_angles[1], euler_angles[2]

    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    # Construct 3x3 rotation matrix (R = Rz * Ry * Rx)
    # Row 1
    r00 = cy * cz
    r01 = -cx * sz + sx * sy * cz
    r02 = sx * sz + cx * sy * cz
    
    # Row 2
    r10 = cy * sz
    r11 = cx * cz + sx * sy * sz
    r12 = -sx * cz + cx * sy * sz
    
    # Row 3
    r20 = -sy
    r21 = sx * cy
    r22 = cx * cy

    matrix = torch.tensor([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ], dtype=euler_angles.dtype, device=euler_angles.device)

    return matrix

def matrix_to_euler(matrix: torch.Tensor, degrees: bool = False) -> torch.Tensor:
    """
    Convert a 3x3 rotation matrix back to Euler angles (roll, pitch, yaw).
    Assumes the matrix was created using R = Rz * Ry * Rx convention.
    
    Args:
        matrix: 3x3 rotation matrix
        degrees: If True, return angles in degrees; if False (default), return in radians
    
    Returns:
        Tensor of shape (3,) containing [roll, pitch, yaw] in radians or degrees
    """
    # 1. Calculate Pitch (Y-axis rotation)
    # Matrix element [2, 0] = -sin(y)
    sy = -matrix[2, 0]
    
    # Numerical stability: clamp to [-1, 1] to prevent NaN in asin
    sy = torch.clamp(sy, -1.0, 1.0)
    pitch = torch.asin(sy)

    # 2. Check for gimbal lock
    # When pitch is close to +/- 90 degrees (cos(pitch) ≈ 0),
    # we cannot uniquely distinguish roll and yaw
    # Threshold is 0.999999 (about 89.99 degrees)
    if torch.abs(matrix[2, 0]) > 0.999999:
        # === Gimbal lock case ===
        # Set roll to 0 and compute only yaw
        # Mathematical derivation shows yaw can be computed via atan2(-r01, r11)
        roll = torch.tensor(0.0, dtype=matrix.dtype, device=matrix.device)
        yaw = torch.atan2(-matrix[0, 1], matrix[1, 1])
    else:
        # === Normal case ===
        # Roll (X-axis): atan2(r21, r22)
        roll = torch.atan2(matrix[2, 1], matrix[2, 2])
        
        # Yaw (Z-axis): atan2(r10, r00)
        yaw = torch.atan2(matrix[1, 0], matrix[0, 0])

    result = torch.stack([roll, pitch, yaw])
    
    if degrees:
        result = torch.rad2deg(result)
    
    return result