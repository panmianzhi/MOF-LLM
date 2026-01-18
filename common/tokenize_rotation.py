import numpy as np
import torch
from common import rigid_utils as ru

def _is_torch(x):
    """Check if input is a torch tensor."""
    return isinstance(x, torch.Tensor)

def create_affine_matrix(R, t):
    """Create a 4x4 affine transformation matrix from a rotation matrix R and a translation vector t.
    
    Args:
        R: 3x3 rotation matrix (numpy array or torch tensor)
        t: 3D translation vector (numpy array or torch tensor)
    
    Returns:
        4x4 affine transformation matrix (same type as input)
    """
    if _is_torch(R):
        A = torch.eye(4, dtype=R.dtype, device=R.device)
        A[:3, :3] = R
        A[:3, 3] = t
    else:
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = t
    return A

def inverse_affine_matrix(A):
    """Compute the inverse of a 4x4 affine transformation matrix.
    
    Args:
        A: 4x4 affine matrix (numpy array or torch tensor)
    
    Returns:
        Inverse matrix (same type as input)
    """
    if _is_torch(A):
        A_inv = torch.linalg.inv(A)
    else:
        A_inv = np.linalg.inv(A)
    return A_inv

def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates to cartesian.
    
    Args:
        r, theta, phi: spherical coordinates (numpy array or torch tensor)
    
    Returns:
        x, y, z: cartesian coordinates (same type as input)
    """
    if _is_torch(r):
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
    else:
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
    return x, y, z

def axis_angle_to_rotation_vector(u, theta):
    """Convert axis-angle to rotation vector.
    
    Args:
        u: rotation axis (numpy array or torch tensor)
        theta: rotation angle (numpy array or torch tensor)
    
    Returns:
        rotation vector (same type as input)
    """
    v = theta * u
    return v

def rotation_vector_to_axis_angle(v):
    """Convert rotation vector to axis-angle representation.
    
    Args:
        v: rotation vector (numpy array or torch tensor)
    
    Returns:
        u: rotation axis (same type as input)
        theta: rotation angle (same type as input)
    """
    if _is_torch(v):
        theta = torch.linalg.norm(v)
        u = v / theta if theta > 0 else torch.tensor([1.0, 0.0, 0.0], dtype=v.dtype, device=v.device)
    else:
        theta = np.linalg.norm(v)
        u = v / theta if theta > 0 else np.array([1.0, 0.0, 0.0])
    return u, theta

def axis_angle_to_rotation_matrix(axis, angle):
    """Convert axis-angle to rotation matrix using Rodrigues' formula.
    
    Args:
        axis: rotation axis (numpy array or torch tensor)
        angle: rotation angle (numpy array or torch tensor)
    
    Returns:
        3x3 rotation matrix (same type as input)
    """
    if _is_torch(axis):
        ux, uy, uz = axis[0], axis[1], axis[2]
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)
        
        # Skew-symmetric matrix for axis
        u_skew = torch.tensor([
            [0, -uz, uy],
            [uz, 0, -ux],
            [-uy, ux, 0]
        ], dtype=axis.dtype, device=axis.device)
        
        # Outer product of axis with itself
        uuT = torch.outer(axis, axis)
        
        # Rodrigues' rotation formula
        rotation_matrix = cos_theta * torch.eye(3, dtype=axis.dtype, device=axis.device) + sin_theta * u_skew + (1 - cos_theta) * uuT
    else:
        ux, uy, uz = axis
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        # Skew-symmetric matrix for axis
        u_skew = np.array([
            [0, -uz, uy],
            [uz, 0, -ux],
            [-uy, ux, 0]
        ])
        
        # Outer product of axis with itself
        uuT = np.outer(axis, axis)
        
        # Rodrigues' rotation formula
        rotation_matrix = cos_theta * np.eye(3) + sin_theta * u_skew + (1 - cos_theta) * uuT
    
    return rotation_matrix

def rotation_matrix_to_axis_angle(R):
    """Convert rotation matrix to axis-angle representation.
    
    Args:
        R: 3x3 rotation matrix (numpy array or torch tensor)
    
    Returns:
        axis: rotation axis (same type as input)
        theta: rotation angle (same type as input)
    """
    if _is_torch(R):
        # Calculate the angle of rotation
        trace_val = torch.trace(R)
        theta = torch.arccos(torch.clamp((trace_val - 1) / 2.0, -1.0, 1.0))
        
        # Avoid division by zero by handling the case when theta is 0
        if theta.item() > 1e-6:
            # Calculate the components of the rotation axis
            ux = (R[2, 1] - R[1, 2]) / (2 * torch.sin(theta))
            uy = (R[0, 2] - R[2, 0]) / (2 * torch.sin(theta))
            uz = (R[1, 0] - R[0, 1]) / (2 * torch.sin(theta))
            axis = torch.tensor([ux, uy, uz], dtype=R.dtype, device=R.device)
        else:
            # For small angles, approximate the axis with any unit vector
            axis = torch.tensor([1.0, 0.0, 0.0], dtype=R.dtype, device=R.device)
    else:
        # Calculate the angle of rotation
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))
        
        # Avoid division by zero by handling the case when theta is 0
        if theta > 1e-6:
            # Calculate the components of the rotation axis
            ux = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
            uy = (R[0, 2] - R[2, 0]) / (2 * np.sin(theta))
            uz = (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))
            axis = np.array([ux, uy, uz])
        else:
            # For small angles, approximate the axis with any unit vector
            axis = np.array([1.0, 0.0, 0.0])
    
    return axis, theta

def isRotationMatrix(R):
    """Check if a matrix is a valid rotation matrix.
    
    Args:
        R: matrix to check (numpy array or torch tensor)
    
    Returns:
        bool: True if R is a rotation matrix
    """
    # square matrix test
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    
    if _is_torch(R):
        should_be_identity = torch.allclose(R @ R.T, torch.eye(R.shape[0], dtype=R.dtype, device=R.device), rtol=1e-3, atol=1e-5)
        should_be_one = torch.allclose(torch.linalg.det(R), torch.tensor(1.0, dtype=R.dtype, device=R.device))
    else:
        should_be_identity = np.allclose(R.dot(R.T), np.identity(R.shape[0], float))
        should_be_one = np.allclose(np.linalg.det(R), 1)
    
    return should_be_identity and should_be_one


# ==================================================
# ==== local coords to global ====
def create_rigid(rots, trans):
    rots = ru.Rotation(rot_mats=rots)
    return ru.Rigid(rots=rots, trans=trans)

def assemble_coords(
    local_coords: torch.Tensor, 
    rotmats: torch.Tensor, 
    trans: torch.Tensor, 
    bb_num_vec: torch.Tensor,
):
    """
    Args:
        local_coords (torch.Tensor): [N, 3]
            local coordinates of the building block
        rotmats (torch.Tensor): [M, 3, 3]
            rotation matrices for each building block
        trans (torch.Tensor): [M, 3]
            translation vectors for each building block
        bb_num_vec (torch.Tensor): [M, ]
            each entry is the number of atoms in the corresponding building block
    
    Returns:
        global_coords (torch.Tensor): [N, 3]
            global coordinates of the building block
    """
    local_coords = torch.split(local_coords, bb_num_vec.tolist())   # M list of [N_i, 3]
    rigids = create_rigid(rotmats, trans)                        # [M,] rigid transformations
    
    assemble_coords = []
    for rigid, bb_local_coords in zip(rigids, local_coords):
        bb_global_coord = rigid.apply(bb_local_coords)
        assemble_coords.append(bb_global_coord)
    
    assemble_coords = torch.cat(assemble_coords, dim=0)
    return assemble_coords