
'''
COMPUTE_P
Arguments:
     cube_XY - Each row corresponds to an actual point in 3D space
     image_xy - Each row is projection of 3D point to image plane    
Returns:
    camera_matrix - The calibrated camera matrix (3x4 matrix)
'''
'''
Get_K_from_P
Arguments:
    P - camera matrix
Returns:
    K - extracted camera intrinsic matrix
'''
'''
RQ
Arguments:
     A - Original matrix
Returns:
    R, Q - R Q decomposition of A
'''
'''
P3D_pt_to_image
Arguments:
    x3D - 3D point coordinates in cm
    P - projection matrix onto camera image plane
    pixel_size - pixel size in um
Returns:
    pixels - pixel location
'''
'''
Multi_P3D_pt_to_image
Arguments:
    x3D - Multiple 3D point coordinates in cm
    P - projection matrix onto camera image plane
    pixel_size - pixel size in um
Returns:
    pixels - pixel location
''' 
'''
Project_square_to_image
Arguments:
    square - 3D square corner coordinates in cm:
        [ top left (X, Y), bottom right (X, Y), Z ]
    P - projection matrix onto camera image plane
    pixel_size - pixel size in um
    resolution - desired 3D resolution    
Returns:   
    img - image with square projected
    valid - whether square is within bounds
'''
'''
Create_low_light_image
Arguments:
     A - array containing the projected image (elements labelled as 1)
     Lux - illumination level in lux units
Returns:
    Image_8bit - image inlow light conditions accounting for all noise
'''
'''
Create_training_images
Arguments:
     numImages - desired number of images generated
     P - camera_projection matrix    
Returns:
    None
'''


    

    
    
