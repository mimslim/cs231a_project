# Project

import numpy as np
from skimage.io import imread
from skimage.io import imsave
import matplotlib.pyplot as plt
import scipy.io as sio


'''
COMPUTE_P
Arguments:
     cube_XY - Each row corresponds to an actual point in 3D space
     image_xy - Each row is projection of 3D point to image plane    
Returns:
    camera_matrix - The calibrated camera matrix (3x4 matrix)
'''

def compute_P(cube_XY, image_xy):

    #print( cube_XY )
    #print( image_xy )

    N = cube_XY.shape[0]

    X = np.ones((N,4))
    x = np.ones((N,3))

    X[:,0:3] = cube_XY
    x[:,0:2] = image_xy

    A = np.zeros((2*N,12))

    A[0:2*N:2,0:4] = X
    A[1:2*N:2,4:8] = X

    for i in range(N):
        A[2*i, 8:12] = -x[i,0]*X[i,:]
        A[2*i+1, 8:12] = -x[i,1]*X[i,:]

    #print('A')
    #print(A)
    #b =  np.reshape(x[:,0:2], (2*N,1) )

    U, s, V = np.linalg.svd(A)
    
    #p = np.linalg.lstsq(A, b, rcond=None)[0]
    #p = np.reshape(p, (2,4))

    #print(V[-1,:])

    M = V[-1,:].reshape((3,4))

    #print(M)

    #M[0:2,:] = p
    #M[2,3] = 1

    #a = input('Press any key: ')
    
    return(M)


'''
Get_K_from_P
Arguments:
    P - camera matrix
Returns:
    K - extracted camera intrinsic matrix
'''
def get_K_from_P(camera_matrix):

    P = camera_matrix
    p1 = P[:,0]
    p2 = P[:,1]
    p3 = P[:,2]
    p4 = P[:,3]

    M = np.array([ p1, p2, p3 ]).T

    M2 = P[:,:3]

    #print(M)
    #print(M2)

    K, R = RQ(M)

    return K



'''
RQ
Arguments:
     A - Original matrix
Returns:
    R, Q - R Q decomposition of A
'''
def RQ(A):

    # Find Qx
    d = np.sqrt( A[2,2]**2 + A[2,1]**2 )
    c = -A[2,2]/d
    s = A[2,1]/d
    Qx = np.array( [ [ 1, 0, 0 ], [ 0, c, -s ], [ 0, s, c ] ] )
    R = A.dot(Qx)

    # Find Qy
    d = np.sqrt( R[2,2]**2 + R[2,0]**2 )
    c = R[2,2]/d
    s = R[2,0]/d
    Qy = np.array( [ [ c, 0, s ], [ 0, 1, 0 ], [ -s, 0, c ] ] )
    R = R.dot(Qy)

    # Find Qz
    d = np.sqrt( R[1,1]**2 + R[1,0]**2 )
    c = -R[1,1]/d
    s = R[1,0]/d
    Qz = np.array( [ [ c, -s, 0 ], [ s, c, 0 ], [ 0, 0, 1 ] ] )
    R = R.dot(Qz)

    Q = Qz.T.dot(Qy.T).dot(Qx.T)

    for i in range(3):
        if R[i,i] < 0:
            R[:,i] = -R[:,i]
            Q[i,:] = -Q[i,:]

    return R, Q
    

'''
P3D_pt_to_image
Arguments:
    x3D - 3D point coordinates in cm
    P - projection matrix onto camera image plane
    pixel_size - pixel size in um
Returns:
    pixels - pixel location
'''
def P3D_pt_to_image(x3D, P, pixel_size):
    
    x3D = 1e4*x3D
    x3D = np.concatenate((x3D, [1]))
    p2D = P.dot(x3D)
    p2D /= p2D[2]

    pixels = np.floor(p2D[0:2]/pixel_size).astype(np.uint16)
    
    return pixels

'''
Multi_P3D_pt_to_image
Arguments:
    x3D - Multiple 3D point coordinates in cm
    P - projection matrix onto camera image plane
    pixel_size - pixel size in um
Returns:
    pixels - pixel location
'''
def Multi_P3D_pt_to_image(x3D, P, pixel_size):
    
    x3D = 1e4*x3D
    N = x3D.shape[1]
    homo = np.ones((1,N))
    x3D = np.concatenate((x3D, homo))
    p2D = P.dot(x3D)
    p2D /= p2D[2,:]

    pixels = np.floor(p2D[0:2,:]/pixel_size).astype(np.uint16)

    return pixels
    
    
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
def Project_square_to_image(square, P, pixel_size, resolution):

    array_dim = np.array([ 1570, 1136 ])

    A = np.zeros((array_dim[1], array_dim[0]))
    
    top_left = square[0:2]
    bottom_right = square[2:4]
    Z = square[4]

    x_pts = np.linspace(top_left[0], bottom_right[0], resolution)
    y_pts = np.linspace(top_left[1], bottom_right[1], resolution)
    z_pts = np.array([Z])

    XX, YY, ZZ = np.meshgrid( x_pts, y_pts, z_pts )
    voxels = np.vstack(( XX.reshape(-1), YY.reshape(-1), ZZ.reshape(-1) )).reshape(3,-1)
                         
    pixels = Multi_P3D_pt_to_image(voxels, P, pixel_size)

    check_bounds = np.all( [ pixels[0] > 0, pixels[0] < array_dim[0], pixels[1] > 0, pixels[1] < array_dim[1] ] )

    if check_bounds:
        A[ pixels[1], pixels[0] ] = 1

    return A, check_bounds


'''
Create_low_light_image
Arguments:
     A - array containing the projected image (elements labelled as 1)
     Lux - illumination level in lux units
Returns:
    Image_8bit - image inlow light conditions accounting for all noise
'''
def Create_low_light_image(A, Lux):

    # ---------- User inputs ----------

    # Pixel information

    Pixel_Size = 8      # Pixel size in um
    CF = 7.7            # Conversion Factor DN/e-
    QE = 0.81           # Quantum Efficiency @540nm
    Read_Noise = 0.5    # Read noise in e-
    DC = 16             # Dark current: electrons per pixel per second e-/s/pix, enter 0 to disregard DC component

    # Illumination information
    #Lux = 0.0001
    #Lux = 0.001
    Photon_Flux = 3.8e15    # ph/s/m2 @ 540nm
    #Photon_Flux = 3.8e14    # ph/s/m2 @ 540nm try only

    #Target_Angular_speed = 10     % degrees rotated per frame

    # Output information

    bits = 11           # Number of DN bits
    Dark_Mean = 170    # DN
    Total_frames = 1  # Number of frames
    FPS = 125


    # ---------- End User inputs ----------


    Total_Noise = np.sqrt( Read_Noise**2 + DC/FPS )  # Total noise from read noise and DC shot noise
    Total_Noise_DN = Total_Noise*CF
    Ph_per_pix = Lux*Photon_Flux*Pixel_Size**2*1E-12*QE/FPS

    # Initializations

    Image0 = Dark_Mean*np.ones( A.shape );

    # Create noisy image

    Image = np.floor( Image0 + Total_Noise_DN*np.random.normal(0, 1, Image0.shape) )
    Noisy_Target = np.floor( np.multiply( A, np.random.poisson( Ph_per_pix, A.shape ) )*CF  )

    Image += Noisy_Target
    Image_8bit = (Image/8).astype(np.uint8);

    return Image_8bit


'''
Create_training_images
Arguments:
     numImages - desired number of images generated
     P - camera_projection matrix    
Returns:
    None
'''
def Create_training_images(numImages, P):

    total = 0
    data = np.zeros((1,6))
    line = np.zeros((1,6))

    while total < numImages:

        Z = np.random.randint(low = 0, high = 50)
        X = np.random.randint(low = -50, high = 100)
        Y = np.random.randint(low = -50, high = 100)

        Lux = np.array([ 0.001, 0.01, 0.1 ])
        lux_idx = np.random.randint(3)        

        square = np.array( [ X, Y, X+10, Y+10, Z ] )

        resolution = 1001
        pixel_size = 8

        A, check = Project_square_to_image(square, P, pixel_size, resolution)
        if check:
            Image = Create_low_light_image(A, Lux[lux_idx])
            imsave('../Training_Images2/Train_' + str(total) + '.jpg', Image )

            line[0,5] = Lux[lux_idx]
            line[0,:5] = square[:5]
            
            data = np.concatenate(( data, line ))
            print('Total = ', total)
            total += 1

    np.save('../Training_Images2/Train_data.npy', data[1:])


if __name__ == '__main__':
    # Loading the example coordinates setup

    '''
    real_XY = np.load('real_XY.npy')
    front_image = np.load('front_image.npy')
    back_image = np.load('back_image.npy')

    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)
    print("Camera Matrix:\n", camera_matrix)
    print()
    print("RMS Error: ", rms_error(camera_matrix, real_XY, front_image, back_image))
    '''

    '''
    A = np.random.rand(3,3)
    R, Q = RQ(A)
    print(R)
    print(Q)
    '''

    '''
    P = np.array( [ [ 3.53553e+2, 3.39645e+2,  2.77744e+2, -1.44946e+6 ],
             [ -1.03528e+2,  2.33212e+1,  4.59607e+2, -6.32525e+5 ],
             [ 7.07107e-1, -3.53553e-1,  6.12372e-1, -9.18559e+2] ])
    print(P)
    K = get_K_from_P(P)

    print(K)
    '''


    print('HWK1411:')

    '''
    cube_XY = np.load('cube_XY_2.npy')
    image_xy = np.load('image_xy_2.npy')

    P = compute_P(cube_XY, image_xy)
    print('P:')
    print(P)
    K = get_K_from_P(P)
    print('K:')
    print(K)Create_low_light_image()

    '''

    P = np.array([[-2.72035420e-05,  3.37476949e-07, -1.04082116e-05, -5.58093089e-01],
     [ 5.21515220e-07, -2.78509130e-05, -4.98409423e-06, -8.29778091e-01],
     [ 1.18743650e-10,  3.43068016e-11, -1.06323887e-09, -6.49894543e-04]] )

    print(P)

    '''
    x3D = np.array([ 100, 100, 50 ])
    pixel_size = 8
    p = P3D_pt_to_image(x3D, P, pixel_size)
    #print(p)

    '''
    pixel_size = 8
    square = np.array( [ 0, 0, 10, 10, 0 ])
    resolution = 1001

    A, check = Project_square_to_image(square, P, pixel_size, resolution)
    print(check)

    plt.imshow(A, cmap='gray')
    #plt.show()
    
    Lux = 0.1
    Image = Create_low_light_image(A, Lux)

    plt.imshow(Image, cmap='gray')
    #plt.show()

    #imsave('test.jpg',Image )

    numImages = 3000

    Create_training_images(numImages, P)

    

    
    
