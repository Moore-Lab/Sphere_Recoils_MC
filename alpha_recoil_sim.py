## set of helper functions for simulating escape of nuclear recoils from silica spheres

import numpy as np
import matplotlib.pyplot as plt

## conversions of hours, days, years to seconds
seconds_dict = {"s": 1, "m": 60, "h": 3600, "d": 3600*24, "y": 3600*24*365.24} 

def parse_decay_chain(file):

    with open(file) as fin:
        lines = fin.readlines()

    decay_chain_dict = {}

    active_iso = ""
    for line in lines:
    
        if line[0] == "#": continue
    
        if(len(active_iso) == 0): ## case to start a new isotope
            iso, t12, alpha_q = line.strip().split(',')
            active_iso = iso
            
            t12_num = float(t12[:-1]) * seconds_dict[t12[-1]] ## convert all half lives to seconds

            decay_chain_dict[active_iso + "_t12"] = t12_num
        
            if(t12_num < 0): break ## quit if stable 
            
            decay_options = []
            decay_daughters = []

        else: ## case to assemble decay possibilities

            if( line.startswith("--") ): ## end of that isotope, restart
                ## add some extra probability if it's missing
                decay_options = np.array(decay_options)
                missing_prob = 1 - np.sum(decay_options[:,0]) ## get any small branches that were missing
                decay_options[-1,0] += missing_prob
                decay_chain_dict[active_iso + "_decays"] = decay_options
                decay_chain_dict[active_iso + "_daughters"] = decay_daughters
                active_iso = ""
            else: ## add another decay option
                parts = line.strip().split(',')
                decay_options.append( [float(parts[0]), float(parts[1])] )
                decay_daughters.append(parts[2].strip())


    return decay_chain_dict
    
iso_Z_dict = {"Pb": 82, "Fr": 87, "At": 85, "Bi": 83, "Tl": 81, "Ra": 88, "Rn": 86, "Po": 84}

def get_Z_A_for_iso(iso):
    symbol = iso[:2]
    Z = iso_Z_dict[symbol]
    A = int(iso[3:])

    return Z, A


def random_point_in_sphere(radius):
    # Generate random coordinates in the sphere with given radius (thanks ChatGPT!)
    u, v, w = np.random.uniform(0, 1, size=3)
    r = w**(1/3) * radius

    # Convert random coordinates to Cartesian coordinates
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return x, y, z

def random_angle_on_sphere():
    # Generate random spherical angles, theta is polar angle, phi azimuthal, psi is 3rd euler
    u, v, w = np.random.uniform(0, 1, size=3)
    phi = 2 * np.pi * u
    theta = np.arcsin(2 * v - 1) ## need sine instead of cos for this convention
    psi = 2 * np.pi * w
    return psi, theta, phi

def rotate_3d_data(points, euler_angles):
    ## take a set of points in 3d (3 column array) and rotate them by some euler angles

    Rx = np.array([[1, 0, 0],
                [0, np.cos(euler_angles[0]), -np.sin(euler_angles[0])],
                [0, np.sin(euler_angles[0]), np.cos(euler_angles[0])]])

    Ry = np.array([[np.cos(euler_angles[1]), 0, np.sin(euler_angles[1])],
                [0, 1, 0],
                [-np.sin(euler_angles[1]), 0, np.cos(euler_angles[1])]])

    Rz = np.array([[np.cos(euler_angles[2]), -np.sin(euler_angles[2]), 0],
                [np.sin(euler_angles[2]), np.cos(euler_angles[2]), 0],
                [0, 0, 1]])

    R = Rz @ Ry @ Rx

    rotated_points = np.dot(R, points.T).T

    return rotated_points



def get_random_decay_time(t12):
    ## return a random decay time for some half life t12
    
    decay_constant = np.log(2)/t12
    decay_time = -np.log(np.random.rand()) / decay_constant

    return decay_time

def first_upward_crossing_time(x, threshold):
    """
    Returns the first time the vector x crosses the threshold from below to above.
    If the vector never crosses the threshold, returns None.
    """
    above_threshold = x > threshold
    crossings = np.where(above_threshold[1:] & ~above_threshold[:-1])[0] + 1
    if len(crossings) == 0:
        return None
    else:
        return crossings[0]
    
def first_downward_crossing_time(x, threshold):
    """
    Returns the first time the vector x crosses the threshold from above to below.
    If the vector never crosses the threshold, returns None.
    """
    below_threshold = x < threshold
    crossings = np.where(below_threshold[1:] & ~below_threshold[:-1])[0] + 1
    if len(crossings) == 0:
        return None
    else:
        return crossings[0]

def select_end_of_traj(traj_full, energy, rotate_to_match, init_xyz, prior_traj=[]):
    """ Find the end of a trajectory after a given energy. Rotate it by the specified angle.
    """

    ## find the starting point along that trajectory that we reach the energy
    ## of interest from above
    idx_before = np.where(traj_full[:,0] > energy)[0][-1]
    idx_after = np.where(traj_full[:,0] <= energy)[0][0]

    frac = (energy - traj_full[idx_after,0])/(traj_full[idx_before,0] - traj_full[idx_after,0])
    starting_point = traj_full[idx_after,:] - frac*(traj_full[idx_after,:]-traj_full[idx_before,:])
    shortened_traj = np.vstack((starting_point, traj_full[idx_after:, :]))

    # start at origin
    for xyzidx in range(1,4):
        shortened_traj[:,xyzidx] -= shortened_traj[0,xyzidx]

    if(rotate_to_match): ### rotate this to match the end of the prior trajectory
        vec2 = 1.0*shortened_traj[1,1:4] ## new direction
        vec1 = prior_traj[-1,1:4]-prior_traj[-2,1:4] ## initial direction
        rotation_angles = get_euler_for_two_traj(vec1, vec2)
        shortened_traj[:,1:4] = rotate_3d_data(shortened_traj[:,1:4], rotation_angles)

        print("Initial traj: ", vec1/np.linalg.norm(vec1))
        print("New traj before rot: ", vec2/np.linalg.norm(vec2))
        print("rotation angles: ", rotation_angles)
        print("New traj after rot:", shortened_traj[1,:]/shortened_traj[1,:])

    else: ### choose a random angle for that trajectory
        shortened_traj[:,1:4] = rotate_3d_data(shortened_traj[:,1:4], random_angle_on_sphere())  
    
    ### now that we have a random, rotated trajectory, add back the original starting point
    for xyzidx in range(1,4):
        shortened_traj[:,xyzidx] += init_xyz[xyzidx-1]

    return shortened_traj

def check_is_stopped(traj, rin, rout):
    """ For a trajectory, check if it stops in the same domain it started. If
        so then return a crossing list. Otherwise, return the point in the trajectory 
        where it crossed. Also return the final domain (0=inner, 1=shell, 2=outside). 
    """
    rad = np.sqrt( traj[:,1]**2 + traj[:,2]**2 + traj[:,3]**2 )

    if( rad[0] <= rin ):
        cross = first_upward_crossing_time(rad, rin)
        final_domain = 1
    else:
        cross = first_downward_crossing_time(rad, rin)
        final_domain = 0

    if(cross):
        is_stopped = False
    else:
        is_stopped = True
        if( rad[-1] <= rin ):
            final_domain = 0
        elif( rad[-1] <= rout ):
            final_domain = 1
        else:
            final_domain = 2

    return is_stopped, cross, final_domain

def get_euler_for_two_traj(vec1, vec2):

    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    theta = np.arccos(cos_theta)

    # Find the axis of rotation using the cross product
    axis = np.cross(vec1, vec2)
    axis = axis / np.linalg.norm(axis)   # normalize the axis vector

    # Choose a rotation sequence (XYZ or ZYX) and calculate the Euler angles
    yaw = np.arctan2(axis[1]*np.sin(theta) + axis[0]*axis[2]*(1-np.cos(theta)), 1-axis[0]**2*np.cos(theta) - axis[1]**2*np.cos(theta))
    pitch = np.arcsin(-axis[0]*np.sin(theta) + axis[1]*axis[2]*(1-np.cos(theta)))
    roll = np.arctan2(axis[0]*np.sin(theta) + axis[1]*axis[2]*(1-np.cos(theta)), 1-axis[1]**2*np.cos(theta) - axis[2]**2*np.cos(theta))

    return np.array([roll, pitch, yaw])

def follow_trajectory(traj, rin, rout, traj_dict_in, traj_dict_out, NUM_SRIM_TRAJ):
    """ Follows a trajectory back and forth until it ends in the same domain
        it started
    """
    final_traj = []
    curr_traj = traj

    ### radius along the trajectory
    is_stopped, cross, final_domain = check_is_stopped(curr_traj, rin, rout)

    print("initial traj check: ", is_stopped, cross, final_domain)
    
    while not is_stopped:

        trimmed_traj = curr_traj[:cross+1, :] ## initial trajectory up to crossing
        if(len(final_traj)>0):
            final_traj = np.vstack((final_traj, trimmed_traj))
        else:
            final_traj = trimmed_traj

        if(final_domain == 0):
            dict_to_use = traj_dict_in
        elif(final_domain == 1):
            dict_to_use = traj_dict_out

        ## get new trajectory in the relevant domain
        new_traj = dict_to_use[np.random.choice(NUM_SRIM_TRAJ)]

        
        shortened_traj = select_end_of_traj(new_traj, trimmed_traj[-1,0], True, 
                                            trimmed_traj[-1,1:4], prior_traj=trimmed_traj)


        curr_traj = shortened_traj

        ##check if this new traj is stopped or not        
        is_stopped, cross, final_domain = check_is_stopped(curr_traj, rin, rout)
            
        print("following traj check: ", is_stopped, cross, final_domain)

    ## now that we're stopped, add the last bit of trajectory
    if(len(final_traj)>0):
        final_traj = np.vstack((final_traj, curr_traj))
    else:
        final_traj = curr_traj

    return final_traj, final_domain

def plot_trajectory(data, rin, rout):
    # Create a 3D plot
    fig = plt.figure(figsize=(12,3))
    #ax = fig.add_subplot(141, projection='3d')

    ## Plot the array in 3D
    #ax.plot3D(data[:,0], data[:,1], data[:,2], c='k', marker='o')

    circle_in = plt.Circle((0, 0), rin, facecolor='none', edgecolor='k', linestyle="-")
    circle_out = plt.Circle((0, 0), rout, facecolor='none', edgecolor='k', linestyle="-")

    ax1 = plt.subplot(1,3,1)
    ax1.plot(data[:,1], data[:,2], 'k-')
    c=ax1.scatter(data[:,1], data[:,2], c=data[:,0])
    ax1.set_xlim(-rout*1.2, rout*1.2)
    ax1.set_ylim(-rout*1.2, rout*1.2)
    ax1.add_patch(circle_in)
    ax1.add_patch(circle_out)
    plt.colorbar(c)

    circle_in = plt.Circle((0, 0), rin, facecolor='none', edgecolor='k', linestyle="-")
    circle_out = plt.Circle((0, 0), rout, facecolor='none', edgecolor='k', linestyle="-")

    ax2=plt.subplot(1,3,2)
    ax2.plot(data[:,2], data[:,3], 'k-')
    c=ax2.scatter(data[:,2], data[:,3], c=data[:,0])
    ax2.set_xlim(-rout*1.2, rout*1.2)
    ax2.set_ylim(-rout*1.2, rout*1.2)
    ax2.add_patch(circle_in)
    ax2.add_patch(circle_out)
    plt.colorbar(c)

    circle_in = plt.Circle((0, 0), rin, facecolor='none', edgecolor='k', linestyle="-")
    circle_out = plt.Circle((0, 0), rout, facecolor='none', edgecolor='k', linestyle="-")
    ax3 = plt.subplot(1,3,3)
    ax3.plot(data[:,1], data[:,3], 'k-')
    c=ax3.scatter(data[:,1], data[:,3], c=data[:,0])
    ax3.set_xlim(-rout*1.2, rout*1.2)
    ax3.set_ylim(-rout*1.2, rout*1.2)
    ax3.add_patch(circle_in)
    ax3.add_patch(circle_out)
    plt.colorbar(c)

    # Show the plot
    plt.show()

def sim_N_events(nmc, iso, iso_dict, sphere_dict, MC_dict):
    """ Function to simulate the alpha transport through a sphere
    """

    NUM_SRIM_TRAJ = 10000 ## number of simulated trajectories for each ion type
    alpha_mass = 4 ## mass of alpha particle in AMU

    ## construct sphere materials and parameters
    r_inner = sphere_dict['inner_radius']
    r_outer = r_inner + sphere_dict['outer_shell_thick']

    mat_inner = sphere_dict['inner_material']
    mat_outer = sphere_dict['shell_material']

    decay_dict = iso_dict[iso]

    for n in range(nmc):

        curr_iso = iso ## starting isotope at top of chain
        t = 0 ## start at time zero
        x, y, z = random_point_in_sphere(r_inner)
        print(x,y,z)
        curr_mat = mat_inner
        curr_t12 = decay_dict[curr_iso + "_t12"] ## get this to start the while loop

        event_record = {} ## hold details of the entire decay chain for this event
        decay_idx = 0

        while curr_t12 > 0:

            decay_record = {} ## hold details of this decay

            ### get a random time at which this isotope decays
            t_decay = get_random_decay_time(curr_t12)
            t += t_decay
            
            decay_record['time'] = t

            ### find the daughter that recoils and its starting energy
            curr_decay_info = decay_dict[curr_iso + "_decays"]
            curr_daughter_info = decay_dict[curr_iso + "_daughters"]

            rand_idx = np.random.choice(len(curr_daughter_info), p=curr_decay_info[:,0])

            decay_alpha_energy = curr_decay_info[rand_idx,1] ## in keV
            decay_daughter = curr_daughter_info[rand_idx]
            daughter_mass = float(curr_iso.split("-")[-1])
            decay_NR_energy = decay_alpha_energy * alpha_mass/daughter_mass

            decay_record['energy'] = decay_NR_energy
            decay_record['iso'] = decay_daughter

            traj_dict = MC_dict[decay_daughter + '_' + curr_mat]

            ### get a random trajectory for its recoil
            traj_idx = np.random.choice(NUM_SRIM_TRAJ)
            curr_traj_full = traj_dict[traj_idx]
            init_xyz = [x,y,z]
            shortened_traj = select_end_of_traj(curr_traj_full,decay_NR_energy, False, init_xyz)

            traj_dict_inner = MC_dict[decay_daughter + '_' + mat_inner]
            traj_dict_outer = MC_dict[decay_daughter + '_' + mat_outer]
            ## follow trajectory until it stops
            final_traj, final_domain = follow_trajectory(shortened_traj, r_inner, r_outer, 
                                                         traj_dict_inner, traj_dict_outer, NUM_SRIM_TRAJ)

            plot_trajectory(final_traj, r_inner, r_outer)
            break

            ### update to the new isotope and t12
            curr_iso = decay_daughter
            curr_t12 = decay_dict[curr_iso + "_t12"]


