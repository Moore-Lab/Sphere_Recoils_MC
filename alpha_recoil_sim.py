## set of helper functions for simulating escape of nuclear recoils from silica spheres

import numpy as np

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

            traj_dict = MC_dict[decay_daughter + '_' + curr_mat]

            ### get a random trajectory for its recoil
            traj_idx = np.random.choice(NUM_SRIM_TRAJ)
            curr_traj_full = traj_dict[traj_idx]

            ## find the starting point along that trajectory that we reach the energy
            ## of interest from above
            idx_before = np.where(curr_traj_full[:,0] > decay_NR_energy)[0][-1]
            idx_after = np.where(curr_traj_full[:,0] <= decay_NR_energy)[0][0]

            print(idx_before, idx_after)

            frac = (decay_NR_energy - curr_traj_full[idx_after,0])/(curr_traj_full[idx_before,0] - curr_traj_full[idx_after,0])
            starting_point = curr_traj_full[idx_after,:] - frac*(curr_traj_full[idx_after,:]-curr_traj_full[idx_before,:])

            print("Starting point: ", starting_point)
            print("Decay energy : ", decay_NR_energy)
            print(curr_traj_full[(idx_before-1):(idx_after+1), :])

            ### choose a random angle for that trajectory

            ### find where the daughter crosses an interface (if at all)


            ### update to the new isotope and t12
            #curr_iso = 
            curr_t12 = decay_dict[curr_iso + "_t12"]


