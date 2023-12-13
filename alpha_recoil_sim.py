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

def random_point_on_surface(r):
    # Generate random coordinates in the sphere with given radius (thanks ChatGPT!)
    u, v = np.random.uniform(0, 1, size=2)

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

def rotate_3d_data(points, euler_angles, R=[]):
    ## take a set of points in 3d (3 column array) and rotate them by some euler angles

    if(len(euler_angles)>0):
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

    if(energy < traj_full[-1,0]): 
        ## here the energy is less then the end of the traj, so we should just take the last two points
        shortened_traj = traj_full[-2:, :]
    else:
        idx_before = np.where(traj_full[:,0] > energy)[0][-1]
        idx_after = np.where(traj_full[:,0] <= energy)[0][0]

        frac = (energy - traj_full[idx_after,0])/(traj_full[idx_before,0] - traj_full[idx_after,0])
        starting_point = traj_full[idx_after,:] - frac*(traj_full[idx_after,:]-traj_full[idx_before,:])
        shortened_traj = np.vstack((starting_point, traj_full[idx_after:, :]))

    # start at origin
    for xyzidx in range(1,4):
        shortened_traj[:,xyzidx] -= shortened_traj[0,xyzidx]

    if(rotate_to_match): ### rotate this to match the end of the prior trajectory
        vec1 = 1.0*shortened_traj[1,1:4] ## new direction (want to rotate this back to initial)
        vec2 = prior_traj[-1,1:4]-prior_traj[-2,1:4] ## initial direction
        R = get_euler_for_two_traj(vec2, vec1)
        shortened_traj[:,1:4] = rotate_3d_data(shortened_traj[:,1:4], [], R)

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

    if(cross and traj[cross,0]>1.0): ## make sure it crosses with at least 1 keV
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

    ct = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    st = np.sqrt(1 - ct**2)

    # Find the axis of rotation using the cross product
    u = -np.cross(vec1, vec2) ## minus to do inverse rotation
    u = u / np.linalg.norm(u)   # normalize the axis vector
    ux, uy, uz = u[0], u[1], u[2]

    R = [[ct + ux**2 * (1-ct), ux*uy*(1-ct)-uz*st, ux*uz*(1-ct) + uy*st],
         [uy*ux*(1-ct) + uz*st, ct + uy**2*(1-ct), uy*uz*(1-ct) - ux*st],
         [uz*ux*(1-ct) - uy*st, uz*uy*(1-ct) + ux*st, ct + uz**2*(1-ct)]]

    return np.array(R)

def follow_trajectory(traj, rin, rout, traj_dict_in, traj_dict_out, NUM_SRIM_TRAJ):
    """ Follows a trajectory back and forth until it ends in the same domain
        it started
    """
    final_traj = []
    curr_traj = traj

    ### radius along the trajectory
    is_stopped, cross, final_domain = check_is_stopped(curr_traj, rin, rout)
    
    while not is_stopped:

        trimmed_traj = curr_traj[:(cross+1), :] ## initial trajectory up to crossing (+1)
        if(len(final_traj)>0):
            final_traj = np.vstack((final_traj, trimmed_traj))
        else:
            final_traj = trimmed_traj

        if(final_domain == 0):
            dict_to_use = traj_dict_in
        elif(final_domain == 1):
            dict_to_use = traj_dict_out

        ## get new trajectory in the relevant domain
        new_traj = dict_to_use[np.random.choice(NUM_SRIM_TRAJ)+1]

        
        shortened_traj = select_end_of_traj(new_traj, trimmed_traj[-1,0], True, 
                                            trimmed_traj[-1,1:4], prior_traj=trimmed_traj)

        curr_traj = shortened_traj

        ##check if this new traj is stopped or not        
        is_stopped, cross, final_domain = check_is_stopped(curr_traj, rin, rout)

    ## now that we're stopped, add the last bit of trajectory
    if(len(final_traj)>0):
        final_traj = np.vstack((final_traj, curr_traj))
    else:
        final_traj = curr_traj

    return final_traj, final_domain

def plot_circles(ax, rin, rout, cin, cout):
    circle_in = plt.Circle((0, 0), rin, facecolor='none', edgecolor=cin, linestyle="-")
    circle_out = plt.Circle((0, 0), rout, facecolor='none', edgecolor=cout, linestyle="-")
    ax.add_patch(circle_in)
    ax.add_patch(circle_out)

def plot_sphere(ax, r, c, alph=0.5):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r*np.outer(np.cos(u), np.sin(v))
    y = r*np.outer(np.sin(u), np.sin(v))
    z = r*np.outer(np.ones(np.size(u)), np.cos(v))

    # plot sphere with transparency
    ax.plot_surface(x, y, z, alpha=alph, color=c)

def plot_event(event_dict, sd, rad_lims=[], sphere_coords=True):

    color_list = ['k', 'r', 'b', 'g', 'c', 'm', 'y', 'orange', 'purple']
    c1_list = [1,2,1]
    c2_list = [2,3,3]

    sphere_colors = {"SiO2": "gray", "Au": "gold", "Ag": "silver"}

    rin = sd['inner_radius']
    rout = sd['inner_radius'] + sd['outer_shell_thick']

    inner_sphere_color = sphere_colors[sd["inner_material"]]
    outer_sphere_color = sphere_colors[sd["shell_material"]]

    # Create a 3D plot
    fig = plt.figure(figsize=(12,8), facecolor='white', layout='constrained')

    subfigs = fig.subfigures(2, 1, hspace=0.07, height_ratios=[2.5, 1])

    ax3d = subfigs[0].add_subplot(1,1,1, projection='3d')
    ax2d = subfigs[1].subplots(1,4)
            
    if(sphere_coords and False):
        for ax in ax2d[:-1]:
            plot_circles(ax, rin, rout, inner_sphere_color, outer_sphere_color)

    decays = event_dict.keys()
    lost_e = 0

    ## plot the starting location of the parent
    xyz = event_dict['start_pos']
    ax3d.scatter(xyz[0], xyz[1], xyz[2], 'o', c=color_list[0], label=event_dict['parent']) ## plot the starting location
    if(sphere_coords):
        for j,ax in enumerate(ax2d[:-1]):
            ax.plot(xyz[c1_list[j]-1],xyz[c2_list[j]-1], 'o', color=color_list[0])
    else:
        for j,ax in enumerate(ax2d[:-1]):
            ax.plot(0,xyz[j], 'o', color=color_list[0])
            
    ax2d[-1].plot(0,np.linalg.norm(xyz),'o', c=color_list[0])   
    data = np.array( [[0, xyz[0], xyz[1], xyz[2]],])
    curr_lost_e = [0]
    rad = [np.linalg.norm(xyz)]

    for didx in range(len(decays)-4): ## four non numerical keys

        idx_for_colors = (didx + 1) ## starting isotope is 0

        data = event_dict[didx]['traj']

        if(event_dict[didx]['energy']==0): 
            ## this is a beta, so just plot its dot at the same position and go on
            ax3d.scatter(data[-1,1], data[-1,2], data[-1,3], 'o', s=2, c=color_list[idx_for_colors], label=event_dict[didx]['iso'])
            if(sphere_coords):
                for j,ax in enumerate(ax2d[:-1]):
                    ax.plot(data[-1,c1_list[j]], data[-1,c2_list[j]], 'o', ms=2, color=color_list[idx_for_colors])
            else:
                for j,ax in enumerate(ax2d[:-1]):
                    ax.plot(curr_lost_e[-1], data[-1,j+1], 'o', ms=2, color=color_list[idx_for_colors])
            ax2d[-1].plot(curr_lost_e[-1], rad[-1], 'o', ms=2, color=color_list[idx_for_colors])
            continue



        ## Plot the array in 3D
        ax3d.plot3D(data[:,1], data[:,2], data[:,3], c=color_list[idx_for_colors])
        ax3d.scatter(data[-1,1], data[-1,2], data[-1,3], 'o', c=color_list[idx_for_colors], label=event_dict[didx]['iso'])

        ## now plot the radius
        rad = np.sqrt( data[:,1]**2 + data[:,2]**2 + data[:,3]**2 )
        curr_lost_e = lost_e + (data[0,0]-data[:,0])
        ax2d[-1].plot(curr_lost_e, rad, '-', color=color_list[idx_for_colors])
        ax2d[-1].plot(curr_lost_e[-1], rad[-1], 'o', color=color_list[idx_for_colors])
        lost_e = curr_lost_e[-1]

        if(sphere_coords):
            for j,ax in enumerate(ax2d[:-1]):
                ax.plot(data[:,c1_list[j]], data[:,c2_list[j]], '-', color=color_list[idx_for_colors])
                ax.plot(data[-1,c1_list[j]], data[-1,c2_list[j]], 'o', color=color_list[idx_for_colors])
        else:
            for j,ax in enumerate(ax2d[:-1]):
                ax.plot(curr_lost_e, data[:,j+1], '-', color=color_list[idx_for_colors])
                ax.plot(curr_lost_e[-1], data[-1,j+1], 'o', color=color_list[idx_for_colors])

    ax3d.set_xlim(-rout*1.2, rout*1.2)
    ax3d.set_ylim(-rout*1.2, rout*1.2)
    ax3d.set_zlim(-rout*1.2, rout*1.2)
    ax3d.set_xlabel('x [nm]')
    ax3d.set_ylabel('y [nm]')
    ax3d.set_zlabel('z [nm]')
    ax3d.legend(bbox_to_anchor=(1.3, 0.5))

    if(sphere_coords):
        col_names = ['x', 'y', 'z']
        for j,ax in enumerate(ax2d[:-1]):
            c1, c2 = c1_list[j], c2_list[j]
            ax.set_xlabel(col_names[c1-1] + " [nm]")
            ax.set_ylabel(col_names[c2-1] + " [nm]")
            #ax.set_xlim(-rout*1.2, rout*1.2)
            #ax.set_ylim(-rout*1.2, rout*1.2)

    ax2d[-1].set_xlabel('Cumulative energy loss [keV]')
    ax2d[-1].set_ylabel('Radius [nm]')
    xm = curr_lost_e[-1]*1.1
    if(len(rad_lims)==0):
        ax2d[-1].set_ylim(0, rout*1.2)
        plt.text(xm*0.99, rin*1.01, sd["inner_material"], va='bottom', ha='right', color='k')
        plt.text(xm*0.99, rout*1.01, sd["shell_material"], va='bottom', ha='right', color='k')
    else:
        if(rad_lims[0] > 0):
            ax2d[-1].set_ylim(rad_lims[0], rad_lims[1])
    ax2d[-1].set_xlim(-0.01*xm, xm)
    #ax2d[-1].plot([0, xm],[rin, rin], color=inner_sphere_color)
    ax2d[-1].plot([0, xm],[rout, rout], color=outer_sphere_color)

    plot_sphere(ax3d, rin, inner_sphere_color)
    plot_sphere(ax3d, rout, outer_sphere_color, alph=0.2)

    #plt.tight_layout()
    return fig

def sim_N_events(nmc, iso, iso_dict, sphere_dict, MC_dict):
    """ Function to simulate the alpha transport through a sphere
    """

    NUM_SRIM_TRAJ = 9999 ## number of simulated trajectories for each ion type
    alpha_mass = 4 ## mass of alpha particle in AMU

    ## construct sphere materials and parameters
    r_inner = sphere_dict['inner_radius']
    r_outer = r_inner + sphere_dict['outer_shell_thick']

    mat_inner = sphere_dict['inner_material']
    mat_outer = sphere_dict['shell_material']

    decay_dict = iso_dict[iso]

    output_record = {}

    for n in range(nmc):

        curr_iso = iso ## starting isotope at top of chain
        t = 0 ## start at time zero
        if("starting_loc" not in sphere_dict.keys()):
            x, y, z = random_point_in_sphere(r_inner) ## backwards compatibility
            curr_mat = mat_inner
        elif(sphere_dict["starting_loc"] == "core"):
            x, y, z = random_point_in_sphere(r_inner)
            curr_mat = mat_inner
        elif(sphere_dict["starting_loc"] == "shell"):
            x, y, z = random_point_on_surface(r_inner + sphere_dict['outer_shell_thick'] - 1) ## 1 nm from surface
            curr_mat = mat_outer
        else:
            print("Starting location not recognized")
            return -1 
    

        curr_t12 = decay_dict[curr_iso + "_t12"] ## get this to start the while loop

        event_record = {} ## hold details of the entire decay chain for this event
        decay_idx = 0

        event_record['parent'] = curr_iso
        event_record['start_pos'] = [x,y,z]

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

            if(decay_alpha_energy == 0): ## this is actually a beta decay, so we don't need to simulate anything
                ### update to the new isotope and t12
                curr_iso = decay_daughter
                curr_t12 = decay_dict[curr_iso + "_t12"]
                decay_record['traj'] = np.array([[0,x,y,z],])
                ## save the data
                event_record[decay_idx] = decay_record
                decay_idx += 1    
                continue   

            traj_dict = MC_dict[decay_daughter + '_' + curr_mat]

            ### get a random trajectory for its recoil
            traj_idx = np.random.choice(NUM_SRIM_TRAJ)+1
            curr_traj_full = traj_dict[traj_idx]
            init_xyz = [x,y,z]
            shortened_traj = select_end_of_traj(curr_traj_full,decay_NR_energy, False, init_xyz)

            ## essentially the alpha leaves with negligible momentum loss, eventually can add a real
            ## SRIM sim for this
            init_alpha_momentum_dir = -(shortened_traj[-1,1:4] - shortened_traj[-2,1:4])
            init_alpha_momentum_dir = init_alpha_momentum_dir/np.linalg.norm(init_alpha_momentum_dir)
            decay_record['alpha_momentum'] = np.sqrt(2 * decay_alpha_energy * alpha_mass) * init_alpha_momentum_dir

            traj_dict_inner = MC_dict[decay_daughter + '_' + mat_inner]
            traj_dict_outer = MC_dict[decay_daughter + '_' + mat_outer]
            ## follow trajectory until it stops
            final_traj, final_domain = follow_trajectory(shortened_traj, r_inner, r_outer, 
                                                         traj_dict_inner, traj_dict_outer, NUM_SRIM_TRAJ)

            ## shorten trajectory if it exited the sphere
            traj_rad = np.sqrt(final_traj[:, 1]**2 + final_traj[:, 2]**2 + final_traj[:, 3]**2)
            exit_idx = np.where(traj_rad > r_outer)[0]
            if(len(exit_idx)>0):
                final_traj = final_traj[:(exit_idx[0]+1), :]
                final_domain = 2 ## vacuum
                final_momentum = np.sqrt(2*daughter_mass*final_traj[-1,0]) ## in MeV
                final_momentum_dir = (final_traj[-1,1:4] - final_traj[-2,1:4])
                final_momentum_dir = final_momentum_dir / np.linalg.norm(final_momentum_dir)
                decay_record['NR_momentum'] = final_momentum*final_momentum_dir
                event_record['final_NR_momentum'] = final_momentum*final_momentum_dir
            else:
                decay_record['NR_momentum'] = np.array([0,0,0])               
                

            decay_record['traj'] = final_traj
            decay_record['final_domain'] = final_domain

            ## update position to end of trajectory
            x, y, z = final_traj[-1, 1], final_traj[-1, 2], final_traj[-1, 3]

            # update material to wherever we ended up
            if(np.sqrt(x**2 + y**2 + z**2) <= r_inner):
                curr_mat = mat_inner
            else:
                curr_mat = mat_outer

            ## save the data
            event_record[decay_idx] = decay_record
            decay_idx += 1

            ## stop if we've exited the sphere
            if(final_domain == 2):  
                break

            ### update to the new isotope and t12
            curr_iso = decay_daughter
            curr_t12 = decay_dict[curr_iso + "_t12"]

        event_record['final_pos'] = np.array([x,y,z])

        if('final_NR_momentum' not in event_record.keys()): ## never exited
            event_record['final_NR_momentum'] = np.array([0,0,0])
        output_record[n] = event_record

    return output_record


def analyze_simulation(sim_dict, sphere_dict=[]):
    """ Take a simulation dictionary and analyze it:
          1) Make a histogram of the distribution of final positions
          2) Calculate the fraction of escaped daughters vs time
    """

    #sphere_colors = {"SiO2": "gray", "Au": "gold", "Ag": "silver"}

    #rin = sd['inner_radius']
    #rout = sd['inner_radius'] + sd['outer_shell_thick']

    #inner_sphere_color = sphere_colors[sd["inner_material"]]
    #outer_sphere_color = sphere_colors[sd["shell_material"]]

    final_radii = []
    final_momentum = []
    num_bad_pts = 0
    N = len(sim_dict.keys())
    for i in range(N):
        curr_event = sim_dict[i]
        
        rad = np.sqrt(np.sum(curr_event['final_pos']**2))
        if(~np.isnan(rad)):
            final_radii.append(rad)
        else:
            num_bad_pts += 1

        #if(final_radii[-1]>20 and final_radii[-1]<21):
        #    print(i)

    print("Found %d bad points out of %d: %.3f%%"%(num_bad_pts,N,num_bad_pts/N*100))

    ## final radius distribution
    if(len(sphere_dict)>0):
        bins = np.arange(0,sphere_dict['inner_radius']+sphere_dict['outer_shell_thick'],2)
    else:
        bins = np.arange(0,500,2)
    final_rad_dist, be = np.histogram(final_radii, bins=bins)
    final_rad_bin_cents = be[:-1] + np.diff(be)/2

    ## escape fraction
    sdf = 1-np.cumsum(final_rad_dist)/np.sum(final_rad_dist)

    return final_rad_bin_cents, final_rad_dist, sdf

    #plt.figure(facecolor='white')
    #plt.plot(bc, h, 'k')
    #yy = plt.ylim()
    #plt.plot([rin, rin], yy, color=inner_sphere_color)
    #plt.plot([rout, rout], yy, color=outer_sphere_color)
    #plt.ylim(yy)
    #plt.xlabel("Radius [nm]")
    #plt.ylabel("Counts/[2 nm]")
    #plt.show()

def reconstruct_momenta(sim_dict, add_noise = 0, binsize=10):
    """ Take a simulation dictionary and analyze it:
          1) For each alpha decay, reconstruct the total momentum given to the sphere
          2) Separate this by isotope
    """

    isos_to_use = ["Tl-208", 'Pb-208']

    momentum_dict = {}
    for ciso in isos_to_use:
        momentum_dict[ciso] = []

    num_bad_pts = 0
    N = len(sim_dict.keys())
    for i in range(N):
        curr_event = sim_dict[i]

        for k in curr_event.keys():
            if(isinstance(k, str)): continue

            curr_iso = curr_event[k]['iso']

            if curr_iso in isos_to_use:

                if( 'alpha_momentum' not in curr_event[k].keys()):
                    if(curr_event[k]['traj'][0,0]> 0):
                        num_bad_pts += 1
                    continue

                p_alpha = curr_event[k]['alpha_momentum']
                p_NR = curr_event[k]['NR_momentum']
                tot_momentum = p_alpha + p_NR

                pt, pa, pn = np.linalg.norm(tot_momentum), np.linalg.norm(p_alpha), np.linalg.norm(p_NR)
                px = np.abs(tot_momentum[0])
                noise_vec = np.random.randn(4)*add_noise
                pt += noise_vec[0]
                pa += noise_vec[1]
                pn += noise_vec[2]
                px += noise_vec[3]

                momentum_dict[curr_iso].append([pt, pa, pn, px])

    print("Found %d bad points out of %d: %.3f%%"%(num_bad_pts,N,num_bad_pts/N*100))    

    pdf_fig = plt.figure(figsize=(15,4))
    cdf_fig = plt.figure(figsize=(6,4))
    x_fig = plt.figure(figsize=(6,4))
    bins = np.arange(0,350,binsize)

    tot_hist_dict = {}

    labs = ['Total momentum', r"$\alpha$ momentum", "NR momentum"]
    for col in range(3):
       
        tot_hist_dict[col] = np.zeros(len(bins)-1)
        for iso in isos_to_use:

            curr_moms = np.array(momentum_dict[iso])

            hh, be = np.histogram(curr_moms[:,col], bins=bins)
            bc = be[:-1] + np.diff(be)/2

            tot_hist_dict[col] += hh

            plt.figure(pdf_fig.number)
            plt.subplot(1,3,col+1)
            plt.plot(bc, hh, label=iso)
            plt.ylim(0,np.max(hh[bc>1])*1.2)
            plt.xlim(bins[0], bins[-1])

            if(col == 2):
                plt.figure(cdf_fig.number)
                plt.plot(bc, 1-np.cumsum(hh)/np.sum(hh), label=iso)


        plt.figure(pdf_fig.number)
        plt.plot(bc, tot_hist_dict[col], 'k', label='Total')
        plt.legend()
        plt.title(labs[col])
        plt.xlabel("Momentum [MeV]")
        plt.ylabel("Counts/(%d MeV)"%binsize)

    plt.figure(cdf_fig.number)
    plt.plot(bc, 1-np.cumsum(tot_hist_dict[2])/np.sum(tot_hist_dict[2]), 'k', label="Total")
    plt.xlabel("Momentum [MeV]")
    plt.ylabel("Survival function (1-CDF)")
    plt.ylim(0,1)
    plt.legend()
    plt.xlim(bins[0], bins[-1])
    plt.title("1 - CDF for NR momentum")

    plt.figure(x_fig.number)
    tot_hist = np.zeros(len(bins)-1)
    for iso in isos_to_use:

        curr_moms = np.array(momentum_dict[iso])

        hh, be = np.histogram(curr_moms[:,3], bins=bins)
        bc = be[:-1] + np.diff(be)/2
        tot_hist += hh

        plt.plot(bc, hh, label=iso)
        plt.xlim(bins[0], bins[-1])

    plt.plot(bc, tot_hist, 'k', label='Total')
    plt.ylim(0,np.max(tot_hist[bc>1])*1.2)
    plt.xlabel("$x$ Momentum [MeV]")
    plt.ylabel("Counts/(%d MeV)"%binsize)
    plt.legend()
    plt.title("Projected 1D momentum")

    plt.show()