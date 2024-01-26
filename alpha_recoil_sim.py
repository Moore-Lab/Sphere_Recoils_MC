## set of helper functions for simulating escape of nuclear recoils from silica spheres
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.special import gamma

## conversions of hours, days, years to seconds
seconds_dict = {"s": 1, "m": 60, "h": 3600, "d": 3600*24, "y": 3600*24*365.24} 
me = 511 #electron mass in keV
alpha = 1/137.036 ## fine structure constant
R0 = 1.03e6 ## nuclear radius [keV]

def parse_decay_chain(file):

    with open(file) as fin:
        lines = fin.readlines()

    decay_chain_dict = {}

    active_iso = ""
    for line in lines:
    
        if line[0] == "#": continue
    
        if(len(active_iso) == 0): ## case to start a new isotope
            lineparts = line.strip().split(',')
            iso, t12, alpha_q = lineparts[0], lineparts[1], lineparts[2]
            active_iso = iso
            
            t12_num = float(t12[:-1]) * seconds_dict[t12[-1]] ## convert all half lives to seconds

            decay_chain_dict[active_iso + "_t12"] = t12_num
        
            if(t12_num < 0): break ## quit if stable 
            
            decay_options = []
            decay_daughters = []
            decay_type = []

        else: ## case to assemble decay possibilities

            if( line.startswith("--") ): ## end of that isotope, restart
                ## add some extra probability if it's missing
                decay_options = np.array(decay_options)
                missing_prob = 1 - np.sum(decay_options[:,0]) ## get any small branches that were missing
                decay_options[-1,0] += missing_prob
                decay_chain_dict[active_iso + "_decays"] = decay_options
                decay_chain_dict[active_iso + "_daughters"] = decay_daughters
                decay_chain_dict[active_iso + "_type"] = decay_type
                active_iso = ""
            else: ## add another decay option
                parts = line.strip().split(',')
                decay_options.append( [float(parts[0]), float(parts[1])] )
                decay_daughters.append(parts[2].strip())
                decay_type.append(parts[3].strip())

    return decay_chain_dict
    
iso_Z_dict = {"Pb": 82, "Fr": 87, "At": 85, "Bi": 83, "Tl": 81, "Ra": 88, "Rn": 86, "Po": 84, "He": 2}

def get_Z_A_for_iso(iso):
    symbol = iso[:2]
    Z = iso_Z_dict[symbol]
    A = int(iso[3:])

    return Z, A

def fermi_func(A, Z, E):
  ## Z should be positive for beta minus decay
  gpts = E > 0
  f = np.zeros_like(E)
  R = R0 * A**(1/3)
  Et = E[gpts] + me
  p = np.sqrt( (Et)**2 - me**2 )
  eval = 2*(np.sqrt(1 - alpha**2 * Z**2)-1)
  gam = np.abs( gamma(np.sqrt(1 - alpha**2 * Z**2) + 1j* alpha * Z * Et/p ) )**2 / gamma(2*np.sqrt(1-alpha**2 * Z**2) + 1)**2
  f[gpts] = 2*(1 + np.sqrt(1-alpha**2 * Z**2) ) * (2*p*R)**eval * np.exp(np.pi*alpha*Z*Et/p) * gam
  return f

def simple_beta(E, Q, ms, A, Z):
  #return a simple spectrum of counts vs electron KE
  ## assumes E in keV
  ## Q is Q value in keV
  ## ms is nu mass in keV
  N = np.zeros_like(E)
  gpts = E < Q-ms
  N[gpts] = np.sqrt(E[gpts]**2 + 2*E[gpts]*me)*(E[gpts] + me)*np.sqrt((Q-E[gpts])**2 - ms**2)*(Q-E[gpts])
  ff = fermi_func(A, Z+1, E) ## Z+1 for the daughter 
  out = N*ff
  out[np.isnan(out)] = 0
  out[np.isinf(out)] = 0
  return out

def draw_from_pdf(n, pdf_x, pdf_p):
  ## function to draw n values from a PDF
  cdf = np.cumsum(pdf_p)
  cdf /= np.max(cdf)
  rv = np.random.rand(n)
  return np.interp(rv, cdf, pdf_x)

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
    elif( rad[0] <= rout ):
        cross_down = first_downward_crossing_time(rad, rin)
        cross_up = first_upward_crossing_time(rad, rout)
        
        if(cross_down and not cross_up):
            final_domain = 0
            cross = cross_down
        elif(cross_up and not cross_down):
            final_domain = 2
            cross = cross_up
        elif(cross_up and cross_down):
            if(cross_down < cross_up):
                final_domain = 0
                cross = cross_down
            else:
                final_domain = 2
                cross = cross_up
        else:
            final_domain = 1
            cross = None
    else:
        cross = first_downward_crossing_time(rad, rout)
        final_domain = 1

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

def follow_trajectory(traj, rin, rout, traj_dict_in, traj_dict_out, NUM_SRIM_TRAJ, traj_dict_ext=None):
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
        elif(final_domain == 2):
            dict_to_use = traj_dict_ext

        if(not dict_to_use): ## we're outside the sphere, and not interested in the trajectory
            break

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

def plot_circles(ax, rin, rout, cin, cout, lw=1):
    circle_in = plt.Circle((0, 0), rin, facecolor='none', edgecolor=cin, linestyle="-", lw=lw)
    circle_out = plt.Circle((0, 0), rout, facecolor='none', edgecolor=cout, linestyle="-", lw=lw)
    ax.add_patch(circle_in)
    ax.add_patch(circle_out)

def plot_sphere(ax, r, c, alph=0.5):
    u = np.linspace(0, 2 * np.pi, 100)
    #u = np.linspace(np.pi/4, 3*np.pi/4, 100)
    v = np.linspace(0, np.pi, 100)
    x = r*np.outer(np.cos(u), np.sin(v))
    y = r*np.outer(np.sin(u), np.sin(v))
    z = r*np.outer(np.ones(np.size(u)), np.cos(v))

    # plot sphere with transparency
    ax.plot_surface(x, y, z, alpha=alph, color=c)

def plot_event(event_dict, sd, rad_lims=[], sphere_coords=True, plot_alphas=False, plot_betas=False):

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

    xext, yext, zext  = [1e20,-1e20], [1e20,-1e20], [1e20,-1e20]
    for didx in range(len(decays)-4): ## four non numerical keys

        idx_for_colors = (didx + 1) ## starting isotope is 0

        data = event_dict[didx]['traj']
        if('traj_alpha' in event_dict[didx].keys()):
            alpha_data = event_dict[didx]['traj_alpha']
        else:
            alpha_data = None ## this was a beta or not saved

        if('traj_beta' in event_dict[didx].keys()):
            beta_data = event_dict[didx]['traj_beta']
        else:
            beta_data = None ## this was an alpha or not saved

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

            if(plot_betas and beta_data is not None):
                r_beta = np.sqrt(beta_data[:,1]**2 + beta_data[:,2]**2 + beta_data[:,3]**2)
                gpts = r_beta < 1e20*rout ## betas go straight so no issue with plotting
                ax3d.plot3D(beta_data[gpts,1], beta_data[gpts,2], beta_data[gpts,3], c=color_list[idx_for_colors], ls=':')

                if(sphere_coords):
                    for j,ax in enumerate(ax2d[:-1]):
                            ax.plot(beta_data[:,c1_list[j]], beta_data[:,c2_list[j]], c=color_list[idx_for_colors], ls=':')
            continue

        ## Plot the array in 3D
        ax3d.plot3D(data[:,1], data[:,2], data[:,3], c=color_list[idx_for_colors])
        ax3d.scatter(data[-1,1], data[-1,2], data[-1,3], 'o', c=color_list[idx_for_colors], label=event_dict[didx]['iso'])
        
        if(np.max(data[:,1]) > xext[1]): xext[1] = np.max(data[:,1])
        if(np.min(data[:,1]) < xext[0]): xext[0] = np.min(data[:,1])
        if(np.max(data[:,2]) > yext[1]): yext[1] = np.max(data[:,2])
        if(np.min(data[:,2]) < yext[0]): yext[0] = np.min(data[:,2])
        if(np.max(data[:,3]) > zext[1]): zext[1] = np.max(data[:,3])
        if(np.min(data[:,3]) < zext[0]): zext[0] = np.min(data[:,3])


        if(plot_alphas and alpha_data is not None):
            r_alpha = np.sqrt(alpha_data[:,1]**2 + alpha_data[:,2]**2 + alpha_data[:,3]**2)
            gpts = r_alpha < 10*rout ## some of the alphas eventually loop back through the 3D view
                                    ## and make the plots confusing
            ax3d.plot3D(alpha_data[gpts,1], alpha_data[gpts,2], alpha_data[gpts,3], c=color_list[idx_for_colors], ls='--')

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

        if(sphere_coords):
            for j,ax in enumerate(ax2d[:-1]):
                if(plot_alphas and alpha_data is not None):
                    ax.plot(alpha_data[:,c1_list[j]], alpha_data[:,c2_list[j]], c=color_list[idx_for_colors], ls='--')

    #ax3d.set_xlim(-rout*1.2, rout*1.2)
    #ax3d.set_ylim(-rout*1.2, rout*1.2)
    #ax3d.set_zlim(-rout*1.2, rout*1.2)
    ax3d.set_xlabel('x [nm]')
    ax3d.set_ylabel('y [nm]')
    ax3d.set_zlabel('z [nm]')
    ax3d.legend(bbox_to_anchor=(1.3, 0.5))

    lims = [xext, yext, zext]
    print(lims)
    if(sphere_coords):
        col_names = ['x', 'y', 'z']
        for j,ax in enumerate(ax2d[:-1]):
            c1, c2 = c1_list[j], c2_list[j]
            ax.set_xlabel(col_names[c1-1] + " [nm]")
            ax.set_ylabel(col_names[c2-1] + " [nm]")
            #ax.set_xlim(lims[c1-1][0]-10, lims[c1-1][1]+10 )
            #ax.set_ylim(lims[c2-1][0]-10, lims[c2-1][1]+10 )

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
    ax2d[-1].plot([0, xm],[rin, rin], color=inner_sphere_color)
    ax2d[-1].plot([0, xm],[rout, rout], color=outer_sphere_color)

    plot_sphere(ax3d, rin, inner_sphere_color)
    plot_sphere(ax3d, rout, outer_sphere_color, alph=0.2)

    #plt.tight_layout()
    return fig

def plot_event_row(event_dict_full, idx_list, sd, rad_lims=[], sphere_coords=True):

    # Create a 3D plot
    fig = plt.figure(figsize=(15,3*len(idx_list)), facecolor='white', layout='constrained')
    subfigs = fig.subfigures(len(idx_list), 2, wspace=0, hspace=0.05, width_ratios=[1, 2.8])
    print
    for j, idx in enumerate(idx_list):

        event_dict = event_dict_full[idx]

        ## single row plot for supplement
        color_list = ['k', 'r', 'b', 'g', 'c', 'm', 'y', 'orange', 'purple']
        c1_list = [1,2]
        c2_list = [2,3]

        sphere_colors = {"SiO2": "gray", "Au": "gold", "Ag": "silver"}

        rin = sd['inner_radius']
        rout = sd['inner_radius'] + sd['outer_shell_thick']

        inner_sphere_color = sphere_colors[sd["inner_material"]]
        outer_sphere_color = sphere_colors[sd["shell_material"]]

        ax3d = subfigs[j][0].add_subplot(1,1,1, projection='3d')
        ax2d = subfigs[j][1].subplots(1,3)
                
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
                ax.plot(xyz[c1_list[j]-1],xyz[c2_list[j]-1], 'o', color=color_list[0], label=event_dict['parent'])
        else:
            for j,ax in enumerate(ax2d[:-1]):
                ax.plot(0,xyz[j], 'o', color=color_list[0])
                
        ax2d[-1].plot(0,np.linalg.norm(xyz),'o', c=color_list[0], label=event_dict['parent'])   
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
                        ax.plot(data[-1,c1_list[j]], data[-1,c2_list[j]], 'o', ms=2, color=color_list[idx_for_colors], label=event_dict[didx]['iso'])
                else:
                    for j,ax in enumerate(ax2d[:-1]):
                        ax.plot(curr_lost_e[-1], data[-1,j+1], 'o', ms=2, color=color_list[idx_for_colors], label=event_dict[didx]['iso'])
                ax2d[-1].plot(curr_lost_e[-1], rad[-1], 'o', ms=2, color=color_list[idx_for_colors], label=event_dict[didx]['iso'])
                continue



            ## Plot the array in 3D
            ax3d.plot3D(data[:,1], data[:,2], data[:,3], c=color_list[idx_for_colors])
            ax3d.scatter(data[-1,1], data[-1,2], data[-1,3], 'o', c=color_list[idx_for_colors], label=event_dict[didx]['iso'])

            ## now plot the radius
            rad = np.sqrt( data[:,1]**2 + data[:,2]**2 + data[:,3]**2 )
            curr_lost_e = lost_e + (data[0,0]-data[:,0])
            ax2d[-1].plot(curr_lost_e, rad, '-', color=color_list[idx_for_colors])
            ax2d[-1].plot(curr_lost_e[-1], rad[-1], 'o', color=color_list[idx_for_colors], label=event_dict[didx]['iso'])
            lost_e = curr_lost_e[-1]

            if(sphere_coords):
                for j,ax in enumerate(ax2d[:-1]):
                    ax.plot(data[:,c1_list[j]], data[:,c2_list[j]], '-', color=color_list[idx_for_colors])
                    ax.plot(data[-1,c1_list[j]], data[-1,c2_list[j]], 'o', color=color_list[idx_for_colors], label=event_dict[didx]['iso'])
            else:
                for j,ax in enumerate(ax2d[:-1]):
                    ax.plot(curr_lost_e, data[:,j+1], '-', color=color_list[idx_for_colors])
                    ax.plot(curr_lost_e[-1], data[-1,j+1], 'o', color=color_list[idx_for_colors], label=event_dict[didx]['iso'])

        #ax3d.set_xlim(-rout*1.2, rout*1.2)
        #ax3d.set_ylim(-rout*1.2, rout*1.2)
        #ax3d.set_zlim(-rout*1.2, rout*1.2)
        ax3d.set_xlabel('x [nm]')
        ax3d.set_ylabel('y [nm]')
        ax3d.set_zlabel('z [nm]')
        ax2d[-1].legend() #bbox_to_anchor=(1.1, 0.75))

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
        #if(len(rad_lims)==0):
        #    ax2d[-1].set_ylim(0, rout*1.2)
        #    plt.text(xm*0.99, rin*1.01, sd["inner_material"], va='bottom', ha='right', color='k')
        #    plt.text(xm*0.99, rout*1.01, sd["shell_material"], va='bottom', ha='right', color='k')
        #else:
        #    if(rad_lims[0] > 0):
        #        ax2d[-1].set_ylim(rad_lims[0], rad_lims[1])
        #ax2d[-1].set_xlim(-0.01*xm, xm)
        #ax2d[-1].plot([0, xm],[rin, rin], color=inner_sphere_color)
        #ax2d[-1].plot([0, xm],[rout, rout], color=outer_sphere_color)

        #plot_sphere(ax3d, rin, inner_sphere_color)

        for ax in ax2d[:-1]:
            xx = ax.get_xlim()
            yy = ax.get_ylim()
            plot_circles(ax, 0, rout, 'gray', 'gray')
            ax.set_xlim(xx)
            ax.set_ylim(yy)

        xx = ax2d[-1].get_xlim()
        yy = ax2d[-1].get_ylim()
        plt.plot(xx, [rout, rout], color='gray')
        ax2d[-1].set_xlim(xx)
        ax2d[-1].set_ylim(yy)

    #xx = ax3d.get_xlim()
    #yy = ax3d.get_ylim()
    #zz = ax3d.get_zlim()
    #plot_sphere(ax3d, rout, outer_sphere_color, alph=0.02)
    #ax3d.set_xlim(xx)
    #ax3d.set_ylim(yy)
    #ax3d.set_zlim(zz)

    #plt.tight_layout()
    return fig

def sim_N_events(nmc, iso, iso_dict, sphere_dict, MC_dict, beta_dict={}, start_point=[], 
                 exterior_mat='vacuum', simulate_alpha=False, simulate_beta=False):
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
        
        if(n%int(1e6) == 0): print("MC iteration: ", n, " of ", nmc)

        curr_iso = iso ## starting isotope at top of chain
        t = 0 ## start at time zero
        if(len(start_point)>0):
            x, y, z = start_point[0], start_point[1], start_point[2]
            r = np.sqrt(x**2 + y**2 + z**2)
            if(r <= r_inner):
                curr_mat = mat_inner
            else:
                curr_mat = mat_outer
        else:
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
            curr_type_info = decay_dict[curr_iso + "_type"]

            rand_idx = np.random.choice(len(curr_daughter_info), p=curr_decay_info[:,0])

            decay_type = curr_type_info[rand_idx]
            daughter_mass = float(curr_iso.split("-")[-1])

            if(decay_type == 'alpha'):
                decay_alpha_energy = curr_decay_info[rand_idx,1] ## in keV
                decay_NR_energy = decay_alpha_energy * alpha_mass/daughter_mass
            elif(decay_type == 'beta'):
                decay_alpha_energy = 0 ## in keV
                decay_NR_energy = decay_alpha_energy * alpha_mass/daughter_mass
                decay_beta_Q = curr_decay_info[rand_idx,1] ## in keV

            decay_daughter = curr_daughter_info[rand_idx]
            decay_record['energy'] = decay_NR_energy
            decay_record['iso'] = decay_daughter

            init_xyz = [x,y,z]

            if(decay_type == 'beta'): ## this is actually a beta decay, so we won't simulate in detail
                ### update to the new isotope and t12
                curr_iso = decay_daughter
                curr_t12 = decay_dict[curr_iso + "_t12"]
                if(simulate_beta):
                    Z, A = get_Z_A_for_iso(curr_iso)
                    E = np.linspace(0,decay_beta_Q,1000)
                    spec = simple_beta(E, decay_beta_Q, 0, A, Z)
                    energy_beta = draw_from_pdf(1, E, spec)[0]
                    decay_record['energy_beta'] = energy_beta

                    traj_beta = [[energy_beta,x,y,z],]
                    psi, theta, phi = random_angle_on_sphere() ## euler angles, theta polar, phi azimuthal
                    dx, dy, dz = np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(theta)
                    beta_x, beta_y, beta_z = 1.0*x, 1.0*y, 1.0*z

                    curr_energy = 1.0*energy_beta
                    nsteps = 0
                    while(curr_energy > 0.001):
                        
                        curr_energy = traj_beta[-1][0]

                        curr_rad = np.sqrt(beta_x**2 + beta_y**2 + beta_z**2)
                        if(curr_rad < r_inner):
                            curr_mat = mat_inner
                            dd = r_inner/10 
                        elif(curr_rad < r_outer):
                            curr_mat = mat_outer
                            dd = (r_outer-r_inner)/10
                        else:
                            curr_mat = exterior_mat
                            dd = 10000 ## 10 um step outside

                        interp_func = beta_dict[curr_mat]

                        dE = interp_func(curr_energy)*dd
                        curr_energy -= dE
                        if(curr_energy < 0):
                            curr_energy = 0
                        beta_x, beta_y, beta_z = beta_x+dx*dd, beta_y+dy*dd, beta_z+dz*dd
                        traj_beta.append([curr_energy, beta_x, beta_y, beta_z])
                        nsteps += 1

                    decay_record['traj_beta'] = np.array(traj_beta)

                decay_record['traj'] = np.array([[0,x,y,z],])
                ## save the data
                event_record[decay_idx] = decay_record
                decay_idx += 1    
                continue   

            traj_dict = MC_dict[decay_daughter + '_' + curr_mat]

            ### get a random trajectory for its recoil
            traj_idx = np.random.choice(NUM_SRIM_TRAJ)+1
            curr_traj_full = traj_dict[traj_idx]
            shortened_traj = select_end_of_traj(curr_traj_full,decay_NR_energy, False, init_xyz)

            ## essentially the alpha leaves with negligible momentum loss, eventually can add a real
            ## SRIM sim for this
            init_alpha_momentum_dir = -(shortened_traj[1,1:4] - shortened_traj[0,1:4])
            init_alpha_momentum_dir = init_alpha_momentum_dir/np.linalg.norm(init_alpha_momentum_dir)
            decay_record['alpha_momentum'] = np.sqrt(2 * decay_alpha_energy * alpha_mass) * init_alpha_momentum_dir

            if(simulate_alpha):
                traj_dict_alpha = MC_dict['He-4_' + curr_mat]
                traj_idx_alpha = np.random.choice(NUM_SRIM_TRAJ)+1
                curr_traj_full_alpha = traj_dict_alpha[traj_idx_alpha]
                prior_traj_start = np.vstack((shortened_traj[0,:], shortened_traj[1,:]))
                shortened_traj_alpha = select_end_of_traj(curr_traj_full_alpha, decay_alpha_energy, True, init_xyz, prior_traj=-prior_traj_start)

                ## get the trajectory for the alpha
                traj_dict_inner_alpha = MC_dict['He-4_' + mat_inner]
                traj_dict_outer_alpha = MC_dict['He-4_' + mat_outer]
                if(exterior_mat != 'vacuum'):
                    traj_dict_ext_alpha = MC_dict['He-4_' + exterior_mat]
                else:
                    traj_dict_ext_alpha = None

                ## follow trajectory until it stops
                final_traj_alpha, final_domain_alpha = follow_trajectory(shortened_traj_alpha, r_inner, r_outer, 
                                                                            traj_dict_inner_alpha, traj_dict_outer_alpha, NUM_SRIM_TRAJ,
                                                                            traj_dict_ext=traj_dict_ext_alpha)
                decay_record['traj_alpha'] = final_traj_alpha
                decay_record['final_domain_alpha'] = final_domain_alpha
            else:
                decay_record['traj_alpha'] = None
                decay_record['final_domain_alpha'] = None


            traj_dict_inner = MC_dict[decay_daughter + '_' + mat_inner]
            traj_dict_outer = MC_dict[decay_daughter + '_' + mat_outer]
            if(exterior_mat != 'vacuum'):
                traj_dict_ext = MC_dict[decay_daughter + '_' + exterior_mat]
            else:
                traj_dict_ext = None

            ## follow trajectory until it stops
            final_traj, final_domain = follow_trajectory(shortened_traj, r_inner, r_outer, 
                                                         traj_dict_inner, traj_dict_outer, NUM_SRIM_TRAJ,
                                                         traj_dict_ext=traj_dict_ext)

            ## shorten trajectory if it exited the sphere (assuming we aren't simulating exterior)
            traj_rad = np.sqrt(final_traj[:, 1]**2 + final_traj[:, 2]**2 + final_traj[:, 3]**2)
            exit_idx = np.where(traj_rad > r_outer)[0]
            if(len(exit_idx)>0 and not traj_dict_ext):
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

def reconstruct_momenta(sim_dict, add_noise = {'x': [0], 'y': [0], 'z': 0}, binsize=5):
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
                coord_noise_vec = []
                for coord in ['x', 'y', 'z']:
                    curr_noise = np.random.choice(add_noise[coord])
                    coord_noise_vec.append(curr_noise)

                noise_vec = np.random.randn(3)*coord_noise_vec 
                tot_momentum = p_alpha + p_NR + noise_vec

                pt, pa, pn = np.linalg.norm(tot_momentum), np.linalg.norm(p_alpha), np.linalg.norm(p_NR)
                px = np.abs(tot_momentum[0])

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
            plt.ylim(0,np.max(hh[bc>50])*1.5)
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
    plt.ylim(0,np.max(tot_hist[bc>50])*1.1)
    plt.xlabel("$x$ Momentum [MeV]")
    plt.ylabel("Counts/(%d MeV)"%binsize)
    plt.legend()
    plt.title("Projected 1D momentum")

    plt.show()


def reconstruct_momenta_2panel(file_base, nfiles, add_noise = {'x': [0], 'y': [0], 'z': 0}, binsize=5):
    """ Take a simulation dictionary and analyze it:
          1) For each alpha decay, reconstruct the total momentum given to the sphere
          2) Separate this by isotope
    """

    isos_to_use = ["Tl-208", 'Pb-208']
    iso_labels = ["$^{212}$Bi", "$^{212}$Po"]
    colors = ['darkorange', 'orange']
    alphas = [0.6, 0.3]

    tag = 'Po-216_SiO2'

    momentum_dict = {}
    for ciso in isos_to_use:
        momentum_dict[ciso] = []

    num_bad_pts = 0

    for nf in range(nfiles):
        print("Loading file %d of %d"%(nf+1, nfiles))

        with open(file_base%nf, 'rb') as f:
            sim_dict = pickle.load(f)

        N = len(sim_dict[tag].keys())
        for i in range(int(N/1)):
            curr_event = sim_dict[tag][i]

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
                    coord_noise_vec = []
                    for coord in ['x', 'y', 'z']:
                        curr_noise = np.random.choice(add_noise[coord])
                        coord_noise_vec.append(curr_noise)

                    noise_vec = np.random.randn(3)*coord_noise_vec 

                    tot_momentum = p_alpha + p_NR + noise_vec

                    pt, pa, pn = np.linalg.norm(tot_momentum), np.linalg.norm(p_alpha), np.linalg.norm(p_NR)
                    px = np.abs(tot_momentum[0])

                    momentum_dict[curr_iso].append([pt, pa, pn, px])

    print("Found %d bad points out of %d: %.3f%%"%(num_bad_pts,N,num_bad_pts/N*100))    

    pdf_fig = plt.figure(figsize=(11,2.67*11/8))
    bins = np.arange(0,600,binsize)

    tot_hist_dict = 0

    norm_fac = 0
    for j,iso in enumerate(isos_to_use):
        curr_moms = np.array(momentum_dict[iso])
        norm_fac += len(curr_moms[:,0])

    for j,iso in enumerate(isos_to_use):

        curr_moms = np.array(momentum_dict[iso])

        hh, be = np.histogram(curr_moms[:,0], bins=bins)
        bsize = be[1]-be[0]
        hh = 1.0*hh/norm_fac ## density
        bc = be[:-1] + np.diff(be)/2

        tot_hist_dict += hh

        plt.figure(pdf_fig.number)
        plt.subplot(1,2,1)
        plt.plot(bc, hh/bsize, label=iso_labels[j], color=colors[j])
        plt.fill_between(bc, np.zeros_like(bc), hh/bsize, color=colors[j], alpha=0.4, edgecolor=None)
        plt.xlim(bins[0], bins[-1])


    plt.plot(bc, tot_hist_dict/bsize, 'k', label='Total')
    #plt.legend()
    plt.ylim(0,0.01)
    plt.xlim(0,500)
    plt.xlabel("Total momentum [MeV/c]")
    #plt.ylabel("Counts/(%d MeV/c)"%binsize)
    plt.ylabel("Probability density [(MeV/c)$^{-1}$]")

    plt.subplot(1,2,2)

    pdf_dat = []

    tot_hist = np.zeros(len(bins)-1)
    for j, iso in enumerate(isos_to_use):

        curr_moms = np.array(momentum_dict[iso])

        hh, be = np.histogram(curr_moms[:,3], bins=bins)
        bsize = be[1]-be[0]
        hh = 1.0*hh/norm_fac ## density
        bc = be[:-1] + np.diff(be)/2
        tot_hist += hh

        plt.plot(bc, hh/bsize, label=iso_labels[j], color=colors[j], alpha=1.5*alphas[j])
        plt.fill_between(bc, np.zeros_like(bc), hh/bsize, color=colors[j], alpha=alphas[j], edgecolor=None)
        plt.xlim(bins[0], bins[-1])
        pdf_dat.append(hh)

    plt.plot(bc, tot_hist/bsize, 'k', label='Total')
    #plt.ylim(0,np.max(tot_hist[bc>50])*1.1)
    plt.ylim(0,0.006)
    plt.xlim(0,500)
    plt.xlabel("Projected $x$ momentum [MeV/c]")
    #plt.ylabel("Counts/(%d MeV/c)"%binsize)
    plt.ylabel("Probability density [(MeV/c)$^{-1}$]", labelpad=0)
    plt.legend()
    #plt.title("Projected 1D momentum")

    return pdf_fig, bc, pdf_dat



def analyze_implantation(sim_dict, binsize=2, sphere_rad=1500):
    """ Take a simulation dictionary and analyze it:
          1) For each alpha decay, reconstruct the total momentum given to the sphere
          2) Separate this by isotope
    """

    isos_to_use = ["Pb-212", "Tl-208", 'Pb-208']

    momentum_dict = {}
    for ciso in isos_to_use:
        momentum_dict[ciso] = []

    num_bad_pts = 0
    N = len(sim_dict.keys())

    implant_rad = {}
    implant_distance = []
    for iso in isos_to_use:
        implant_rad[iso] = []

    plt.figure(figsize=(10,3.5))
    plt.subplot(1,2,1)

    for i in range(N):
        curr_event = sim_dict[i]

        for k in curr_event.keys():
            if(isinstance(k, str)): continue

            curr_iso = curr_event[k]['iso']
            if(curr_iso not in isos_to_use): continue
        
            implant_rad[curr_iso].append( np.sqrt(np.sum(curr_event[k]['traj'][-1, 1:]**2)) )

            if(curr_iso == "Pb-212"):
                dist = np.sqrt( np.sum( (curr_event[k]['traj'][-1, 1:] - curr_event['start_pos'])**2) )
                implant_distance.append(dist)

                if((i < 1000)):
                    if(implant_rad[curr_iso][-1] < sphere_rad):
                        curr_traj = curr_event[k]['traj'][:,1:] # - curr_event[k]['traj'][0,1:] + np.array([3000, 0, 0])
                        plt.plot(curr_traj[:,0], curr_traj[:,1], 'r', lw=1, alpha=0.1, rasterized=True)
                        plt.plot(curr_traj[-1,0], curr_traj[-1,1], 'ro', ms=2, alpha=0.1)
                    else:
                        curr_traj = curr_event[k]['traj'][:,1:] # - curr_event[k]['traj'][0,1:] + np.array([3000, 0, 0])
                        plt.plot(curr_traj[:-1,0], curr_traj[:-1,1], 'r', lw=1, alpha=0.1, rasterized=True)
                        lin_traj = curr_traj[-1,:] - curr_traj[-2,:]
                        lin_traj_lengthened = 1000*lin_traj
                        lin_traj_lengthened += curr_traj[-2,:]
                        lin_traj_lengthened = np.vstack((curr_traj[-2,:], lin_traj_lengthened))
                        plt.plot(lin_traj_lengthened[:,0], lin_traj_lengthened[:,1], 'r', lw=1, alpha=0.1, rasterized=True)
                    

    ax = plt.gca()
    #plot_circles(ax, 0, sphere_rad, 'k', 'k', lw=2)
    #plt.plot([3000, 3000], [-200, 200])
    circle_out = plt.Circle((0, 0), sphere_rad, facecolor='gray', edgecolor='k', linestyle="-", lw=2, alpha=0.2)
    ax.add_patch(circle_out)
    plt.plot(sphere_rad, 0, 'ko', ms=2)
    plt.xlim(sphere_rad-150, sphere_rad+50)
    plt.ylim(-100, 100)
    plt.xlabel("x [nm]")
    plt.ylabel("y [nm]")
    plt.gca().set_aspect('equal')

    plt.subplot(1,2,2)
    implant_r = np.array(implant_rad["Pb-212"])
    implant_distance = np.array(implant_distance)
    implant_cut = implant_r < sphere_rad
    eff = np.sum(implant_cut)/N
    print("Implantation efficiency: ", eff)

    bins = np.arange(-10, 200, binsize) #sphere_rad-150,sphere_rad+10,binsize)
    hh, be = np.histogram(sphere_rad - implant_r[implant_cut], bins=bins)
    depth = sphere_rad - implant_r[implant_cut]
    print("median: ", np.median(depth), np.percentile(depth, 95))
    bc = be[:-1] + np.diff(be)/2
    plt.step(bc, hh, 'k', where='mid', label="Radial position")

    hhd, bed = np.histogram(implant_distance[implant_cut], bins=bins)
    bcd = bed[:-1] + np.diff(bed)/2
    plt.step(bcd, hhd, 'tab:orange', where='mid', label="Cartesian distance")
    #plt.legend()
    plt.xlabel("Distance [nm]")
    plt.ylabel("Counts/(%d nm)"%binsize)    
    plt.xlim(0, 120)
    yy = plt.ylim()
    plt.ylim(0, yy[1])
    #plt.title("Implantation distribution, efficiency = %.3f"%eff)
    plt.savefig("implantation_sim.pdf", bbox_inches='tight')

    plt.figure()
    bins = np.arange(sphere_rad-200,sphere_rad+10,binsize)
    for iso in isos_to_use:
        
        implant_r = np.array(implant_rad[iso])
        implant_cut = implant_r < sphere_rad
        hh, be = np.histogram(implant_r[implant_cut], bins=bins)
        #implant_cut = implant_rad < sphere_rad
        bc = be[:-1] + np.diff(be)/2
        plt.plot(bc, hh, label=iso)

    plt.legend()
    plt.xlabel("Distance from center [nm]")
    plt.ylabel("Counts/(%d nm)"%binsize)  
    plt.show()

def parse_transrec_file(daughtfile, recfile):

    ## parse the transmitted NR for the daughter of the decay (daughtfile)
    ## and the secondary recoils (recfile)

    A_to_nm = 0.1 #convert angstrom to nm   
    u_to_GeV = 0.931 # convert amu to GeV
    eV_to_keV = 1e-3 # convert eV to keV

    daught_dict = {}
    ## first find the transmitted daughters
    with open(daughtfile, 'r') as f:
        daughtlines = f.readlines()

    for l in daughtlines:

        if not l.startswith("T"): continue 

        curr_dat = l.strip().split()
        if(curr_dat[0] != "T"): 
            curr_dat.insert(1, curr_dat[0][1:]) # T runs into event id for large ids
        line_event = int(curr_dat[1])
    
        if(len(curr_dat) < 10): 
            print("Bad line: ", line_event)
            continue

        Z = int(curr_dat[2])
        if( Z == 81):
            A = 208 * u_to_GeV 
        elif( Z == 82):
            A = 208 * u_to_GeV
        else:
            print("Unknown Z: ", Z)
            return
        
        E = float(curr_dat[3]) * eV_to_keV
        p = np.sqrt(2 * E * A) ## in MeV

        daught_dict[line_event] = [Z, A, p, float(curr_dat[4])*A_to_nm, float(curr_dat[5])*A_to_nm, float(curr_dat[6])*A_to_nm,
                                            float(curr_dat[7]), float(curr_dat[8]), float(curr_dat[9])]

    with open(recfile, 'r') as f:
        reclines = f.readlines()

    event_dict = {}
    curr_event = -1
    for l in reclines:

        if not l.startswith("0"): continue 

        curr_dat = l.strip().split()
        line_event = int(curr_dat[0])

        if(line_event not in event_dict.keys()):
            event_dict[line_event] = {}
        
        if(len(curr_dat) == 7):

            ## fix rare issue with SRIM files
            try:
                test = [float(curr_dat[1]), float(curr_dat[2]), float(curr_dat[3]), float(curr_dat[4])]
            except:
                continue

            ## this is the ion trajectory, so save into the traj
            recoil_data = [float(curr_dat[1]), float(curr_dat[2])*A_to_nm, 
                           float(curr_dat[3])*A_to_nm, float(curr_dat[4])*A_to_nm]
            
            if( abs(line_event - curr_event) < 0.1):
                event_dict[line_event]['traj'].append(recoil_data)
            else:
                curr_event = line_event
                event_dict[line_event]['traj'] = [recoil_data,]

        elif(len(curr_dat) == 9):
            ## this is a transmitted ion

            Z = int(curr_dat[1])
            if( Z == 14):
                A = 28.0855 * u_to_GeV
            elif( Z == 8):
                A = 15.999 * u_to_GeV
            else:
                print("Unknown Z: ", Z)
                return
            
            E = float(curr_dat[2]) * eV_to_keV
            p = np.sqrt(2 * E * A) ## in MeV

            recoil_data = [Z, A, p, float(curr_dat[3])*A_to_nm, float(curr_dat[4])*A_to_nm, float(curr_dat[5])*A_to_nm,
                                    float(curr_dat[6]), float(curr_dat[7]), float(curr_dat[8])]

            if( abs(line_event - curr_event) < 0.1 and 'recoil' in event_dict[line_event].keys()):
                event_dict[line_event]['recoil'].append(recoil_data)
            else:
                if( line_event % 10000 == 0 ): 
                    print("Working on event %d"%(line_event))
                curr_event = line_event
                event_dict[line_event]['recoil'] = [recoil_data,]

    return daught_dict, event_dict

def analyze_trans_data(daught_dict, recoil_dict):
    ## analyze the data regarding transmitted daughters and secondary recoils

    ## first find the distribution of number
    last_event = list(recoil_dict.keys())[-1]
    si_recoils = []
    o_recoils = []

    daughter_momentum = []
    secondary_momentum = []
    daughter_alpha_mom = []
    total_momentum = []

    for k in range(1,last_event+1):

        traj = np.array(recoil_dict[k]['traj'])
        if(np.shape(traj)[0] < 2): continue
        alpha_vec = traj[0,1:]-traj[1,1:]
        alpha_vec = alpha_vec/np.linalg.norm(alpha_vec)
        p_alpha = 265*alpha_vec

        if('recoil' not in recoil_dict[k].keys()): 
            si_recoils.append(0)
            o_recoils.append(0)

            if(k in daught_dict.keys()):
                daughter_momentum.append(daught_dict[k][2])
                total_momentum.append(daught_dict[k][2])
            else:
                daughter_momentum.append(0)
                total_momentum.append(0)

            secondary_momentum.append(0)

        else:
            recoils = np.array(recoil_dict[k]['recoil'])
            n_si = np.sum(recoils[:,0] == 14)
            n_o = np.sum(recoils[:,0] == 8)
            si_recoils.append(n_si)
            o_recoils.append(n_o)

            if(k in daught_dict.keys()):
                dm = np.array(daught_dict[k][6:])*daught_dict[k][2]
            else:
                dm = np.array([0.0, 0.0, 0.0])

            sm = np.array([0.0, 0.0, 0.0])
            for recoil in recoils:
                p_recoil = recoil[2]
                recoil_vec = np.array(recoil[6:])
                sm += p_recoil*recoil_vec

            daughter_momentum.append(np.linalg.norm(dm))
            secondary_momentum.append(np.linalg.norm(sm))
            total_momentum.append(np.linalg.norm(dm + sm + p_alpha))
            daughter_alpha_mom.append(np.linalg.norm(dm + p_alpha))
    
    hsi, besi = np.histogram(si_recoils, bins=np.arange(-0.5, 40.5, 1))
    ho, beo = np.histogram(o_recoils, bins=np.arange(-0.5, 40.5, 1))

    hsi = hsi/last_event
    ho = ho/last_event

    plt.figure()
    plt.step(besi[:-1], hsi, 'tab:red', where='post', label="Si")
    plt.step(beo[:-1], ho, 'tab:orange', where='post', label="O")
    plt.xlim(-0.5,40)
    plt.gca().set_yscale('log')
    plt.legend()
    plt.xlabel("Number of ejected secondary recoils")
    plt.ylabel("Fraction of events")

    bin_size = 10
    pmax = 500
    bins = np.arange(0,pmax,bin_size)
    hd, bed = np.histogram(daughter_momentum, bins=bins)
    hs, bes = np.histogram(secondary_momentum, bins=bins)
    hda, beda = np.histogram(daughter_alpha_mom, bins=bins)
    ht, bet = np.histogram(total_momentum, bins=bins)

    plt.figure()
    plt.step(bed[:-1], hd/last_event, 'b', where='post', label="Daughter")
    plt.step(beda[:-1], hda/last_event, 'b:', where='post', label="Daughter + $\\alpha$")
    plt.step(bes[:-1], hs/last_event, 'r', where='post', label="Secondary")
    plt.step(bet[:-1], ht/last_event, 'k', where='post', label="Total")
    plt.xlim(0,pmax)
    #plt.gca().set_yscale('log')
    plt.legend()
    plt.xlabel("Momentum from ejected nuclei [MeV]")
    plt.ylabel("Fraction of events/(%d MeV)"%bin_size)


def plot_transmitted(daught_dict, recoil_dict, event_num, sphere_rad=1500, sim_thick=30):

    iso_dict = {14: ["Si", 'tab:red'], 8: ["O", 'tab:orange']}

    if(event_num in daught_dict.keys()):
        daught_dat = daught_dict[event_num]
    else:
        daught_dat = np.zeros(9)

    recoils = np.array(recoil_dict[event_num]['recoil'])
    traj = np.array(recoil_dict[event_num]['traj'])

    fig = plt.figure()

    norm = 1 ## for plotting

    traj[:,1] += sphere_rad-sim_thick
    #print(traj)
    plt.plot(traj[:,1], traj[:,2], 'k', lw=1)


    if(np.shape(traj)[0] > 1): 
        alpha_vec = traj[0,1:]-traj[1,1:]
        alpha_vec = alpha_vec/np.linalg.norm(alpha_vec)
        p_alpha = 265

        plt.arrow(traj[0][1], traj[0][2], p_alpha*alpha_vec[0]*norm, p_alpha*alpha_vec[1]*norm, 
              width=0.5, head_width=3, color='gray')



    ptot = np.array([0.0,0.0,0.0])

    ## now do the secondaries
    for recoil in recoils:
        Z = recoil[0]
        p_recoil = recoil[2]
        daught_pos = np.array(recoil[3:6])
        daught_pos[0] += sphere_rad-sim_thick
        daught_vec = np.array(recoil[6:])
        plt.arrow(daught_pos[0], daught_pos[1], daught_vec[0]*p_recoil*norm, daught_vec[1]*p_recoil*norm, width=0.5, head_width=3, color=iso_dict[Z][1])
        #print(recoil)

        ptot += p_recoil*daught_vec

    p_daught = daught_dat[2]
    daught_pos = np.array(daught_dat[3:6])
    daught_pos[0] += sphere_rad-sim_thick
    daught_vec = np.array(daught_dat[6:])

    plt.arrow(daught_pos[0], daught_pos[1], daught_vec[0]*p_daught*norm, daught_vec[1]*p_daught*norm, 
              width=0.5, head_width=3, color='k', label=r"$|\vec{p}| = %.1f$ MeV"%p_daught)

    alpha_recoil_mom_x = p_alpha*alpha_vec[0] + daught_vec[0]*p_daught
    alpha_recoil_mom_y = p_alpha*alpha_vec[1] + daught_vec[1]*p_daught
    alpha_recoil_mom_z = p_alpha*alpha_vec[2] + daught_vec[2]*p_daught
    plt.arrow(traj[0][1], traj[0][2], alpha_recoil_mom_x*norm, alpha_recoil_mom_y*norm,
              width=0.5, head_width=3, color='b')
    
    tot_mom_x = ptot[0] + alpha_recoil_mom_x
    tot_mom_y = ptot[1] + alpha_recoil_mom_y
    tot_mom_z = ptot[2] + alpha_recoil_mom_z
    ptot_mag = np.sqrt(tot_mom_x**2 + tot_mom_y**2 + tot_mom_z**2)
    plt.arrow(traj[0][1], traj[0][2], tot_mom_x*norm, tot_mom_y*norm,
              width=0.5, head_width=3, color='g', label=r"$|\vec{p}| = %.1f$ MeV"%ptot_mag)

    ax = plt.gca()
    circle_out = plt.Circle((0, 0), sphere_rad, facecolor='gray', edgecolor='k', linestyle="-", lw=2, alpha=0.2)
    ax.add_patch(circle_out)

    plt.xlim(sphere_rad-150, sphere_rad+150)
    plt.ylim(-100, 100)
    plt.xlabel("x [nm]")
    plt.ylabel("y [nm]")
    plt.gca().set_aspect('equal')
    plt.title("Total momentum = %.1f MeV"%np.linalg.norm(ptot + daught_vec*p_daught))
    plt.legend()

    return fig
        