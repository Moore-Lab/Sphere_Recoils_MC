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
            