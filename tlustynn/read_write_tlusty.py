import numpy as np
import pandas as pd


def write_tlusty_model7(filename, model_data):
    df = model_data['dataframe']
    n_depth = model_data['n_depth']
    n_params = model_data['n_params']
    
    with open(filename, 'w') as f:
        f.write(f"   {n_depth:3d}   {n_params:3d}\n")
        tau_values = df['tau'].values
        for i in range(0, n_depth, 6):
            line_values = tau_values[i:i+6]
            line_str = "".join(f" {val:13.6E}" for val in line_values)
            f.write(line_str + "\n")
        
        for depth_idx in range(n_depth):
            row_data = df.iloc[depth_idx, 2:].values 
            
            for i in range(0, n_params, 6):
                line_values = row_data[i:i+6]
                line_str = "  " + "".join(f" {val:13.6E}" for val in line_values)
                f.write(line_str + "\n")

def read_tlusty_model7(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    first_line = lines[0].split()
    n_depth = int(first_line[0])  
    n_params = int(first_line[1])  
        
    all_data = []
    for line in lines[1:]:
        if line.strip():  
            line_data = [float(x) for x in line.split()]
            all_data.extend(line_data)
    
    all_data = np.array(all_data)
    
    tau_values = all_data[:n_depth]
    
    param_data = all_data[n_depth:]
    
    parameters = param_data.reshape(n_depth, n_params)
    
    column_names = ['T', 'ne', 'rho']  
    n_levels = n_params - 3  
    
    for i in range(n_levels):
        column_names.append(f'level_{i+1}')
    
    df = pd.DataFrame(parameters, columns=column_names)
    
    df.insert(0, 'tau', tau_values)
    df.insert(0, 'depth_index', range(1, n_depth + 1))
    
    return {
        'n_depth': n_depth,
        'n_params': n_params,
        'n_levels': n_levels,
        'tau': tau_values,
        'parameters': parameters,
        'dataframe': df,
        'column_names': column_names}
