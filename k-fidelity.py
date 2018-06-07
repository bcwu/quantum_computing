import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D
import colorsys

def gen_wave_from_file(filename):
    keyword = 'x=['
    skip = 0
    with open(filename) as f:
        for n, line in enumerate(f):
            if keyword in line:
                skip = n+1
                break    
    df00 = pd.read_csv(filename, skiprows=skip, header=None, skipfooter=2, engine='python', sep=' ')
    df0 = df00.as_matrix(columns=[0,1])
    psi = df0[...,0] + 1j * df0[...,1]
    return psi

def density_matrix(psi):
    psi = psi.reshape(psi.shape[0], 1)
    psiT = psi.transpose()
    psi_adjoint = psiT.conjugate()
    dm = np.matmul(psi, psi_adjoint)
    return dm

def fidelity(rho, delta):
    assert(rho.shape == delta.shape)
    F = np.trace(np.matmul(rho,delta)) + 2 * np.sqrt(np.linalg.det(rho) * np.linalg.det(delta))
    if F > 1:
        print('WARNING: Fidelity is greater than one!')
        rho_det = np.linalg.det(rho)
        if rho_det > 1:
            print('WARNING: rho determinant (reference_dm) is greater than 1: {}'.format(rho_det))
        delta_det = np.linalg.det(delta)
        if delta_det > 1:
            print('WARNING: delta determinant (avg_dm) is greater than 1: {}'.format(delta_det))
    return F

def gen_dm(Intel_QS_interface, qasm_dir, qasm_name, output_dir, i, T1, T2, save=False):
    output_name = output_dir + qasm_name + '_T1={}_T2={}_out{}.txt'.format(T1, T2, i)     
    cmd = '{} {} {} {} < {} > {} '.format(Intel_QS_interface, i, T1, T2, qasm_dir + qasm_name, output_name)      
    os.system(cmd)
    psi = gen_wave_from_file(output_name)
    dm = density_matrix(psi)
    if save is False:
        cmd = 'rm {}'.format(output_name)
        os.system(cmd)   
    return dm

def gen_avg_dm(Intel_QS_interface, qasm_dir, qasm_name, output_dir, K, T1, T2):
    sum_dm = None
    for i in range(1, K+1):
        dm = gen_dm(Intel_QS_interface, qasm_dir, qasm_name, output_dir, i, T1, T2)
        if sum_dm is None:
            sum_dm = dm
        else:
            sum_dm += dm
    avg_dm = sum_dm / K
    return avg_dm

def k_sample_fidelity(Intel_QS_interface, reference_dm, qasm_dir, qasm_name, output_dir, K, T1, T2, save=False):
    sum_dm = None
    fidelity_list = []
    for i in range(1, K+1):
        output_name = output_dir + qasm_name + '_T1={}_T2={}_out{}.txt'.format(T1, T2, i) 
        cmd = '{} {} {} {} < {} > {} '.format(Intel_QS_interface, i, T1, T2, qasm_dir + qasm_name, output_name)        
        os.system(cmd)
        psi = gen_wave_from_file(output_name)
        if save is False:
            cmd = 'rm {}'.format(output_name)
            os.system(cmd)                
        dm = density_matrix(psi)
        if sum_dm is None:
            sum_dm = dm
        else:
            sum_dm += dm
        avg_dm = sum_dm / i        
        fidelity_list.append(fidelity(reference_dm, avg_dm))
    print(avg_dm)
    return fidelity_list

def plot_3d(x, y, z, K, T1, T2, graphs_dir, title_prefix=None):
    print('Processing summary graph')
    N = max(x)
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    color_d = {}
    for i, c in enumerate(RGB_tuples):        
        color_d[i+1] = c        
    colors = []
    for i in x:
        colors.append(color_d[i])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=x, ys=y , zs=z, c=colors, label='fidelity')
    ax.set_xlabel('qubits')
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    ax.set_ylabel('gates')
    ax.set_zlabel('fidelity')
    graph_suffix = '_{}-sample_fidelity_T1={}_T2={}'.format(K, T1, T2)
    if title_prefix:        
        graph_suffix = title_prefix + graph_suffix + '.png'
    plt.savefig(graphs_dir + 'Summary_' + graph_suffix)
    plt.title('Summary_' + graph_suffix)
    plt.close()
    print('Finished processing summary graph\n')

def compare_fidelity(Intel_QS_interface, reference_dm, qasm_dir, qasm_name, graphs_dir, output_dir, K, T1, T2, title, suffix, save=False):    
    print('Comparing fidelity with {} runs of Intel QS at T1={} T2={}'.format(K, T1, T2))
    f = k_sample_fidelity(Intel_QS_interface, reference_dm, qasm_dir, qasm_name, output_dir, K, T1, T2, save=save)
    plt.plot([i for i in range(1, K+1)], f)
    plt.xlabel('k-th cumulative average dm')
    plt.ylabel('Fidelity')
    plt.title('{}'.format(title))
    graph_suffix = '_{}-sample_fidelity_T1={}_T2={}_{}.png'.format(K, T1, T2, suffix)
    plt.savefig(graphs_dir + qasm_name + graph_suffix)
    plt.close()
    print('Finished comparing fidelity with {} runs of Intel QS'.format(K))
    x = []
    j = 0
    for i, c in enumerate(qasm_name):
        if not c.isnumeric():
            j = i
            break
        x.append(c)
    x = int(''.join(x))

    y = []
    for c in qasm_name[j:]:
        if c.isnumeric():
            y.append(c)
    y = int(''.join(y))      

    return f[-1].real, x, y

def add_QuaC_noise(qasm_name, T1, T2):
    # x = int(qasm_name[0])
    x = []
    for c in qasm_name:
        if not c.isnumeric():
            break
        x.append(c)
    x = int(''.join(x))
    t1, t2 = 1/T1, 1/T2
    noise_param = []
    for i in range(x):
        noise_param.append('-gam{} {} -dep{} {}'.format(i, t1, i, t2))
    return ' '.join(noise_param)

def QuaC_wrapper(QuaC_interface, qasm_dir, qasm_name, output_dir, T1, T2, save=False):
    noise_param = add_QuaC_noise(qasm_name, T1, T2)
    output_name = output_dir + qasm_name + '_T1={}_T2={}.log'.format(T1, T2) 
    cmd = '{} -file_circ {} {} > {}'.format(QuaC_interface, qasm_dir + qasm_name, noise_param, output_name)    
    # cmd = '~/Quantum_Computing/QuaC/projectq_simple_circuit -file_circ {} {} > {}'.format(qasm_dir + qasm_name, noise_param, output_name)    
    os.system(cmd)  
    QuaC_dm = extract_QuaC_dm(output_name)
    if save is False:
        cmd = 'rm {}'.format(output_name)
        os.system(cmd)
    return QuaC_dm

def extract_QuaC_dm(filename):
    keyword = 'Steps'
    skip = 0
    with open(filename) as f:
        for n, line in enumerate(f):
            if keyword in line:
                skip = n+1
                break
    df00 = pd.read_csv(filename, skiprows=skip, header=None, skipfooter=0, engine='python', sep=' ')
    df01 = df00.dropna(axis='columns')
    df02 = df01.values
    df02 = df02.astype(str).astype(complex)
    return df02

def frontend():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '-?', '--help', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit.')
    parser.add_argument('-K', help='run the noisy Intel_QS simulation k number of times')
    parser.add_argument('-T1', help='T1 amplitude damping')
    parser.add_argument('-T2', help='T2 dephasing')    
    parser.add_argument('-Intel_QS_interface', type=str, help='Directory of qasm_interface.exe', 
                            # required=True,
                            default='/home/xp3js2v/Intel-QS/interface/qasm_interface.exe')
    parser.add_argument('-QuaC_interface', type=str, help='Directory of projectq_simple_circuit. \
                            If left empty, the program will compare noisy Intel-QS simulations against a non-noisy Intel-QS reference simulation', 
                            required=False,
                            default='/home/xp3js2v/QuaC/projectq_simple_circuit')
    parser.add_argument('-Intel_QS_qasm_dir', type=str, help='Directory of Intel-QS qasms', 
                            # required=True,                                        
                            default='/home/xp3js2v/QASMs/Intel_QS_qasm/')
    parser.add_argument('-QuaC_qasm_dir', type=str, help='Directory of QuaC qasms, \n \
                            e.g. QASMs/QuaC_qasm/ \n \
                            If left empty, the program will compare noisy Intel-QS simulations against a non-noisy Intel-QS reference simulation', 
                            required=False,
                            default='/home/xp3js2v/QASMs/QuaC_qasm/')
    parser.add_argument('-graphs_dir', type=str, help='directory for graphs', 
                            # required=True,
                            default='/home/xp3js2v/graphs/') 
    parser.add_argument('-output_dir', type=str, help='directory for output', 
                            # required=True,
                            default='/home/xp3js2v/output/')                                             
    args = parser.parse_args()  
    return args 

def main():
    args = frontend()
    # print(args)
    Intel_QS_interface = args.Intel_QS_interface
    QuaC_interface = args.QuaC_interface
    Intel_QS_qasm_dir = args.Intel_QS_qasm_dir
    QuaC_qasm_dir = args.QuaC_qasm_dir
    graphs_dir = args.graphs_dir
    output_dir = args.output_dir
    K = int(args.K)
    T1s = list(map(float, args.T1.split(',')))
    T2s = list(map(float, args.T2.split(',')))
    Intel_QS_qasms = os.listdir(args.Intel_QS_qasm_dir) 
    x, y, z = [], [], []   
    if QuaC_qasm_dir == 'None' or QuaC_interface == 'None':
        print('Comparing noisy Intel-QS simulation against non-noisy Intel-QS reference simulation')
        for T1 in T1s:
            for T2 in T2s:        
                for Intel_QS_qasm_name in Intel_QS_qasms:
                    print('Processing {} as reference dm'.format(Intel_QS_qasm_name))
                    reference_dm = gen_dm(Intel_QS_interface, Intel_QS_qasm_dir, Intel_QS_qasm_name, output_dir, i=1, T1=1e16, T2=1e16, save=False)
                    print(reference_dm)
                    print('Processing {}'.format(Intel_QS_qasm_name))
                    title_name = 'Noisy vs Non-Noisy Intel-QS ' + Intel_QS_qasm_name[:-6] + ' K={} T1={} T2={}'.format(K, T1, T2) 
                    fidelity, qubits, gates = compare_fidelity(Intel_QS_interface, reference_dm, Intel_QS_qasm_dir, Intel_QS_qasm_name, graphs_dir, output_dir, K, T1, T2, save=False, title=title_name, suffix='Intel-QS_only')        
                    x.append(qubits)
                    y.append(gates)
                    z.append(fidelity)                              
                plot_3d(x, y, z, K, T1, T2, graphs_dir, 'Intel-QS_only')
                x, y, z = [], [], [] 
        return
    QuaC_qasms = os.listdir(args.QuaC_qasm_dir) 
    for T1 in T1s:
        for T2 in T2s:        
            for QuaC_qasm_name in QuaC_qasms:
                print('Processing {}'.format(QuaC_qasm_name))
                reference_dm = QuaC_wrapper(QuaC_interface, QuaC_qasm_dir,  QuaC_qasm_name, output_dir, T1, T2, save=True)
                print(reference_dm)
                Intel_QS_qasm_name = QuaC_qasm_name[:-5] + '.qasmf'
                print('Processing {}'.format(Intel_QS_qasm_name))
                title_name = 'QuaC vs Intel-QS ' + Intel_QS_qasm_name[:-6] + ' k={} T1={} T2={}'.format(K, T1, T2) 
                fidelity, qubits, gates = compare_fidelity(Intel_QS_interface, reference_dm, Intel_QS_qasm_dir, Intel_QS_qasm_name, 
                                                graphs_dir, output_dir, K, T1, T2, save=False, 
                                                title=title_name, suffix='QuaC_vs_Intel-QS')        
                x.append(qubits)
                y.append(gates)
                z.append(fidelity)
            plot_3d(x, y, z, K, T1, T2, graphs_dir, 'QuaC_vs_Intel-QS')
            x, y, z = [], [], [] 


if __name__ == '__main__':
    main()