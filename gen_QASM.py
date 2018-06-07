import random

k_qubits = 5
n_gates = [(2**n) for n in range(11)]
circuits = ['H', 'X']
Intel_QS_QASM_dir = 'QASMs/Intel_QS_qasm/'
QuaC_QASM_dir = 'QASMs/QuaC_qasm/'

for k in range(1, k_qubits+1):
    for n in n_gates:
        Intel_QS_filename = Intel_QS_QASM_dir + str(k) + 'qubits_' + str(n) + 'gates'+'.qasmf'
        QuaC_filename = QuaC_QASM_dir + str(k) + 'qubits_' + str(n) + 'gates'+'.quac'
        with open(Intel_QS_filename, 'w', newline='\n') as f:
            print('.malloc' + ' ' + str(k), file=f)
        with open(QuaC_filename, 'w') as f:
            for i in range(k):                
                print('Allocate | Qureg['+ str(i) + ']', file=f)   
                
        for i in range(n):
            j = random.randint(0, k-1)
            c = random.choice(circuits)
            with open(Intel_QS_filename, 'a+', newline='\n') as f:
                print(c + ' q' + str(j), file=f)
            with open(QuaC_filename, 'a+') as f:
                print(c + ' | Qureg[' + str(j) + ']', file=f)
                
        with open(Intel_QS_filename, 'a+', newline='\n') as f:
            print('.free', file=f)
        with open(QuaC_filename, 'a+') as f:
            for i in range(k):                
                print('Deallocate | Qureg['+ str(i) + ']', file=f)  