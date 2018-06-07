[graph]: /images/5_groups_Summary_Intel-QS_only_10-sample_fidelity_T1=10.0_T2=10.0.png "Fidelity of 5 qubit groups"

![alt text][graph]


k-fidelity is a program that can be used to compare the fidelity of quantum simulation results. k-fidelity targets Intel-QS and QuaC. 

The following fidelity comparisons can be performed:
- between noisy and non-noisy Intel-QS
- between Intel-QS and QuaC

To run the program, simply type the following to bring up the help menu:

```sh
python3 k-fidelity -h 
```

to compare results from Intel-QS against itself(noisy vs non-noisy), set any QuaC parameters to 'None'. The program will execute Intel-QS only. Example below:

```sh
python3 k-fidelity.py -K=10 -T1=10 -T2=10 -Intel_QS_interface='/home/Intel-QS/interface/qasm_interface.exe' -Intel_QS_qasm_dir='/home/QASMs/Intel_QS_qasm/' -QuaC_interface='None'
```


to compare results between Intel-QS and QuaC, all parameters need to be populated. Below is an example:

```sh
python3 k-fidelity.py -K=10 -T1=10 -T2=10 -Intel_QS_interface='/home/Intel-QS/interface/qasm_interface.exe' -QuaC_interface='/home/QuaC/projectq_simple_circuit' -Intel_QS_qasm_dir='/home/QASMs/Intel_QS_qasm/' -QuaC_qasm_dir='/home/QASMs/QuaC_qasm/' -graphs_dir='/home/graphs/' -output_dir='/home/output/'
```

Program parameters:

'-K', run the noisy Intel_QS simulation k number of times

'-T1', T1 amplitude damping

'-T2', T2 dephasing    

'-Intel_QS_interface', type=str, Directory of qasm_interface.exe

'-QuaC_interface', type=str, Directory of projectq_simple_circuit. 
                        If left empty, the program will compare noisy Intel-QS simulations against a non-noisy Intel-QS reference simulation'

'-Intel_QS_qasm_dir', type=str, Directory of Intel-QS qasms'

'-QuaC_qasm_dir', type=str, Directory of QuaC qasms, 
                        e.g. QASMs/QuaC_qasm/ 
                        If left empty, the program will compare noisy Intel-QS simulations against a non-noisy Intel-QS reference simulation'

'-graphs_dir', type=str, directory for graphs'

'-output_dir', type=str, directory for output'