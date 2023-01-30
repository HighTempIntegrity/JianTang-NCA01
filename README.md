# Neural Cellular Automata for Solidification Microstructure Modeling
Building NCA for accelerating solidification microstructure modeling

Folder *MUlti-GPU-Code* contains codes for training and evaluating the NCA.  The file names and corresponding functions are as follow:

 |         **File**                          |                     **Function**                                      |
 | ------------------------------------:  | :---------------------------------------------------------------: |
 | MUlti-GPU-Code/CA.py                   |    CA simulation for generating training and validation data  |
 | MUlti-GPU-Code/CA_test_run.py          |    CA simulation for testing                                  |
 | MUlti-GPU-Code/Evaluate_model.py       |    Perform test run                                           |
 | MUlti-GPU-Code/LoadFunc_train.py       |    Load useful function and python package                    |
 | MUlti-GPU-Code/Loadmodel_T_train.py    |    Define and load the NCA model                              |
 | MUlti-GPU-Code/generate_larger_grad.py |    Setting for generating training and validation data        |
 | MUlti-GPU-Code/mgpu_ppre_t_grad.py     |    Train the NCA                                              |
  
  The files of testing results and trained models for various cases are in Folder *Paper_review/Cases*. Details see *Paper_review/Cases/README.md*. 
