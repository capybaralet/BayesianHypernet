                #!/bin/sh

                ## Reasonable default values
                # Execute the job from the current working directory.
                #PBS -d /home/capybara/BayesianHypernet/Active_Learning_Tasks/BHN_AL

                #All jobs must be submitted with an estimated run time
                #PBS -l walltime=23:59:59

                ## Job name
                #PBS -N dbi_d33c27a1bec

                ## log out/err files
                # We cannot use output_file and error_file here for now.
                # We will use dbi_...out-id and dbi_...err-id instead
                # Torque append -{JOBID} to the output filename
                # But not Moab, so we add it
                #PBS -o /home/capybara/BayesianHypernet/Active_Learning_Tasks/BHN_AL/LOGS/python__home_capybara_BayesianHypernet_Active_Learning_Tasks_BHN_AL_dk_AL.py_2017-05-18_17-01-57.997227/dbi_d33c27a1bec.out-${MOAB_JOBARRAYINDEX}
                #PBS -e /home/capybara/BayesianHypernet/Active_Learning_Tasks/BHN_AL/LOGS/python__home_capybara_BayesianHypernet_Active_Learning_Tasks_BHN_AL_dk_AL.py_2017-05-18_17-01-57.997227/dbi_d33c27a1bec.err-${MOAB_JOBARRAYINDEX}


                ## The project name for accounting/permission
                #PBS -A jvb-000-ag

        ## Number of CPU (on the same node) per job
        #PBS -l nodes=1:gpus=1

                ## Execute as many jobs as needed
                #PBS -t 0-0%30

                ## Queue name
                #PBS -q gpu_1
export "THEANO_FLAGS=device=gpu,floatX=float32"
export "JOBDISPATCH_RESUBMIT=msub /home/capybara/BayesianHypernet/Active_Learning_Tasks/BHN_AL/LOGS/python__home_capybara_BayesianHypernet_Active_Learning_Tasks_BHN_AL_dk_AL.py_2017-05-18_17-01-57.997227/submit.sh"

                ## Variable to put into the environment
                #PBS -v THEANO_FLAGS,JOBDISPATCH_RESUBMIT

                ## Execute the 'launcher' script in bash
                # Bash is needed because we use its "array" data structure
                # the -l flag means it will act like a login shell,
                # and source the .profile, .bashrc, and so on
                /bin/bash -l -e /home/capybara/BayesianHypernet/Active_Learning_Tasks/BHN_AL/LOGS/python__home_capybara_BayesianHypernet_Active_Learning_Tasks_BHN_AL_dk_AL.py_2017-05-18_17-01-57.997227/launcher
