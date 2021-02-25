Three different approaches
==========================

NERSC TaskFarmer manager
------------------------

The `submit_taskfarmer.sl` is the SLURM script that should be executed. This will read the list of operations listed under the `tasks.txt` file, which calls for each task the `wrapper.sh` script containing the full python command to be executed. However, as suggested in `this forum <https://unix.stackexchange.com/questions/197192/correct-xargs-parallel-usage>`_, this appraoch, like the the `xargs` approach mentioned `here <https://support.ceci-hpc.be/doc/_contents/QuickStart/SubmittingJobs/SlurmTutorial.html#packed-jobs-example>`_ is likely to give poor performance because the running time will be dominated by starting up a different python virtual machine for each operation.

Local execution
---------------

daily.sh  parallel.py  parallel.py~  submit_taskfarmer.sl  tasks.txt  tasks.txt.tfin  wrapper.sh

