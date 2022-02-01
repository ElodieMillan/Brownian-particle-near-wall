#!/usr/bin/env python

import os


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = "%s/.job" % os.getcwd()
scratch = "/scratch/emillan"
data_dir = os.path.join(scratch, '/project/cumulant4')

# Make top level directories
mkdir_p(job_directory)
mkdir_p(data_dir)

lizards = ["test"]

for lizard in lizards:
    job_file = os.path.join(job_directory, "%s.job" % lizard)
    lizard_data = os.path.join(data_dir, lizard)

    # Create lizard directories
    mkdir_p(lizard_data)

    with open(job_file) as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s.job\n" % lizard)
        #fh.writelines("#SBATCH --output=.out/%s.out\n" % lizard)
        #fh.writelines("#SBATCH --error=.out/%s.err\n" % lizard)
        fh.writelines("#SBATCH --time=2-00:00\n")
        fh.writelines("#SBATCH --mem=96000\n")
        fh.writelines("#SBATCH -N 1\n")
        fh.writelines("#SBATCH --ntasks-per-node=1\n")
        fh.writelines("Rscript /gpfs/home/emillan/projects/Stage2020-Nageurs-actifs-proche-de-parois-deformable/Optimisation_experimental_C4/simulation_C4.py 1e-5 1000000 1 1.5e-6 50e-9 1.5e-6 10\n")

    os.system("sbatch %s" % job_file)
