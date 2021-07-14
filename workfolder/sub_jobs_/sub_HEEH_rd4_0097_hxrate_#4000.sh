#!/bin/bash
#SBATCH -A p31346
#SBATCH -p short
#SBATCH -t 1:15:00
#SBATCH -N 1
#SBATCH --mem=2G
#SBATCH --ntasks-per-node=6
#SBATCH --output=../../workfolder/sub_jobs_/HEEH_rd4_0097_hxrate_#4000.out
#SBATCH --error=../../workfolder/sub_jobs_/HEEH_rd4_0097_hxrate_#4000.err
#SBATCH --job-name=HEEH_rd4_0097_hxrate_#4000

#sub commands
module purge
#
source ~/.bashrc
#
source activate hxrate
#

#python script command
python /projects/p31346/suggie/hxrate/scripts/hx_rate/hx_ratefit_cml.py -i ../../workfolder/input_hx_dist/HEEH_rd4_0097_hx_mass_dist.csv -s HMTQVHVDGVTYTFSNPEEAKKFADEMAKRKGGTWEIKDGHIHVE -n HEEH_rd4_0097 -p ../../params/params.yml -o ../../workfolder/output_hxrate

#end
