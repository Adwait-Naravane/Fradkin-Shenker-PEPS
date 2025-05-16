<<<<<<< HEAD
#!/bin/bash

hx_vals=(0.1 0.2 0.3 0.32 0.33 0.34 0.4 0.5 1.0)
hz_vals=(0.1 0.2 0.3 0.32 0.33 0.34 0.4 0.5 1.0)
chi_vals=(48)
D_vals=(6)
=======
hx_vals=(0.1 0.2 0.25 0.3 0.31 0.32 0.325 0.33 0.335 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
hz_vals=(0.1 0.2 0.25 0.3 0.31 0.32 0.325 0.33 0.335 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
chi_vals=(48)
D_vals=(4 6)
>>>>>>> main

for hx in "${hx_vals[@]}"; do
  for hz in "${hz_vals[@]}"; do
    for chi in "${chi_vals[@]}"; do
      for D in "${D_vals[@]}"; do

        JOBNAME="Gauge-D${D}-chi${chi}-hx${hx}-hz${hz}"
        FILENAME="pbs_job_${hx}_${hz}_chi${chi}_D${D}.pbs"

<<<<<<< HEAD
        cat > $FILENAME <<EOF
#!/bin/bash
#PBS -N ${JOBNAME}
#PBS -l nodes=1:ppn=20
#PBS -l mem=128gb
#PBS -l walltime=48:00:00
#PBS -o pbs_logs/${JOBNAME}_\$PBS_JOBID.out
#PBS -e pbs_logs/${JOBNAME}_\$PBS_JOBID.err
#PBS -m abe
#PBS -M adwait.naravane@ugent.be
#PBS -j oe

cd \$PBS_O_WORKDIR
=======
        cat > job_${hx}_${hz}_chi${chi}_D${D}.sub <<EOF
JOBNAME = ${JOBNAME}
UG_NAME = anaravan
request_cpus = 20
request_memory = 64 G
request_disk = 2 G
>>>>>>> main

# Load Julia module (if available on your HPC)
module load juliaup/1.17.9-GCCcore-12.3.0

# Run Julia script with args
julia --project=. tests.jl ${hx} ${hz} ${chi} ${D}

<<<<<<< HEAD
mkdir -p Saved_content
mv final_Psi_trivial_hx=${hx}_hz=${hz}_χ=${chi}_D=${D}.jld2 Saved_content/
=======
transfer_output_files = final_Psi_trivial_1e4_hx=${hx}_hz=${hz}_χ=${chi}_D=${D}.jld2

should_transfer_files   = TRUE
preserve_relative_paths = TRUE
when_to_transfer_output = ON_EXIT

output = condorlogs/\$(JOBNAME)_\$(JobId).out
error  = condorlogs/\$(JOBNAME)_\$(JobId).err
log    = condorlogs/dump.log

environment = "HOME=/tmp/\$(UG_NAME)"

executable = /usr/local/bin/julia
transfer_executable = FALSE

notification = always
notify_user = adwait.naravane@ugent.be

queue 1
>>>>>>> main
EOF

        qsub $FILENAME

      done
    done
  done
done