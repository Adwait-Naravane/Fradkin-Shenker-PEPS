#!/bin/bash

hx_vals=(0.0)
hz_vals=(0.1 0.13 0.15 0.17 0.2 0.23 0.25 0.27 0.3 0.31 0.315 0.32 0.325 0.327 0.328 0.329 0.33 0.331 0.332 0.333 0.334 0.335 0.336 0.337 0.338 0.34 0.343 0.348 0.35 0.36 0.38 0.4 0.45 0.47 0.5 )
chi_vals=(20 36)
D_vals=(4)

for hx in "${hx_vals[@]}"; do
  for hz in "${hz_vals[@]}"; do
    for chi in "${chi_vals[@]}"; do
      for D in "${D_vals[@]}"; do

        JOBNAME="Gauge-D${D}-chi${chi}-hx${hx}-hz${hz}"
        FILENAME="pbs_job_${hx}_${hz}_chi${chi}_D${D}.pbs"

        cat > $FILENAME <<EOF
#!/bin/bash
#PBS -N ${JOBNAME}
#PBS -l nodes=1:ppn=20
#PBS -l mem=128gb
#PBS -l walltime=48:00:00
#PBS -o pbs_logs/${JOBNAME}_\$PBS_JOBID.out
#PBS -e pbs_logs/${JOBNAME}_\$PBS_JOBID.err
#PBS -m abe

#PBS -j oe

cd \$PBS_O_WORKDIR

# Load Julia module (if available on your HPC)
module load juliaup/1.17.9-GCCcore-12.3.0

# Run Julia script with args
julia --project=. tests.jl ${hx} ${hz} ${chi} ${D}

mkdir -p Saved_content
mv final_Psi_trivial_hx=${hx}_hz=${hz}_χ=${chi}_D=${D}.jld2 Saved_content/
EOF

        qsub $FILENAME

      done
    done
  done
done