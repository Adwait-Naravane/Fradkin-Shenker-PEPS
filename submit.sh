#!/bin/bash

hx_vals=(0.1 0.2 0.3 0.32 0.33 0.34 0.4 0.5 1.0)
hz_vals=(0.1 0.2 0.3 0.32 0.33 0.34 0.4 0.5 1.0)
chi_vals=(48)
D_vals=(6)

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
#PBS -M adwait.naravane@ugent.be
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