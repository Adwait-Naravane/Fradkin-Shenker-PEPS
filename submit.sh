#!/bin/bash

# Define groups of (hx, hz) points
group1_hx=($(seq 0.2 0.01 0.5))
group1_hz=(0.0)

group2_hx=(0.0)
group2_hz=($(seq 0.2 0.01 0.5))

group3_hx=($(seq 0.25 0.05 0.4))
group3_hz=(0.2)

group4_hx=(0.2)
group4_hz=($(seq 0.25 0.05 0.4))

# All combinations
hx_list=()
hz_list=()

for h in "${group1_hx[@]}"; do
  hx_list+=("$h")
  hz_list+=("0.0")
done

for h in "${group2_hz[@]}"; do
  hx_list+=("0.0")
  hz_list+=("$h")
done

for h in "${group3_hx[@]}"; do
  hx_list+=("$h")
  hz_list+=("0.2")
done

for h in "${group4_hz[@]}"; do
  hx_list+=("0.2")
  hz_list+=("$h")
done

# Chi and D values
chi_vals=(12 16 24 36)
D_vals=(4)

# Ensure log folder exists
mkdir -p pbs_logs

# Submit jobs for all combinations
for ((i=0; i<${#hx_list[@]}; i++)); do
  hx=${hx_list[$i]}
  hz=${hz_list[$i]}

  for chi in "${chi_vals[@]}"; do
    for D in "${D_vals[@]}"; do

      JOBNAME="Gauge-D${D}-chi${chi}-hx${hx}-hz${hz}"
      FILENAME="pbs_job_${hx}_${hz}_chi${chi}_D${D}.pbs"

      cat > "$FILENAME" <<EOF
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
module load juliaup/1.17.9-GCCcore-12.3.0

julia --project=. tests.jl ${hx} ${hz} ${chi} ${D}

mkdir -p Saved_content
mv final_Psi_trivial_hx=${hx}_hz=${hz}_χ=${chi}_D=${D}.jld2 Saved_content/
EOF

      qsub "$FILENAME"
    done
  done
done