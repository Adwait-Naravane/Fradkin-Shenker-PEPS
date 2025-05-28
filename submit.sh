#!/bin/bash

export LC_NUMERIC=C

# Define groups
group1_hx=($(seq 0.25 0.01 0.45))
group1_hz=(0.0)
group2_hx=(0.0)
group2_hz=($(seq 0.25 0.01 0.45))
group3_hx=($(seq 0.25 0.005 0.4))
group3_hz=(0.21)
group4_hx=(0.21)
group4_hz=($(seq 0.25 0.005 0.4))
group5_hx=($(seq 0.25 0.01 0.45))
group5_hz=($(seq 0.25 0.01 0.45))
group6_hx=($(seq 0.25 0.01 0.45))
group6_hz=($(seq 0.45 -0.01 0.25))
group7_hx=($(seq 0.5 0.01 0.6))
group7_hz=($(seq 0.6 -0.01 0.5))

# Build hx/hz list
hx_list=()
hz_list=()
for h in "${group1_hx[@]}"; do hx_list+=("$h"); hz_list+=("0.0"); done
for h in "${group2_hz[@]}"; do hx_list+=("0.0"); hz_list+=("$h"); done
for h in "${group3_hx[@]}"; do hx_list+=("$h"); hz_list+=("0.21"); done
for h in "${group4_hz[@]}"; do hx_list+=("0.21"); hz_list+=("$h"); done
for h in "${group5_hx[@]}"; do hx_list+=("$h"); hz_list+=("$h"); done
for ((i=0; i<${#group6_hx[@]}; i++)); do hx_list+=("${group6_hx[$i]}"); hz_list+=("${group6_hz[$i]}"); done
for ((i=0; i<${#group7_hx[@]}; i++)); do hx_list+=("${group7_hx[$i]}"); hz_list+=("${group7_hz[$i]}"); done

chi_vals=(16 24 36)
D_vals=(6)

mkdir -p pbs_logs

for D in "${D_vals[@]}"; do
  for chi in "${chi_vals[@]}"; do
    for ((i=0; i<${#hx_list[@]}; i++)); do
      hx=${hx_list[$i]}
      hz=${hz_list[$i]}

      # Clean float formatting
      hx=${hx/,/.}
      hz=${hz/,/.}
      printf -v hx_fmt "%.3f" "$hx"
      printf -v hz_fmt "%.3f" "$hz"

      JOBNAME="Gauge-D${D}-chi${chi}-hx${hx_fmt}-hz${hz_fmt}"
      FILENAME="pbs_job_${hx_fmt}_${hz_fmt}_chi${chi}_D${D}.pbs"

      cat > "$FILENAME" <<EOF
#!/bin/bash
#PBS -N ${JOBNAME}
#PBS -l nodes=1:ppn=20
#PBS -l mem=64gb
#PBS -l walltime=48:00:00
#PBS -o pbs_logs/${JOBNAME}_\$PBS_JOBID.out
#PBS -e pbs_logs/${JOBNAME}_\$PBS_JOBID.err
#PBS -j oe

cd \$PBS_O_WORKDIR
module load juliaup/1.17.9-GCCcore-12.3.0

julia --project=. tests.jl ${hx_fmt} ${hz_fmt} ${chi} ${D}
EOF

      qsub "$FILENAME"
    done
  done
done