# Define groups of (hx, hz) points
export LC_NUMERIC=C

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
# Build combined hx/hz pairs
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

for h in "${group5_hx[@]}"; do
  hx_list+=("$h")
  hz_list+=("$h")
done

for ((i=0; i<${#group6_hx[@]}; i++)); do
  hx_list+=("${group6_hx[$i]}")
  hz_list+=("${group6_hz[$i]}")
done

for ((i=0; i<${#group7_hx[@]}; i++)); do
  hx_list+=("${group7_hx[$i]}")
  hz_list+=("${group7_hz[$i]}")
done

chi_vals=(16 24 36)
D_vals=(4)


for D in "${D_vals[@]}"; do
  for chi in "${chi_vals[@]}"; do
    for ((i=0; i<${#hx_list[@]}; i++)); do
      hx=${hx_list[$i]}
      hz=${hz_list[$i]}

      # Clean locale
      hx_clean=${hx/,/.}
      hz_clean=${hz/,/.}
      printf -v hx_fmt "%.3f" "$hx_clean"
      printf -v hz_fmt "%.3f" "$hz_clean"

      JOBNAME="Gauge-D${D}-chi${chi}-hx${hx_fmt}-hz${hz_fmt}"
      FILENAME="job_${hx_fmt}_${hz_fmt}_chi${chi}_D${D}.sub"

      cat > "$FILENAME" <<EOF
JOBNAME = ${JOBNAME}
UG_NAME = anaravan
request_cpus = 20
request_memory = 64 G
request_disk = 2 G

requirements = has_julia && (cpu_performance > 8)

nice_user = TRUE
arguments = --project=. FSmodel_PEPSopt.jl ${hx_clean} ${hz_clean} ${chi} ${D}
transfer_input_files = OptimKit.jl, FSmodel_PEPSopt.jl, new_toolbox.jl, Project.toml

transfer_output_files = final_Psi_trivial_1e4_hx=${hx_fmt}_hz=${hz_fmt}_χ=${chi}_D=${D}.jld2

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
EOF

      condor_submit -pool octopus -remote octopus "$FILENAME"

    done
  done
done