hx_vals=(0.1 0.2 0.3 0.32 0.33 0.34 0.5 1.0)
hz_vals=(0.1 0.2 0.3 0.32 0.33 0.34 0.5 1.0)
chi_vals=(36)
D_vals=(4)

for hx in "${hx_vals[@]}"; do
  for hz in "${hz_vals[@]}"; do
    for chi in "${chi_vals[@]}"; do
      for D in "${D_vals[@]}"; do
        JOBNAME="Gauge-D${D}-chi${chi}-hx${hx}-hz${hz}"

        cat > job_${hx}_${hz}_chi${chi}_D${D}.sub <<EOF
JOBNAME = ${JOBNAME}
UG_NAME = anaravan

request_cpus = 20
request_memory = 64 G
request_disk = 2 G

requirements = has_julia && (cpu_performance > 8)

nice_user = TRUE
arguments = --project=. tests.jl ${hx} ${hz} ${chi} ${D}
transfer_input_files = tests.jl, new_toolbox.jl, Project.toml

transfer_output_files = final_Psi_trivial_hx=${hx}_hz=${hz}_χ=${chi}_D=${D}.jld2

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

        condor_submit -pool octopus -remote octopus job_${hx}_${hz}_chi${chi}_D${D}.sub

      done
    done
  done
done