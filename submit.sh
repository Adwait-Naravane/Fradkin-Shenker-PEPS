# ======================
JOBNAME = Gauge-D4-chi24-hx0-hz0
UG_NAME = anaravan

request_cpus = 40
request_memory = 64 G
request_disk = 2 G

requirements = has_julia && (cpu_performance > 8)

nice_user = TRUE
arguments = --project=. tests.jl
transfer_input_files = tests.jl, new_toolbox.jl, Project.toml

transfer_output_files = CTMHOTRG-ising-32.csv
# ======================

should_transfer_files   = TRUE
preserve_relative_paths = TRUE
when_to_transfer_output = ON_EXIT

output = condorlogs/$(JOBNAME)_$(JobId).out
error  = condorlogs/$(JOBNAME)_$(JobId).err
log    = condorlogs/dump.log

environment = "HOME=/tmp/$(UG_NAME)"

executable = /usr/local/bin/julia
transfer_executable = FALSE

# Send email on events (always, complete, error or never).
# default: never
notification = always
notify_user = adwait.naravane@ugent.be

queue 1