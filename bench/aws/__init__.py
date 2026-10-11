"""Linux benchmark jobs for the lead's AWS runner (DESIGN.md section 12.4; WP11).

Transport (bundle download, result upload, self-termination) belongs to the lead's runner
(``research/aws/runner.py``). A job here is one shell command run from the bundle root with
``SREMU_JOB_OUT`` set: ``bash bench/aws/job.sh JOB [--deadline-min M]`` (host preparation)
-> :mod:`bench.aws.run_job` (CPU plan, pinning, suites, ``status.json``). See
``bench/README.md`` for the commands and instance types.
"""
