"""Local rehearsals: run an E-job's arms on the PC's small model before the A100.

A rehearsal checks that every arm runs end to end and reports structural behaviour
(calls, rounds, tokens, exits, parse failures). It never produces results: accuracy and
latency on a small quantised local model are not comparable with the A100 runs.
"""
