# Determinism harness helpers.
#
# Tests that compare a hash of a computed buffer against a known-good
# golden value are labeled "determinism". The ci-determinism workflow
# preset filters to only these. Smoke-level tests use the "smoke" label;
# everything else runs in the full CI test set by default.
#
# Usage:
#   dyphur_add_smoke_test(test_target)
#   dyphur_add_determinism_test(test_target)

function(dyphur_add_smoke_test target)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "smoke")
endfunction()

function(dyphur_add_determinism_test target)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "determinism")
endfunction()
