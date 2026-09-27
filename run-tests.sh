#!/bin/sh

set -e

pass=0
fail=0
for bin in "$@"
do
    echo "Running ./$bin"
    if "./$bin"; then
        echo "./$bin passed."
        pass=$((pass + 1))
    else
        echo "./$bin failed!"
        fail=$((fail + 1))
    fi
done

echo "$pass passed. $fail failed."
exit $fail
