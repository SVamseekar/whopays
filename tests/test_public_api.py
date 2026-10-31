import subprocess
import sys


def test_public_run_stays_callable_after_submodule_import():
    subprocess.run([sys.executable, '-c',
                    'import whopays.run; import whopays; assert callable(whopays.run)'], check=True)
