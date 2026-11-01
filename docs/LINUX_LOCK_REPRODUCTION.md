# Reproduce the pinned Linux dependencies

The runtime freeze previously included a machine-local torchvision file URL
and an editable adapter URL at an earlier source commit. Neither is a portable
installation instruction. The dependency lock now uses the same torchvision
version and SHA256 as `linux-stack.txt`; adapter source selection is separate.
Historical environment and GPU receipts are unchanged.

On the independently supported Linux host, acquire and verify the three
platform wheels against `requirements/linux-stack.txt`, using an authorized
local wheelhouse. Install those verified wheels without dependency resolution,
then install `requirements/linux-cp310.lock.txt` with that wheelhouse/index
available. This version freeze is not an all-package hash lock: retain the
wheel receipts, complete install report and environment identity in each run.

Install the adapter from the audited checkout using
`python -m pip install --no-deps -e .`, and record its exact Git commit before
running doctor, dependency checks and physical probes. Do not use a second
editable Git URL to silently replace that source. These instructions require
Linux validation; the Windows review does not establish clean Linux install
or GPU reproducibility.
