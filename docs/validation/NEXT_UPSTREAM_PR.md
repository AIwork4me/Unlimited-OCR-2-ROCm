# Engineering follow-up after local FIX_VERIFIED

The source-built artifact passes the physical Windows gfx1151 BF16 forward gate.
Recommend a ROCm/TheRock packaged-wheel regression gate that installs each exact
output wheel into a fresh environment, verifies the complete loaded native/image
identity, then executes the frozen forced FLASH/EFFICIENT and representative
masked-vision cases in independent processes with numerical and HIP launch checks.

Keep runtime/header/image source identities and archive hashes in the build
manifest. Verify the actual published wheel separately; the original AMD wheel
remains RED and this result validates the local source-built replacement.

This mission did not isolate one fixing commit or establish source-fix ownership.
No upstream Issue/PR, release, production OCR change or merge was performed.
