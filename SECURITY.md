# Security

Do not post credentials or private OCR documents in issues. Report security
concerns through the repository's private security reporting feature if enabled;
otherwise contact the maintainer through their GitHub profile without including
an exploit or private data in a public issue.

Pinned trust_remote_code executes reviewed upstream Python with user privileges.
The original generated-coordinate visualization uses eval; this project does
not call it. Raw OCR remains untrusted content. See docs/SECURITY.md for source
evidence, license boundaries and a proposed minimal postprocessor fix.

Use isolated environments, exact packages, safetensors and immutable revisions.
Do not disable Windows security features to bypass an unexplained failure.
