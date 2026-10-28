# Independent code review

An independent reviewer inspected the initial harness at 9839ad9 and then
the substantial new compatibility/postprocessing/reproduction code at
26335ab. No Critical or Minor findings. Eight Important findings were
addressed with failing regressions before fixes:

| Finding | Regression / resolution |
|---|---|
| Adjacent table cells merged for numerical checks | Correct HTML table passes; wrong/missing numbers fail |
| Gate E lacked linked comparable baseline | Distinct producing environments, machine/artifact/package equality and required cases |
| Inherited DASH environment could change routing | Absent/matching overrides allowed; conflicting routing/interpreter overrides refused |
| Windows newline conversion changed raw output | Exact UTF-8 bytes, including LF/CRLF/lone CR |
| GPU telemetry could hide original error | Original traceback and separate telemetry failure saved |
| Windows11/CPU evidence not enforced | Wrong/missing client build and processor rejected |
| Summarizer/fresh flag could manufacture Gate E | Producer IDs cross-checked; run ID and creation witness; inventory from producing reports |
| Title level caused unbounded Markdown expansion | Bound levels/input length; vendor unchanged; raw/failure manifest retained |

The reviewer performed CPU-only checks and did not claim hardware validation.
Fixes after each review are verified by regression tests and the final suite;
there was no second opinion or claim of an independently reviewed final
commit. Physical evidence belongs to actual local commands/manifests.

Deferred minors: none.

Review limits: GPU correctness and clean reproduction were reserved for
physical runs; alternative hardware, concurrent embedding and Phase2 work
were outside scope. Final documentation followed the physical results.
The supported CLI serializes model calls in a dedicated process.
