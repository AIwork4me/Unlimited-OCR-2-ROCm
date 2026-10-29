# Authored synthetic scorer fixtures

These documents and raw strings were written for this repository; they contain
no benchmark source content. `synthetic.json` covers a two-page continuation of
text and a table, a heading and a page-attributed figure. Convert
`synthetic.raw.txt` with the byte-identical official V2 converter, then run the
original evaluator. The seven non-formula metrics should be perfect. FormulaCDM
has a separate `x+y=z` identical-formula known-answer test in the worker; it must
render successfully and score 1 before the eight-metric gate can pass.

The worker also supplies imperfect matched-sample fixtures with hand-derived
expected values, checking document averages against length-weighted averages.
