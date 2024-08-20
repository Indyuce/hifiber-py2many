# pyrs TODO

## General

- [x] Map function and constructors to existing HiFiber classes.
- [x] Fix constructor of `Tensor` class
- [ ] All tensors must be mutable. Check what `mutability_transformer.py` does and adapt it to HiFiber.
- [ ] Problems with integer types. Replace `i32`'s with `usize`'s
- [x] Add basic HiFiber class/method mapping
- [x] Support HiFiber types in type inference
- [ ] Ability to hide annotations.
- [x] Assignments should register new usings
- [ ] Delete redundant usings

## Future issues to address

- [ ] What do these passes do: `declaration_extractor.py`, `tracer.py`, `context.py`
- [ ] Merge c-like transpiler with rust transpiler classes
- [ ] Investigate if MonkeyType can help provide annotations that you cannot provide otherwise
- [ ] Do not call mutable getters unless necessary

There are multiple pass but only one abstract syntax tree type. Extra data from successive passes only store data directly inside of Python object dictionaries of ast Nodes.

## Things done in pyrs

- [x] Make cli output utf8 file
- [x] Include a `main()` function in output file
- [x] Include error logs in main command
- [x] Map function and constructors to existing HiFiber classes.
- [x] Fix constructor of `Tensor` class

Limitations: no type inference. Cannot make a difference between `Payload::+=` and `&mut Payload::+=` which creates a problem with iterators.