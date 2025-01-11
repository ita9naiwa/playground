## build and emit compile_commands.json

```
rm -rf build
bear -- python setup.py build_ext --inplace |& tee log
```
