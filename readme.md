## build and emit compile_commands.json

```
rm -rf build
bear -- python setup.py build_ext --inplace |& tee log
```


## Run `clang-format`
```
find . -name "*.cpp" -o -name "*.cu" | xargs clang-format -i
```

## run sanitizer
```
/usr/local/cuda/bin/compute-sanitizer python test_conv.py
```