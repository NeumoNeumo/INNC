#  INNC
INNC Is Neural Networks in CPP.

## Build From Source
```bash
git clone --depth 1 --recurse-submodules https://github.com/NeumoNeumo/libtensor.git
cd libtensor
meson setup --buildtype release releaseBuild && cd releaseBuild
meson compile -j $(nproc)
meson test
```

## Test from source

```bash
git clone --depth 1 --recurse-submodules https://github.com/NeumoNeumo/libtensor.git
cd libtensor
meson setup -Db_sanitize=address build && cd build
meson compile
meson test
```

## Doc generation

[Sphinx](https://www.sphinx-doc.org),
[Breathe](https://github.com/breathe-doc/breathe), and
[Doxygen](https://www.doxygen.nl/) are required to build the html documentation.

For Debian users
```
apt-get install doxygen
python3 -m pip install pipx
pipx install sphinx
pipx install breathe
pipx install sphinx-rtd-theme
```

## Development

1. build system: [meson](https://github.com/mesonbuild/meson)
2. cpp standard: c++20
3. format: default [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
4. documentation generator: [sphinx](https://www.sphinx-doc.org)
5. test framework: [googletest](https://github.com/google/googletest)

You may also simply run
```bash
conda env create -f environment.yml
```
or
```bash
mamba env create -f environment.yml
```

<!-- How to combile googletest with meson? Check this post:
https://stackoverflow.com/questions/57473395/how-to-set-up-googletest-wtih-meson
-->

## TODO
- [ ] different optimization level for debug and release in meson
- [ ] detect memory leakage by `sanitizer`
- [ ] test coverage
- [ ] auto doc generation

