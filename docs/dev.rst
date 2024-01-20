Development
============

Standards
------------

* Build system: `Meson <https://github.com/mesonbuild/meson>`_.
* C++ standard: C++20
* Formatter: default `clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_.
* Test framework: `GoogleTest <https://github.com/google/googletest>`_.

Developing Environment
-----------------------

**Automatically**:

You may also simply run

.. code-block:: bash

   conda env create -f environment.yml

or

.. code-block:: bash

   mamba env create -f environment.yml

**Manually**

We need

* `Meson <https://github.com/mesonbuild/meson>`_
* `Ninja <https://ninja-build.org/>`_

to build from source and 

* `Doxygen <https://www.doxygen.nl/>`_
* `Breathe <https://breathe.readthedocs.io/en/latest/>`_
* `Furo <https://pradyunsg.me/furo/quickstart/>`_
* `Sphinx <https://www.sphinx-doc.org/en/master/>`_

to generate the documentation.


The following commands are helpful to Debian users.

.. code-block:: bash

  apt-get install doxygen
  python3 -m pip install pipx
  pipx install sphinx
  pipx install breathe
  pipx install furo
  pipx install gcovr
  pipx install pybind11

Build From Source
-------------------------

.. code-block:: bash

   git clone --depth 1 --recurse-submodules https://github.com/NeumoNeumo/INNC.git
   cd INNC
   meson setup -Db_sanitize=address -Db_coverage=true build/ && cd build
   meson compile -j $(nproc)
   meson test

If you are going to use `clang`, use the following commands instead.

.. code-block:: bash

   git clone --depth 1 --recurse-submodules https://github.com/NeumoNeumo/INNC.git
   cd INNC
   CC=clang CXX=clang++ meson setup -Db_coverage=true -Db_sanitize=address -Db_lu ndef=false --wipe build
   cd build
   meson compile -j $(nproc)
   meson test

Document generation
------------------------

.. code-block:: bash

   git clone --depth 1 --recurse-submodules https://github.com/NeumoNeumo/INNC.git
   cd INNC
   meson setup build && cd build
   ninja docs/sphinx

