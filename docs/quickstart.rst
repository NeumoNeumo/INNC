Quick Start
===========

Installation
------------

.. code-block:: bash

   git clone --depth 1 --recurse-submodules https://github.com/NeumoNeumo/libtensor.git
   cd libtensor
   meson setup --buildtype release releaseBuild && cd releaseBuild
   meson compile -j $(nproc)
   meson install

First Trial
-----------

.. code-block:: cpp

   #include <INNC/INNC.h>
   #include <iostream>
   int main(){
     double data1[] = {1,2,3,4,5,6};
     double data2[] = {2,7,1,8,2,8};
     auto a = INNC::Tensor::from_blob(data1, {2, 3}, INNC::f64);
     auto b = INNC::Tensor::from_blob(data1, {2, 3}, INNC::f64);
     std::cout << "a:" << a.to_string() << std::endl;
     std::cout << "b:" << b.to_string() << std::endl;
     a.requires_grad(true);
     auto c = (a - b) * (a * a - a * b + b * b);
     c.sum().backward();
     std::cout << a.grad().to_string() << std::endl;
     return 0;
   }


More Examples
--------------

.. TODO
