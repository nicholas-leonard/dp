package = "dp"
version = "scm-1"

source = {
   url = "https://github.com/nicholas-leonard/dp"
}

description = {
   summary = "A deep learning library designed for streamlining research and development using the Torch7 distribution, with an emphasis on flexibility through the elegant use of object-oriented design patterns.",
   detailed = [[
dp is a deep learning library designed for streamlining research and development using the Torch7 distribution, 
with an emphasis on flexibility through the elegant use of object-oriented design patterns.
Inspired by pylearn2/Theano, it provides common datasets like MNIST, CIFAR-10 and CIFAR-100, 
preprocessing like Zero-Component Analysis whitening, Global Contrast Normalization, 
Lecunn's Local Contrast Normalization and facilities for interfacing your own. Additionally, 
it provides a high-level framework that abstracts away common usage patterns of the nn and torch7 
package such as loading datasets and early stopping. The library includes hyperparameter optimization 
facilities for sampling and running experiments from the command-line or prior hyper-parameter distributions.
]],
   homepage = "https://github.com/nicholas-leonard/dp"
}

dependencies = {
   "torch >= 7.0",
   "optim >= 1.0.5",
   "moses >= 1.3.1",
   "nnx >= 0.1",
   "fs >= 0.3",
   "xlua >= 1.0",
   "image >= 1.0",
   "luafilesystem >= 1.6.2",
   "sys >= 1.1"
}


build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
