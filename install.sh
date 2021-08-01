#!/bin/bash

set -e

has_cuda=0

while : ; do
    case "$1" in
        "")
            break;;
        -use-cuda|--use-cuda)
         has_cuda=1;;
        *)
            echo "unknown option $1, try -help"
            exit 2;;
    esac
    shift
done


#wget ftp://ftp.gnu.org/pub/gnu/m4/m4-1.4.18.tar.gz
#tar -xvzf m4-1.4.18.tar.gz
#cd m4-1.4.18
#./configure
#make
#make install
#cp src/m4 /usr/bin
#cd ..
#rm m4-1.4.18.tar.gz



wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
tar -xvf gmp-6.1.2.tar.xz
cd gmp-6.1.2
./configure --enable-cxx
make
make install
cd ..
rm gmp-6.1.2.tar.xz



wget https://files.sri.inf.ethz.ch/eran/mpfr/mpfr-4.1.0.tar.xz
tar -xvf mpfr-4.1.0.tar.xz
cd mpfr-4.1.0
./configure
make
make install
cd ..
rm mpfr-4.1.0.tar.xz

wget https://github.com/cddlib/cddlib/releases/download/0.94m/cddlib-0.94m.tar.gz
tar zxf cddlib-0.94m.tar.gz
cd cddlib-0.94m
./configure
make
make install
cd ..

wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz
tar -xvf gurobi9.1.2_linux64.tar.gz
cd gurobi912/linux64/src/build
sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile
make
cp libgurobi_c++.a ../../lib/
cd ../../
cp lib/libgurobi91.so /usr/local/lib
python3 setup.py install
cd ../../
rm gurobi9.1.2_linux64.tar.gz



export GUROBI_HOME="$(pwd)/gurobi912/linux64"
export PATH="${PATH}:/usr/lib:${GUROBI_HOME}/bin"
export CPATH="${CPATH}:${GUROBI_HOME}/include"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib:${GUROBI_HOME}/lib

git clone https://github.com/eth-sri/ELINA.git
cd ELINA
if test "$has_cuda" -eq 1
then
    ./configure -use-cuda -use-deeppoly -use-gurobi -use-fconv
    cd ./gpupoly/
    cmake .
    cd ..
else
    ./configure -use-deeppoly -use-gurobi -use-fconv
fi
make
make install
cd ..

git clone https://github.com/eth-sri/deepg.git
cd deepg/code
mkdir build
make shared_object
cp ./build/libgeometric.so /usr/lib
cd ../..

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/usr/lib

wget https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_3_50.tf

ldconfig