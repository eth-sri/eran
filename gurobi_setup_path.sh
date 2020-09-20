export GUROBI_HOME="$(pwd)/gurobi903/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export CPATH="${CPATH}:${GUROBI_HOME}/include"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${GUROBI_HOME}/lib
