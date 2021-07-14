solvers:
	f2py -c lwr/src/lwr_del_Cast.f90 -m lwr_del_Cast;

	mkdir -p lwr/bin
	mv *.so lwr/bin/

	if [ -d lwr_del_Cast.cpython-36m-darwin.so.dSYM ]; then rm -Rf lwr_del_Cast.cpython-36m-darwin.so.dSYM; fi


test:
	py.test
