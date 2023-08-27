now that pants(pend_sys()) at least doesn't panic the next steps are 
1) add tests to verify the matching is correct (pretty sure the matching is correct)
2) understand why index-1 is okay for IDA/BDF and if MTK knows to pick the right solver for index-1 
2) reassemble eqs/system. 
3) codegen for sundials 



1) x - y
2) y

eqs:
1.dx - x - y
2.x + 2y 

D(2.)
dx + 2dy => dy = -1/2 dx => dy = -1/2 (x + y)

dx = x + y
dy = -1/2 (x + y)


dy = -(dx + da) / 2
dx = x + y 

dy = -(x + y + da) / 2
dx = x + y 



ex2. 
1.dx + dy - t^2 
2.x + y^2 - t 

D(2.) 
dx + 2ydy - 1 => dx = 1 - 2ydy

dy = t^2 - dx  => dy = t^2 - 1 + 2ydy

manual index reduction of pendulum to index 0:
x^2 + y^2 -> D -> D -> substitute x'' and y'' -> D -> substitute x' x'' y' y'' 

now we have a system that has explicit rhs consisting only of state variables and rhs that are der vars only 
dx = w
dy = z
dw = Tx 
dz = Ty - g
dT = ... some crazy big expression ((-4*T[t]*w[t]*x[t]+g*z[t]-2*T[t]*y[t]*z[t]-2*(-g+T[t]*y[t])*z[t])/(x[t]^2+y[t]^2))


 

links

https://www.fs.isy.liu.se/en/Edu/Courses/Simulation/OH/dae1.pdf 
https://www.fs.isy.liu.se/en/Edu/Courses/Simulation/OH/

pantelides 
https://sci-hub.se/https://epubs.siam.org/doi/abs/10.1137/0909014

structural matrix method 
https://www.sciencedirect.com/science/article/abs/pii/0098135494000945
https://pubs.acs.org/doi/10.1021/ie0341754

the best reference on DAE index reduction is probably https://reference.wolfram.com/language/tutorial/NDSolveDAE (as usual wolfram)

it seems after that the dummy derivative method is needed to prevent numerical drift
https://www.researchgate.net/publication/235324214_Index_Reduction_in_Differential-Algebraic_Equations_Using_Dummy_Derivatives
    