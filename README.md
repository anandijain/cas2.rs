now that pants(pend_sys()) at least doesn't panic the next steps are 
1) add tests to verify the matching is correct
2) reassemble eqs/system
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


 