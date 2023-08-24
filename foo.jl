using ModelingToolkit, DifferentialEquations
@parameters t L g
@variables x(t) y(t) T(t)
D = Differential(t)

eqs2 = [D(D(x)) ~ T * x,
    D(D(y)) ~ T * y - g,
    0 ~ x^2 + y^2 - L^2]
sys = ODESystem(eqs2, t, [x, y, T], [L, g], name = :pendulum)

# Turn into a first order differential equation system
sys2 = ModelingToolkit.ode_order_lowering(sys)

# Perform index reduction to get an Index 1 DAE
new_sys = dae_index_lowering(sys2)

u0 = [
    D(x) => 0.0,
    D(y) => 0.0,
    x => 1.0,
    y => 0.0,
    T => 0.0,
]

p = [
    L => 1.0,
    g => 9.8,
]

prob_auto = ODEProblem(new_sys, u0, (0.0, 10.0), p)
sol = solve(prob_auto, Rodas5());


# VSCodeServer.@enter TearingState(sys) # works really well! 
@which TearingState(sys)

state = TearingState(sys)
state2 = TearingState(sys2) # stick to this 
@assert length(state2.fullvars) == 9 # 2nd order system returns 7 for this 

# vars!(vars, eq.rhs, op=Symbolics.Operator)

@unpack graph, solvable_graph, var_to_diff, eq_to_diff = state.structure
neqs = nsrcs(graph)
var_eq_matching = pantelides!(state; finalize = false, kwargs...)
return invalidate_cache!(pantelides_reassemble(state, var_eq_matching))