using Flux
using OpenAIGym
using StatsBase
import Reinforce: action

const env = GymEnv(:LunarLander, :v2)

const actions = 0:3

const hidden_layer_size = 100

function make_q_model()
    Chain(
        Dense(8 + 1, hidden_layer_size, leakyrelu),
        Dense(hidden_layer_size, hidden_layer_size, leakyrelu),
        Dense(hidden_layer_size, 1))
end

loss(model) = (x, y) -> Flux.mse(model(x), y)
loss(model, x, y) = loss(model)(x, y)

struct SARSF
    s :: Array{Float32, 1}
    a :: Int8
    r :: Float32
    s_next :: Array{Float32, 1}
    f :: Bool
end

struct Policy <: AbstractPolicy end
function action(policy::Policy, s, r, A)
    rand(A)
end

function action_values(q_model, s)
    map(actions) do a
        q_model(to_x(s, a))[1].data
    end
end

V(q_model, s) = maximum(action_values(q_model, s))

function to_x(sarsf::AbstractArray{SARSF})
    to_x((i.s, i.a) for i in sarsf)
end

function to_x(s, a)
    to_x([(s, a)])
end

function to_x(sa)
    xs = map(sa) do (s, a)
        s :: AbstractArray{Float32, 1}
        x = copy(s)
        push!(x, a)
        x
    end
    hcat(xs...)
end

function to_y(q_model, sarsf::AbstractArray{SARSF})
    to_y(q_model, (i.a, i.r, i.s_next, i.f) for i in sarsf)
end

function to_y(q_model, a, r, s, f)
    to_y(q_model, [(a, r, s, f)])
end

function to_y(q_model, arsf; discount_factor=0.9)
    map(arsf) do (a, r, s_next, f)
        if f
            r
        else
            r + discount_factor * V(q_model, s_next)
        end
    end
end

function train(q_model, optimizer, sarsf, epochs)
    x = to_x(sarsf)
    y = to_y(q_model, sarsf)
    for epoch in 1:epochs
        Flux.train!(loss(q_model), Flux.params(q_model), [(x, y)], optimizer)
    end
end

function run_episodes(n_episodes, policy)
    sarsf = SARSF[]
    for episode in 1:n_episodes
        run_episode(env, policy) do (s, a, r, s_next)
            push!(sarsf, SARSF(s, a, r, s_next, finished(env)))
            render(env)
        end
    end
    sarsf
end

truncate(a, n) = a[1:min(n, end)]

function run()
    q_model = make_q_model()
    optimizer = NADAM()
    memory_size = 10_000
    memory = SARSF[]
    train_sample_size = 100
    episodes_per_cycle = 10
    training_epochs_per_cycle = 10
    for learning_cycle in 1:300
        println(learning_cycle)
        new_sarsf = run_episodes(episodes_per_cycle, Policy())
        memory = truncate(vcat(new_sarsf, memory), memory_size)
        train(q_model, optimizer, sample(memory, train_sample_size), training_epochs_per_cycle)
    end
end

if !isinteractive()
    run()
end
