using Flux
using OpenAIGym
using Printf: @printf
using PyCall
import Reinforce: action

const env = GymEnv(:LunarLander, :v2)

const actions = 0:3

const hidden_layer_size = 400

const q_model = Chain(
    Dense(8 + 1, hidden_layer_size, relu),
    Dense(hidden_layer_size, 1))

loss(x, y) = Flux.mse(q_model(x), y) + 0.001 * sum(StatsBase.norm, Flux.params(q_model))

optimizer = NADAM()

function train(sars, epochs; verbose=false)
    x = to_q_x(sars)
    y = to_q_y(sars)
    if verbose
        @printf(
            "Epoch %4d / %4d - Loss: %8.4f\n",
            0,
            epochs,
            loss(x, y).data)
    end
    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(q_model), [(x, y)], optimizer)
        if verbose
            @printf(
                "Epoch %4d / %4d - Loss: %8.4f\n",
                epoch,
                epochs,
                loss(x, y).data)
        end
    end
end

struct QPolicy <: AbstractPolicy
    e
end
function action(policy::QPolicy, r, s, A)
    if rand() < policy.e
        return rand(actions)
    end
    argmax(action_values(s)) - 1
end

function run_episodes(n_episodes, policy)
    sars = Tuple{PyCall.PyArray{Float32,1},Int64,Float64,PyCall.PyArray{Float32,1}}[]
    rewards = Float64[]
    for episode in 1:n_episodes
        reward = run_episode(env, policy) do (s, a, r, s_next)
            push!(sars, (s, a, r, s_next))
            render(env)
        end
        push!(rewards, reward)
    end
    close(env)
    sars, rewards
end

function to_q_x(sars::AbstractArray{Tuple{PyCall.PyArray{Float32,1},Int64,Float64,PyCall.PyArray{Float32,1}}})
    to_q_x((s, a) for (s, a, _, _) in sars)
end

function to_q_x(sas)
    xs = map(sas) do (s, a)
        x = convert(Array, s)
        push!(x, a)
        x
    end
    hcat(xs...)
end

function action_values(s)
    map(actions) do a
        q_model(to_q_x([(s, a)]))[1].data
    end
end

V(s) = maximum(action_values(s))

function to_q_y(sars::AbstractArray{Tuple{PyCall.PyArray{Float32,1},Int64,Float64,PyCall.PyArray{Float32,1}}})
    to_q_y((r, s_next) for (_, _, r, s_next) in sars)
end

function to_q_y(rss)
    map(rss) do (r, s_next)
        r + 0.9 * V(s_next)
    end
end

truncate(a, n) = a[max(1, end-n+1):end]

function run()
    sars = Tuple{PyCall.PyArray{Float32,1},Int64,Float64,PyCall.PyArray{Float32,1}}[]
    for i in 1:1000
        new_sars, rewards = run_episodes(10, QPolicy(0.2))
        sars = truncate(vcat(sars, new_sars), 1000)
        @printf("Average rewards: %8.2f\n", sum(rewards) / length(rewards))
        for _ in 1:10
            train(sars, 10)
        end
    end
end

run()
