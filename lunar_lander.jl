using Flux
using OpenAIGym
using Printf: @printf
using ProgressMeter
using PyCall
import Reinforce: action

const env = GymEnv(:LunarLander, :v2)

const actions = 0:3

const hidden_layer_size = 2_000

const q_model = Chain(
    Dense(8 + 1, hidden_layer_size, leakyrelu),
    Dense(hidden_layer_size, 1, leakyrelu))

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

function run_episodes(n_episodes, policy; close_env=true)
    sars = Tuple{PyCall.PyArray{Float32,1},Int64,Float64,PyCall.PyArray{Float32,1}}[]
    rewards = Float64[]
    for episode in 1:n_episodes
        reward = run_episode(env, policy) do (s, a, r, s_next)
            push!(sars, (s, a, r, s_next))
            render(env)
        end
        push!(rewards, reward)
    end
    if close_env
        close(env)
    end
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

function sars_losses(sars)
    map(sars) do sar
        x = to_q_x([sar])
        y = to_q_y([sar])
        loss(x, y)
    end
end

function trim_memories(sars)
    to_remove = Set(sortperm(sars_losses(sars)[1:round(Int, end/2)])[1:round(Int, end/2)])
    [sars[i] for i in 1:length(sars) if !in(i, to_remove)]
end

function run()
    sars = Tuple{PyCall.PyArray{Float32,1},Int64,Float64,PyCall.PyArray{Float32,1}}[]
    for i in 1:1000
        new_sars, rewards = run_episodes(10, QPolicy(0.2), close_env=false)
        sars = trim_memories(vcat(sars, new_sars))
        pre_loss = sum(sars_losses(sars)) / length(sars)
        for _ in 1:10
            train(sars, 10)
        end
        post_loss = sum(sars_losses(sars)) / length(sars)
        @printf("%4d - Memory length: %4d    Average rewards: %8.2f    Loss: %4.2f -> %4.2f\n",
                i, length(sars), sum(rewards) / length(rewards), pre_loss, post_loss)
    end
end

run()
