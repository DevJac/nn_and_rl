using BSON: @load, @save
using Flux
using OpenAIGym
using Printf: @printf
using ProgressMeter
using PyCall
import Reinforce: action

const env = GymEnv(:LunarLander, :v2)

const actions = 0:3

const hidden_layer_size = 400

const q_model = Chain(
    Dense(8 + 1, hidden_layer_size, leakyrelu),
    Dense(hidden_layer_size, 1, leakyrelu))

function loss(x, y; regularize=true)
    Flux.mse(q_model(x), y) + (regularize ? 0.001 * sum(StatsBase.norm, Flux.params(q_model)) : 0)
end

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
    sars = Tuple{PyCall.PyArray{Float32,1},Int64,Float64,PyCall.PyArray{Float32,1},Bool}[]
    rewards = Float64[]
    for episode in 1:n_episodes
        reward = run_episode(env, policy) do (s, a, r, s_next)
            push!(sars, (s, a, r, s_next, finished(env, s)))
            render(env)
        end
        push!(rewards, reward)
    end
    if close_env
        close(env)
    end
    sars, rewards
end

function to_q_x(sars::AbstractArray{Tuple{PyCall.PyArray{Float32,1},Int64,Float64,PyCall.PyArray{Float32,1},Bool}})
    to_q_x((s, a) for (s, a, _, _, _) in sars)
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

function to_q_y(sars::AbstractArray{Tuple{PyCall.PyArray{Float32,1},Int64,Float64,PyCall.PyArray{Float32,1},Bool}})
    to_q_y((r, s_next, f) for (_, _, r, s_next, f) in sars)
end

function to_q_y(rsfs)
    map(rsfs) do (r, s_next, f)
        if f
            r
        else
            r + 0.99 * V(s_next)
        end
    end
end

truncate(a, n) = a[max(1, end-n+1):end]

function sars_losses(sars)
    map(sars) do sar
        x = to_q_x([sar])
        y = to_q_y([sar])
        loss(x, y, regularize=false)
    end
end

function trim_memories(sars)
    to_remove = Set(sortperm(sars_losses(sars)[1:round(Int, end/2)])[1:round(Int, end/2)])
    [sars[i] for i in 1:length(sars) if !in(i, to_remove) && rand() < 0.9]
end

function save_model()
    q_model_local = q_model
    @save "q_model.bson.temp" q_model_local
    mv("q_model.bson.temp", "q_model.bson", force=true)
end

function load_model()
    @load "q_model.bson" q_model_local
    Flux.loadparams!(q_model, Flux.params(q_model_local))
end

function run()
    if isfile("q_model.bson")
        load_model()
    end
    sars = Tuple{PyCall.PyArray{Float32,1},Int64,Float64,PyCall.PyArray{Float32,1},Bool}[]
    for i in 1:1_000_000
        new_sars, rewards = run_episodes(1, QPolicy(0.2), close_env=false)
        sars = truncate(vcat(sars, new_sars), 10_000)
        pre_loss = sum(sars_losses(sars)) / length(sars)
        for _ in 1:10
            train(StatsBase.sample(sars, 200), 10)
        end
        save_model()
        post_loss = sum(sars_losses(sars)) / length(sars)
        @printf("%4d - Memory length: %4d    Average rewards: %8.2f    Loss: %4.2f -> %4.2f\n",
                i, length(sars), sum(rewards) / length(rewards), pre_loss, post_loss)
    end
end

run()
