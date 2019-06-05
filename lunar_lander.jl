using Flux
using OpenAIGym
using Printf: @printf
using PyCall

const env = GymEnv(:LunarLander, :v2)

const actions = 0:3

const hidden_layer_size = 100

const q_model = Chain(
    Dense(8 + 1, hidden_layer_size, relu),
    Dense(hidden_layer_size, 1))

loss(x, y) = Flux.mse(q_model(x), y) + 0.001 * sum(StatsBase.norm, Flux.params(q_model))

optimizer = NADAM()

function train(epochs)
    x = to_q_x(sars)
    y = to_q_y(sars)
    @printf(
        "Epoch %4d / %4d - Loss: %8.4f\n",
        0,
        epochs,
        loss(x, y).data)
    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(q_model), [(x, y)], optimizer)
        @printf(
            "Epoch %4d / %4d - Loss: %8.4f\n",
            epoch,
            epochs,
            loss(x, y).data)
    end
end

const sars = Tuple{PyCall.PyArray{Float32,1},Int64,Float64,PyCall.PyArray{Float32,1}}[]

function run_episodes(n_episodes)
    rewards = Float64[]
    for episode in 1:n_episodes
        reward = run_episode(env, RandomPolicy()) do (s, a, r, s_next)
            push!(sars, (s, a, r, s_next))
            render(env)
        end
        push!(rewards, reward)
    end
    close(env)
    rewards
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

function V(s)
    action_values = map(actions) do a
        q_model(to_q_x([(s, a)]))[1].data
    end
    maximum(action_values)
end

function to_q_y(sars::AbstractArray{Tuple{PyCall.PyArray{Float32,1},Int64,Float64,PyCall.PyArray{Float32,1}}})
    to_q_y((r, s_next) for (_, _, r, s_next) in sars)
end

function to_q_y(rss)
    map(rss) do (r, s_next)
        r + 0.9 * V(s_next)
    end
end
