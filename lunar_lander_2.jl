module LunarLander2

using BSON: @load, @save
using Flux
using Logging
using OpenAIGym
using Printf
using StatsBase
import Reinforce: action

const env = GymEnv(:LunarLander, :v2)

const actions = 0:3

const hidden_layer_size = 200

function make_q_model()
    Chain(
        Dense(8 + 1, hidden_layer_size, leakyrelu),
        Dense(hidden_layer_size, hidden_layer_size, leakyrelu),
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

struct Policy <: AbstractPolicy
    q_model
    epsilon
end

function action(policy::Policy, r, s, A)
    if rand() < policy.epsilon
        rand(A)
    else
        argmax(action_values(policy.q_model, s)) - 1
    end
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

function run_episodes(n_episodes, policy; render_env=true)
    sarsf = SARSF[]
    episode_rewards = Float64[]
    for episode in 1:n_episodes
        episode_reward = run_episode(env, policy) do (s, a, r, s_next)
            push!(sarsf, SARSF(s, a, r, s_next, finished(env)))
            if render_env
                render(env)
            end
        end
        push!(episode_rewards, episode_reward)
    end
    sarsf, episode_rewards
end

truncate(a, n) = a[1:min(n, end)]

const q_model_file = "q_model.bson"

function run(log=false, render_env=false)
    if isfile(q_model_file)
        @load q_model_file q_model
    else
        q_model = make_q_model()
    end
    optimizer = NADAM()
    policy = Policy(q_model, 0.2)
    memory_size = 20_000
    memory = SARSF[]
    episode_rewards_size = 100
    episode_rewards = Float64[]
    losses_size = 100
    losses = Float64[]
    training_sample_size = 200
    episodes_per_cycle = 1
    training_epochs_per_cycle = 100
    learning_cycles = 3_000
    for learning_cycle in 1:learning_cycles
        learning_cycle_output = @sprintf("%4d - ", learning_cycle)
        print(learning_cycle_output)
        new_sarsf, new_rewards = run_episodes(episodes_per_cycle, policy, render_env=render_env)
        memory = truncate(vcat(new_sarsf, memory), memory_size)
        episode_rewards = truncate(vcat(new_rewards, episode_rewards), episode_rewards_size)
        training_sample = sample(memory, training_sample_size)
        training_sample_x = to_x(training_sample)
        pre_training_loss = loss(q_model, training_sample_x, to_y(q_model, training_sample))
        pushfirst!(losses, pre_training_loss.data)
        losses = truncate(losses, losses_size)
        train(q_model, optimizer, training_sample, training_epochs_per_cycle)
        @save q_model_file q_model
        post_training_loss = loss(q_model, training_sample_x, to_y(q_model, training_sample))
        metrics_output = @sprintf(
            "Average reward: %8.3f    Memory Length: %5d    Loss: %6.3f -> %6.3f    Average loss: %6.3f",
            mean(episode_rewards),
            length(memory),
            pre_training_loss / 1_000,
            post_training_loss / 1_000,
            mean(losses) / 1_000)
        println(metrics_output)
        if log
            @info learning_cycle_output * metrics_output
        end
    end
end

end  # module end

using ArgParse
using Logging
using Flux  # This needs to be here to work around a BSON bug.
if !isinteractive()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--render"
            action = :store_true
        "log_file"
            required = false
    end
    args = parse_args(ARGS, settings)
    if !isnothing(args["log_file"])
        global_logger(SimpleLogger(open(args["log_file"], "a"), Logging.Debug))
    else
        global_logger(NullLogger())
    end
    LunarLander2.run(true, args["render"])
end
