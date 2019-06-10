using Flux

module LunarLander3

using BSON: @save, @load
using Flux
using OpenAIGym
using Printf
import Reinforce: action

const env = GymEnv(:LunarLander, :v2)

const actions = 0:3

function make_p_model(hidden_layer_size=100)
    Chain(
        Dense(8, hidden_layer_size, leakyrelu),
        Dense(hidden_layer_size, hidden_layer_size, leakyrelu),
        Dense(hidden_layer_size, hidden_layer_size, leakyrelu),
        Dense(hidden_layer_size, 4, identity),
        softmax)
end

function make_v_model(hidden_layer_size=100)
    Chain(
        Dense(8, hidden_layer_size, leakyrelu),
        Dense(hidden_layer_size, hidden_layer_size, leakyrelu),
        Dense(hidden_layer_size, hidden_layer_size, leakyrelu),
        Dense(hidden_layer_size, 1, identity))
end

struct Policy <: AbstractPolicy
    p_model
end

function action(policy::Policy, r, s, A)
    sample(actions, Weights(policy.p_model(s)))
end

struct SARS
    s :: Array{Float32, 1}
    a :: Int8
    r :: Float32
    q :: Float32
    s_next :: Array{Float32, 1}
    f :: Bool
end

function loss(p_model, v_model, sars; entropy_bonus=1.0)
    -sum(
        map(sars) do sars
            (sars.q - v_model(sars.s)[1]) * log(p_model(sars.s)[sars.a + 1]) + entropy_bonus * entropy(p_model(sars.s))
        end
    )
end

function run_episodes(n_episodes, policy; render_env=true, discount_factor=0.997)

    function add_q_to_sars(sars)
        sars_with_q = SARS[]
        for i in 1:length(sars)
            q = 0
            for j in Iterators.countfrom(0)
                @assert j < length(sars)
                q += discount_factor ^ j * sars[i + j][3]
                if sars[i + j][5]
                    break
                end
            end
            push!(sars_with_q, SARS(sars[i][1], sars[i][2], sars[i][3], q, sars[i][4], sars[i][5]))
        end
        sars_with_q
    end

    all_sars = SARS[]
    episode_rewards = Float32[]
    for episode in 1:n_episodes
        episode_sars = []
        run_episode(env, policy) do (s, a, r, s_next)
            push!(episode_sars, (convert(Array, s), a, r, convert(Array, s_next), finished(env)))
            if render_env
                render(env)
            end
        end
        @assert !any(sars[5] for sars in episode_sars[1:end-1])
        @assert episode_sars[end][5]
        push!(episode_rewards, sum(r for (_, _, r, _, _) in episode_sars))
        append!(all_sars, add_q_to_sars(episode_sars))
    end
    all_sars, episode_rewards
end

const default_p_model_optimizer = NADAM()

function train_p_model(p_model, v_model, sars, epochs, optimizer=default_p_model_optimizer)
    for epoch in 1:epochs
        Flux.train!((sars) -> loss(p_model, v_model, sars), Flux.params(p_model), [(sars,)], optimizer)
    end
end

const default_v_model_optimizer = NADAM()

function train_v_model(v_model, sars, epochs, optimizer=default_v_model_optimizer)

    loss(x, y) = Flux.mse(v_model(x), y)

    for epoch in 1:epochs
        s = sample(sars, 1000)
        x = hcat((sars.s for sars in s)...)
        y = collect(sars.q for sars in s)
        Flux.train!(loss, Flux.params(v_model), [(x, y)], optimizer)
    end
    return loss(hcat((sars.s for sars in sars)...), collect(sars.q for sars in sars))
end

truncate(a, n) = a[1:min(n, end)]

function run()
    if isfile("model.bson")
        @load "model.bson" p_model
    else
        p_model = make_p_model()
    end
    v_model = make_v_model()
    policy = Policy(p_model)
    memory = SARS[]
    rewards = Float32[]
    losses = Float32[]
    for cycle in 1:3_000
        @printf("%4d - ", cycle)
        new_sars, new_rewards = run_episodes(1, policy, render_env=false)
        memory = truncate(vcat(new_sars, memory), 15_000)
        rewards = truncate(vcat(new_rewards, rewards), 100)
        v_loss = train_v_model(v_model, memory, 10)
        pre_training_loss = loss(p_model, v_model, new_sars)
        pushfirst!(losses, pre_training_loss.data)
        losses = truncate(losses, 100)
        train_p_model(p_model, v_model, new_sars, 1)
        @save "model.bson" p_model
        post_training_loss = loss(p_model, v_model, new_sars)
        @printf(
            "Average Rewards: %10.3f    Loss: %8.3f -> %8.3f    Average Loss: %8.3f    Memory: %8d    V-Loss: %8.3f\n",
            mean(rewards),
            pre_training_loss / 1_000,
            post_training_loss / 1_000,
            mean(losses) / 1_000,
            length(memory),
            v_loss / 1_000)
    end
end

end  # end module

if !isinteractive()
    LunarLander3.run()
end
