module LunarLander3

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
        Dense(hidden_layer_size, 4, identity),
        softmax)
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

function loss(p_model, sars)
    # This formula is not intuitive to me, be watchful of bugs with this formula.
    -sum(
        map(sars) do sars
            sars.q * log(p_model(sars.s)[sars.a + 1])
        end
    )
end

function run_episodes(n_episodes, policy; render_env=true, discount_factor=0.9)

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
        episode_reward = run_episode(env, policy) do (s, a, r, s_next)
            push!(episode_sars, (convert(Array, s), a, r, convert(Array, s_next), finished(env)))
            if render_env
                render(env)
            end
        end
        push!(episode_rewards, episode_reward)
        append!(all_sars, add_q_to_sars(episode_sars))
    end
    all_sars, episode_rewards
end

const default_optimizer = NADAM()

function train(p_model, sars, epochs, optimizer=default_optimizer)
    for epoch in 1:epochs
        Flux.train!((sars) -> loss(p_model, sars), Flux.params(p_model), [(sars,)], optimizer)
    end
end

truncate(a, n) = a[1:min(n, end)]

function run()
    p_model = make_p_model()
    policy = Policy(p_model)
    rewards = Float32[]
    losses = Float32[]
    for cycle in 1:30
        @printf("%4d - ", cycle)
        sars, new_rewards = run_episodes(10, policy)
        rewards = truncate(vcat(new_rewards, rewards), 100)
        pre_training_loss = loss(p_model, sars)
        pushfirst!(losses, pre_training_loss.data)
        losses = truncate(losses, 100)
        train(p_model, sars, 10)
        post_training_loss = loss(p_model, sars)
        @printf(
            "Average Rewards: %8.3f    Loss: %8.3f -> %8.3f    Average Loss: %8.3f\n",
            mean(rewards),
            pre_training_loss / 1_000,
            post_training_loss / 1_000,
            mean(losses) / 1_000)
    end
end

end  # end module

if !isinteractive()
    LunarLander3.run()
end
