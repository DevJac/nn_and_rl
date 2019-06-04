using OpenAIGym

const env = GymEnv(:LunarLander, :v2)

function run_episodes(n_episodes)
    rewards = Float64[]
    for episode in 1:n_episodes
        reward = run_episode(env, RandomPolicy()) do (s, a, r, s_next)
            println((s, a, r, s_next))
            render(env)
        end
        push!(rewards, reward)
    end
    close(env)
    rewards
end
