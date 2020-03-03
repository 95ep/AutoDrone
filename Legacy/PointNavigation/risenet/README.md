<h1>RISENet</h1>

Useful tools can be found in <code>tools.py</code>.

<h2>Example use:</h2>

    import risenet.tools as rsn
    
    actor_critic = rsn.neural_agent(rgb=False)
    rsn.load_pretrained_weights(actor_critic, 'PATH TO WEIGHTS')
    dim_actions = 9
    rsn.change_action_dim(actor_critic, dim_actions)
