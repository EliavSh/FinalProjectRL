class QValues:

    @staticmethod
    def get_current(policy_net, states, actions):
        policy_net_actions = policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
        return policy_net_actions.squeeze(1)  # back to 1-D vector

    @staticmethod
    def get_next(target_net, next_states, policy_net):
        online_network_best_actions = policy_net(next_states).max(dim=1)[1]  # the output of .max is tuple of (data, indexes) so we take [1]
        values = target_net(next_states).gather(dim=1, index=online_network_best_actions.unsqueeze(-1))
        return values.squeeze(1)  # back to 1-D vector


"""
values = values.detach().numpy()
online_network_best_actions = online_network_best_actions.detach().numpy()
target_next_states = target_net(next_states).detach().numpy()
"""
"""
    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values
"""
