import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, normal
from torch.utils.data.sampler import *
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse

# from supplychain_env import SupplyChain
from network_env import SupplyChain
import wandb

##################################################
def args_fn():
    parser = argparse.ArgumentParser(
        "Hyperparameters Setting for MAPPO in MPE environment"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=int(4e5),
        help=" Maximum number of training steps",
    )
    parser.add_argument(
        "--episode_limit",
        type=int,
        default=30,
        help="Maximum number of steps per episode",
    )
    parser.add_argument(
        "--evaluate_freq",
        type=float,
        default=5000,
        help="Evaluate the policy every 'evaluate_freq' steps",
    )
    parser.add_argument(
        "--evaluate_times", type=float, default=3, help="Evaluate times"
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (the number of episodes)"
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=8,
        help="Minibatch size (the number of episodes)",
    )
    parser.add_argument(
        "--rnn_hidden_dim",
        type=int,
        default=64,
        help="The number of neurons in hidden layers of the rnn",
    )
    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=64,
        help="The number of neurons in hidden layers of the mlp",
    )
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument(
        "--use_adv_norm",
        type=bool,
        default=True,
        help="Trick 1:advantage normalization",
    )
    parser.add_argument(
        "--use_reward_norm",
        type=bool,
        default=False,
        help="Trick 3:reward normalization",
    )
    parser.add_argument(
        "--use_reward_scaling",
        type=bool,
        default=True,
        help="Trick 4:reward scaling. Here, we do not use it.",
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=0.0001, help="Trick 5: policy entropy"
    )
    parser.add_argument(
        "--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay"
    )
    parser.add_argument(
        "--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip"
    )
    parser.add_argument(
        "--use_orthogonal_init",
        type=bool,
        default=True,
        help="Trick 8: orthogonal initialization",
    )
    parser.add_argument(
        "--set_adam_eps",
        type=float,
        default=True,
        help="Trick 9: set Adam epsilon=1e-5",
    )
    parser.add_argument(
        "--use_relu",
        type=float,
        default=False,
        help="Whether to use relu, if False, we will use tanh",
    )
    parser.add_argument(
        "--use_rnn", type=bool, default=False, help="Whether to use RNN"
    )
    parser.add_argument(
        "--add_agent_id",
        type=float,
        default=False,
        help="Whether to add agent_id. Here, we do not use it.",
    )
    parser.add_argument(
        "--use_value_clip", type=float, default=True, help="Whether to use value clip."
    )

    args = parser.parse_args()
    return args


###############################
# replay buffer


class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size
        self.a_dim = 9
        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()
        # create a buffer (dictionary)

    def reset_buffer(self):
        self.buffer = {
            "obs_n": np.empty(
                [self.batch_size, self.episode_limit, self.N, self.obs_dim]
            ),
            "s": np.empty([self.batch_size, self.episode_limit, self.state_dim]),
            "v_n": np.empty([self.batch_size, self.episode_limit + 1, self.N]),
            "a_n": np.empty([self.batch_size, self.episode_limit, self.N]),
            "a_logprob_n": np.empty([self.batch_size, self.episode_limit, self.N]),
            "r_n": np.empty([self.batch_size, self.episode_limit, self.N]),
            "done_n": np.empty([self.batch_size, self.episode_limit, self.N]),
        }
        self.episode_num = 0

    def store_transition(
        self, episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n
    ):
        self.buffer["obs_n"][self.episode_num][episode_step] = obs_n
        self.buffer["s"][self.episode_num][episode_step] = s
        self.buffer["v_n"][self.episode_num][episode_step] = v_n
        self.buffer["a_n"][self.episode_num][episode_step] = a_n
        self.buffer["a_logprob_n"][self.episode_num][episode_step] = a_logprob_n
        self.buffer["r_n"][self.episode_num][episode_step] = r_n
        self.buffer["done_n"][self.episode_num][episode_step] = done_n

    def store_last_value(self, episode_step, v_n):
        self.buffer["v_n"][self.episode_num][episode_step] = v_n
        self.episode_num += 1

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == "a_n":
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch


###################################
# normalization


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


############################################
# modules

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        prob = torch.softmax(self.fc2(self.rnn_hidden), dim=-1)
        return prob


class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


"""
class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, episode_limit, N, actor_input_dim), prob.shape(mini_batch_size, episode_limit, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        prob = torch.softmax(self.fc3(x), dim=-1)
        return prob
"""


class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.mu = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mu, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, episode_limit, N, actor_input_dim), prob.shape(mini_batch_size, episode_limit, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        mu = self.mu(x)
        std = self.log_std.exp()
        mu = torch.squeeze(mu)
        dist = normal.Normal(mu, std)

        return dist


class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, episode_limit, N, critic_input_dim), value.shape=(mini_batch_size, episode_limit, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value


class MAPPO_MPE:
    def __init__(self, args):
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_value_clip = args.use_value_clip

        # get the input dimension of actor and critic
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N

        if self.use_rnn:
            print("------use rnn------")
            self.actor = Actor_RNN(args, self.actor_input_dim)
            self.critic = Critic_RNN(args, self.critic_input_dim)
        else:
            self.actor = Actor_MLP(args, self.actor_input_dim)
            self.critic = Critic_MLP(args, self.critic_input_dim)

        self.ac_parameters = list(self.actor.parameters()) + list(
            self.critic.parameters()
        )
        if self.set_adam_eps:
            print("------set adam eps------")
            self.ac_optimizer = torch.optim.Adam(
                self.ac_parameters, lr=self.lr, eps=1e-5
            )
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)

        wandb.watch(self.actor, log_freq=50)
        wandb.watch(self.critic, log_freq=50)

    def choose_action(self, obs_n, evaluate):
        with torch.no_grad():
            actor_inputs = []
            obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim)
            actor_inputs.append(obs_n)
            if self.add_agent_id:
                """
                Add an one-hot vector to represent the agent_id
                For example, if N=3
                [obs of agent_1]+[1,0,0]
                [obs of agent_2]+[0,1,0]
                [obs of agent_3]+[0,0,1]
                So, we need to concatenate a N*N unit matrix(torch.eye(N))
                """
                actor_inputs.append(torch.eye(self.N))

            actor_inputs = torch.cat(
                [x for x in actor_inputs], dim=-1
            )  # actor_input.shape=(N, actor_input_dim)
            dist = self.actor(actor_inputs)  # prob.shape=(N,action_dim)
            if (
                evaluate
            ):  # When evaluating the policy, we select the action with the highest probability
                a_n = dist.sample()  # prob.argmax(dim=-1)
                return a_n.numpy(), None
            else:
                a_n = dist.sample()
                a_logprob_n = dist.log_prob(a_n)
                return a_n.numpy(), a_logprob_n.numpy()

    def get_value(self, s):
        with torch.no_grad():
            critic_inputs = []
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            s = (
                torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)
            )  # (state_dim,)-->(N,state_dim)
            critic_inputs.append(s)
            if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
                critic_inputs.append(torch.eye(self.N))
            critic_inputs = torch.cat(
                [x for x in critic_inputs], dim=-1
            )  # critic_input.shape=(N, critic_input_dim)
            v_n = self.critic(critic_inputs)  # v_n.shape(N,1)
            return v_n.numpy().flatten()

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()  # get training data

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and td_target have no gradient
            deltas = (
                batch["r_n"]
                + self.gamma * batch["v_n"][:, 1:] * (1 - batch["done_n"])
                - batch["v_n"][:, :-1]
            )  # deltas.shape=(batch_size,episode_limit,N)
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,episode_limit,N)
            v_target = (
                adv + batch["v_n"][:, :-1]
            )  # v_target.shape(batch_size,episode_limit,N)
            if self.use_adv_norm:  # Trick 1: advantage normalization
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        actor_inputs, critic_inputs = self.get_inputs(batch)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(
                SequentialSampler(range(self.batch_size)), self.mini_batch_size, False
            ):
                """
                get probs_now and values_now
                probs_now.shape=(mini_batch_size, episode_limit, N, action_dim)
                values_now.shape=(mini_batch_size, episode_limit, N)
                """
                if self.use_rnn:
                    # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                    self.actor.rnn_hidden = None
                    self.critic.rnn_hidden = None
                    probs_now, values_now = [], []
                    for t in range(self.episode_limit):
                        prob = self.actor(
                            actor_inputs[index, t].reshape(
                                self.mini_batch_size * self.N, -1
                            )
                        )  # prob.shape=(mini_batch_size*N, action_dim)
                        probs_now.append(
                            prob.reshape(self.mini_batch_size, self.N, -1)
                        )  # prob.shape=(mini_batch_size,N,action_dim）
                        v = self.critic(
                            critic_inputs[index, t].reshape(
                                self.mini_batch_size * self.N, -1
                            )
                        )  # v.shape=(mini_batch_size*N,1)
                        values_now.append(
                            v.reshape(self.mini_batch_size, self.N)
                        )  # v.shape=(mini_batch_size,N)
                    # Stack them according to the time (dim=1)
                    probs_now = torch.stack(probs_now, dim=1)
                    values_now = torch.stack(values_now, dim=1)
                else:
                    dist_now = self.actor(actor_inputs[index])
                    values_now = self.critic(critic_inputs[index]).squeeze(-1)

                dist_entropy = (
                    dist_now.entropy()
                )  # dist_entropy.shape=(mini_batch_size, episode_limit, N)
                # batch['a_n'][index].shape=(mini_batch_size, episode_limit, N)
                a_logprob_n_now = dist_now.log_prob(
                    batch["a_n"][index]
                )  # a_logprob_n_now.shape=(mini_batch_size, episode_limit, N)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(
                    a_logprob_n_now - batch["a_logprob_n"][index].detach()
                )  # ratios.shape=(mini_batch_size, episode_limit, N)
                surr1 = ratios * adv[index]
                surr2 = (
                    torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                )
                # actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                actor_loss = torch.max(-surr1, -surr2).mean()
                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = (
                        torch.clamp(
                            values_now - values_old, -self.epsilon, self.epsilon
                        )
                        + values_old
                        - v_target[index]
                    )
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(
                        values_error_clip**2, values_error_original**2
                    )
                else:
                    critic_loss = (values_now - v_target[index]) ** 2

                self.ac_optimizer.zero_grad()

                entropy_loss = dist_entropy.mean()
                critic_loss = critic_loss.mean()
                # added in vf_coef was not there before
                ac_loss = (
                    actor_loss - self.entropy_coef * entropy_loss + 0.5 * critic_loss
                )
                ac_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()
                # maybe returns the losses here
                return ac_loss, critic_loss, actor_loss, entropy_loss
        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p["lr"] = lr_now

    def get_inputs(self, batch):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch["obs_n"])
        critic_inputs.append(batch["s"].unsqueeze(2).repeat(1, 1, self.N, 1))
        if self.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = (
                torch.eye(self.N)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(self.batch_size, self.episode_limit, 1, 1)
            )
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat(
            [x for x in actor_inputs], dim=-1
        )  # actor_inputs.shape=(batch_size, episode_limit, N, actor_input_dim)
        critic_inputs = torch.cat(
            [x for x in critic_inputs], dim=-1
        )  # critic_inputs.shape=(batch_size, episode_limit, N, critic_input_dim)
        return actor_inputs, critic_inputs

    def save_model(self, env_name, number, seed, total_steps):
        path = f"/Users/callum/Documents/Github/ChainRL/Experiments/scripts/multi/model/MAPPO_actor_env_{env_name}_number_{number}_seed_{seed}_step_{int(total_steps)/ 1000}k.pth"
        torch.save(self.actor.state_dict(), path)

    def load_model(self, env_name, number, seed, step):
        self.actor.load_state_dict(
            torch.load(
                "./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(
                    env_name, number, seed, step
                )
            )
        )


#############################################
# main runner


class Runner_MAPPO_MPE:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = SupplyChain()  # Discrete action space
        self.args.N = self.env.num_agents  # The number of agents
        self.args.obs_dim_n = [
            self.env.observation_space[i].shape[0] for i in range(self.args.N)
        ]  # obs dimensions of N agents
        self.args.action_dim_n = [
            self.env.action_space[i].shape[0] for i in range(self.args.N)
        ]  # actions dimensions of N agents
        # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
        self.args.obs_dim = self.args.obs_dim_n[
            0
        ]  # The dimensions of an agent's observation space
        self.args.action_dim = self.args.action_dim_n[
            0
        ]  # The dimensions of an agent's action space
        self.args.state_dim = np.sum(
            self.args.obs_dim_n
        )  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Create N agents
        self.agent_n = MAPPO_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(
            log_dir="runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}".format(
                self.env_name, self.number, self.seed
            )
        )

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(
                shape=self.args.N, gamma=self.args.gamma
            )

    def run(
        self,
    ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_mpe(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                ac_loss, critic_loss, actor_loss, entropy_loss = self.agent_n.train(
                    self.replay_buffer, self.total_steps
                )  # Training
                self.replay_buffer.reset_buffer()

                wandb.log({"total-loss": ac_loss})
                wandb.log({"critic-loss": critic_loss})
                wandb.log({"actor-loss": actor_loss})
                wandb.log({"entropy-loss": entropy_loss})

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(
        self,
    ):
        evaluate_reward = 0
        evaluate_node_profits = {}
        for i in range(self.env.num_nodes):
            evaluate_node_profits[f"node_{i}_profit"] = 0

        for _ in range(self.args.evaluate_times):
            episode_reward, node_rewards, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward
            for i in range(self.env.num_nodes):
                evaluate_node_profits[f"node_{i}_profit"] += node_rewards[i]

        evaluate_reward = evaluate_reward / self.args.evaluate_times

        for i in range(self.env.num_nodes):
            evaluate_node_profits[f"node_{i}_profit"] /= self.args.evaluate_times
        evaluate_node_profits["step"] = self.total_steps

        wandb.log({"train-reward": evaluate_reward, "step": self.total_steps})
        wandb.log(evaluate_node_profits)

        self.evaluate_rewards.append(evaluate_reward)
        print(
            f"step:{self.total_steps}/{args.max_train_steps} \t evaluate_reward:{evaluate_reward}"
        )
        self.writer.add_scalar(
            "evaluate_step_rewards_{}".format(self.env_name),
            evaluate_reward,
            global_step=self.total_steps,
        )
        # Save the rewards and models
        path = f"/Users/callum/Documents/Github/ChainRL/Experiments/scripts/multi/data_train/MAPPO_env_{self.env_name}_number_{self.number}_seed_{self.seed}.npy"

        np.save(path, np.array(self.evaluate_rewards))
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

    def run_episode_mpe(self, evaluate=False):
        episode_reward = 0
        node_profits = np.zeros(self.env.num_nodes)
        obs_n, _ = self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if (
            self.args.use_rnn
        ):  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            a_n, a_logprob_n = self.agent_n.choose_action(
                obs_n, evaluate=evaluate
            )  # Get actions and the corresponding log probabilities of N agents
            s = np.array(
                obs_n
            ).flatten()  # In MPE, global state is the concatenation of all agents' local obs.
            # maybe need to alter s this will have duplicate observation spaces
            v_n = self.agent_n.get_value(s)  # Get the state values (V(s)) of N agents
            a_n = a_n.flatten()
            obs_next_n, r_n, done_n, _, info = self.env.step(a_n)
            episode_reward += info["total_reward"]
            node_profits += self.env.node_profits
            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)

                # Store the transition
                a_logprob_n = a_logprob_n.flatten()
                self.replay_buffer.store_transition(
                    episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n
                )

            obs_n = obs_next_n
            if all(done_n):
                break

        if not evaluate:
            # An episode is over, store v_n in the last step
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return episode_reward, node_profits, episode_step + 1


def sweep():
    """
    method of sweep: grid_search, random_search, bayesian_search
    """
    sweep_config = {"method": "random"}
    metric = {"name": "reward", "goal": "maximize"}
    sweep_config["metric"] = metric

    parameters_dict = {
        "lr": {"values": [1e-4, 1e-3, 1e-2]},
        "gamma": {"values": [0.99, 0.97, 0.95]},
        "lamda": {"values": [0.99, 0.97, 0.95]},
        "epsilon": {"values": [0.1, 0.2, 0.5]},
    }

    parameters_dict.update(
        {
            "max_train_steps": {"value": 1e4},
            "episode_limit": {"value": 30},
            "evaluate_freq": {"value": 5000},
            "evaluate_times": {"value": 3},
            "batch_size": {"value": 32},
            "rnn_hidden_dim": {"value": 64},
            "mlp_hidden_dim": {"value": 64},
            "K_epochs": {"value": 15},
            "use_adv_norm": {"value": True},
            "use_reward_norm": {"value": False},
            "use_reward_scaling": {"value": False},
            "entropy_coef": {"value": 0.0001},
            "use_lr_decay": {"value": False},
            "use_grad_clip": {"value": True},
            "use_orthogonal_init": {"value": True},
            "use_relu": {"value": False},
            "set_adam_eps": {"value": True},
            "set_rnn": {"value": False},
            "add_agent_id": {"value": False},
            "use_value_clip": {"value": True},
            "add_agent_id": {"value": False},
            "add_agent_id": {"value": False},
            "add_agent_id": {"value": False},
        }
    )
    sweep_config["parameters"] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project="MultiChainRL")

    def train(config):
        with wandb.init(config=config):
            config = wandb.config
        runner = Runner_MAPPO_MPE(args, env_name="simple_spread", number=1, seed=0)
        runner.run()

    wandb.agent(sweep_id, train, count=5)


def main(args):

    with wandb.init(project="MultiChainRL", config=args):

        runner = Runner_MAPPO_MPE(args, env_name="simple_spread", number=1, seed=0)
        runner.run()


if __name__ == "__main__":
    args = args_fn()
    main(args)
