from typing import Sequence, Optional, List, Union, Any, Type
from scipy.stats import poisson

import numpy as np
import pandas as pd
import os
import gymnasium as gym
from gymnasium.spaces import Box, Dict


class SupplyChain(gym.Env):
    def __init__(self, setup_name="test"):

        node_params = {
            "nodes": np.int32,
            "node_inv_init": np.float32,
            "node_prod_cost": np.float32,
            "node_prod_time": np.int32,
            "node_hold_cost": np.float32,
            "node_prod_capacity": np.float32,
            "node_inv_capacity": np.float32,
            "node_type": str,
            "node_yield": np.float32,
            "node_operating_cost": np.float32,
        }
        edge_params = {
            "edge_sender": np.int32,
            "edge_reciever": np.int32,
            "edge_trans_time": np.int32,
            "edge_cost": np.float32,
            "edge_hold_cost": np.float32,
            "edge_prod_yield": np.float32,
        }

        self.node_df = pd.read_csv(
            f"/Users/callum/Documents/Github/ChainRL/Experiments/scripts/multi/nodes1.csv"
        )
        self.edge_df = pd.read_csv(
            f"/Users/callum/Documents/Github/ChainRL/Experiments/scripts/multi/edges1.csv"
        )

        for param_name, param_type in node_params.items():
            setattr(
                self, param_name, self.node_df[param_name].to_numpy(dtype=param_type)
            )

        for param_name, param_type in edge_params.items():
            setattr(
                self, param_name, self.edge_df[param_name].to_numpy(dtype=param_type)
            )

        self.edges = np.vstack((self.edge_sender, self.edge_reciever))
        self.num_nodes = self.nodes.shape[0]
        self.num_edges = self.edges.shape[1]
        self.agents = self.edge_reciever
        self.num_agents = len(self.agents)
        self.node_look_ahead_time = 8
        self.edge_look_ahead_time = 16
        self.time = 0
        self.max_time_length = int(30)  # 1e5

        # reorder links

        # Define the nodes that will serve demand and the demand time series for each demand node

        self.demand_nodes = np.array([0])
        self.market = []
        self.distrib = []
        self.retail = []
        self.factory = []
        self.rawmat = []
        for i in range(self.num_nodes):
            node = self.node_type[i].split("-")
            if "market" in node:
                self.market.append(i)
            if "distrib" in node:
                self.distrib.append(i)
            if "retail" in node:
                self.retail.append(i)
            if "factory" in node:
                self.factory.append(i)
            if "rawmat" in node:
                self.rawmat.append(i)

        self.main_nodes = np.sort(self.distrib + self.factory)

        self.num_markets = len(self.market)
        self.num_retailers = len(self.retail)
        self.retail_links = [0]
        self.demand = np.random.poisson(
            lam=10, size=(len(self.demand_nodes), self.max_time_length)
        )
        self.demand_sale_price = np.array([2.2])

        self.demand_fns = {
            0: poisson(
                10
                + 10 * np.sin(2 * self.time * np.pi / 7)
                + 40 * np.sin(2 * self.time * np.pi / 365)
            )
        }

        # Define the nodes that will recieve raw supplied and the supply time series for each supply node
        # we have defined the last two nodes to be supply nodes
        self.supply_nodes = np.unique(self.edge_sender)[-2:]
        self.supply = np.random.poisson(
            lam=10, size=(len(self.supply_nodes), self.max_time_length)
        )
        self.supply_buy_price = np.array([1.0, 1.0])

        # Define the action and observation spaces for stable baselines

        self.share_reward = True  # whether or not agents take total reward
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0

        # node and edge feature arrays
        edge_array = self.edge_df.values
        self.edge_costs = np.zeros((self.num_nodes, self.num_nodes))
        self.edge_hold_costs = np.zeros((self.num_nodes, self.num_nodes))
        self.edge_backlog_costs = np.zeros((self.num_nodes, self.num_nodes))
        self.edge_lead_times = np.zeros((self.num_nodes, self.num_nodes))
        self.reorder_links = []
        for row in edge_array:
            i = int(row[0])
            j = int(row[1])
            self.edge_costs[i, j] = row[3]
            self.edge_hold_costs[i, j] = row[4]
            self.edge_lead_times[i, j] = row[2]
            self.edge_backlog_costs[i, j] = row[0]  # need to add index

            if row[2] >= 0:
                self.reorder_links.append((i, j))

        self.num_agents = len(self.reorder_links)

        #####################
        # stuff from or-gym
        self.backlog = True
        self.alpha = 1.0  # this is like inflation
        self.seed = 0
        self.user_D = {(1, 0): np.zeros(self.max_time_length)}
        # need to set market, distrib, retail, factor, rawmat, main_nodes
        self.lt_max = np.amax(self.edge_lead_times)

        ######################

        for idx in range(self.num_agents):
            self.action_space.append(
                Box(low=0.0, high=np.inf, shape=(1,), dtype=np.int32)
            )
            self.observation_space.append(
                Box(
                    low=0.0,
                    high=np.inf,
                    shape=(int(self.lt_max + 1),),
                    dtype=np.int32,
                )
            )
            share_obs_dim += int(self.lt_max + 1)

        self.share_observation_space = [
            Box(
                low=0.0,
                high=np.inf,
                shape=(share_obs_dim,),
                dtype=np.int32,
            )
            for _ in range(self.num_agents)
        ]

    def get_state_vector(self) -> np.ndarray:
        # demand_state = self.demand[:, self.time]  # how does the demand enter the chain?
        node_state = self.nodes_inv
        edge_state = self.edges_inv
        state_vector = np.zeros((self.num_agents, self.observation_space[0].shape[0]))
        for i, id in enumerate(self.agents):
            state_vector[i, :] = np.concatenate([node_state[id], edge_state[id]])
        return state_vector

    def update_state(self):
        # first we shall construct the shared state
        # demand = np.hstack([self.demands[d, self.time] for d in self.retail_links])
        demand = self.demands[0, self.time]
        inventory = np.hstack([self.nodes_inv[n, self.time] for n in self.main_nodes])

        self.state = np.zeros((self.num_agents, int(self.lt_max + 1))).astype(np.int32)
        idxs = np.sort([key[1] - 1 for i, key in enumerate(self.reorder_links)])
        self.state[:, 0] = [inventory[i] for i in idxs]
        # add in inv to state here
        if self.time == 0:
            # posbbily indexed incorrectly
            pipeline = [
                [self.edges_inv[v[0], v[1], 0]]
                for i, v in enumerate(self.reorder_links)
            ]
        else:
            pipeline = []
            for i, v in enumerate(self.reorder_links):
                sender = v[0]
                reciever = v[1]
                lead_time = self.edge_lead_times[sender, reciever]
                pipeline.append(
                    self.edges_inv[
                        sender,
                        reciever,
                        int(max(self.time - lead_time, 0)) : int(self.time),
                    ]
                )
        full_pipeline = []

        lead_times = [
            self.edge_lead_times[v[0], v[1]] for i, v in enumerate(self.reorder_links)
        ]

        for i, (p, v) in enumerate(zip(pipeline, lead_times)):
            v = int(v)
            if v == 0:
                continue
            if len(p) <= v:
                pipe = np.zeros(v)
                pipe[-len(p) :] += p
            full_pipeline.append(pipe)
            self.state[i, 1 : len(pipe) + 1] = pipe
        full_pipeline = np.hstack(full_pipeline)

        self.shared_state = np.hstack([demand, inventory, full_pipeline])

    def reset(self, seed=None):

        self.nodes_inv = np.zeros((self.num_nodes, self.max_time_length + 1))
        self.edges_inv = np.zeros(
            (self.num_edges, self.num_edges, self.max_time_length + 1)
        )
        self.replenishment_orders = np.zeros(
            (self.num_edges, self.num_edges, self.max_time_length)
        )
        self.units_sold = np.zeros(
            (self.num_nodes, self.num_nodes, self.max_time_length)
        )
        self.demands = np.zeros((1, self.max_time_length))
        self.unfufilled_demand = np.zeros(
            (self.num_nodes, self.num_nodes, self.max_time_length)
        )
        self.node_profits = np.zeros((self.num_nodes, self.max_time_length))

        self.time = 0

        for id in self.main_nodes:
            self.nodes_inv[id, 0] = self.node_inv_init[id]

        self.update_state()
        return self.state, {}

    def step(self, actions):
        self.node_profits = np.zeros(shape=(self.num_nodes))
        negative_flags = actions < 0
        actions[negative_flags] = 0
        actions.astype(np.int32)

        # place orders
        for i in range(actions.shape[0]):
            # Get the nodes that are going to do the order
            request = actions[i]
            # supplier = self.edges[:, i][0]
            # purchaser = self.edges[:, i][1]
            supplier = self.reorder_links[i][0]
            purchaser = self.reorder_links[i][1]

            if supplier in self.rawmat:
                # have unlimited stock
                self.replenishment_orders[supplier, purchaser, self.time] = request
                self.units_sold[supplier, purchaser, self.time] = request

            elif supplier in self.distrib:
                supplier_stock = self.nodes_inv[supplier, self.time]
                self.replenishment_orders[supplier, purchaser, self.time] = min(
                    request, supplier_stock
                )
                self.units_sold[supplier, purchaser, self.time] = min(
                    request, supplier_stock
                )

            elif supplier in self.factory:
                # check c is okay
                c = self.node_prod_capacity[supplier]
                v = self.node_yield[supplier]
                supplier_stock = self.nodes_inv[supplier, self.time]
                self.replenishment_orders[supplier, purchaser, self.time] = min(
                    request, c, v * supplier_stock
                )

                self.units_sold[supplier, purchaser, self.time] = min(
                    request, c, v * supplier_stock
                )

        # recieve deliveries and update inventories
        # may need to update so 0, 6, 7 are not included
        for id in self.main_nodes:
            # update pipeline inventories
            incoming = []
            link_idxs = np.where(self.edge_reciever == id)[0]
            reciever_idxs = np.where(self.edge_sender == id)[0]
            for j, jd in enumerate(self.edge_sender[link_idxs]):
                lead_time = self.edge_trans_time[link_idxs[j]]
                if self.time - lead_time >= 0:
                    delivery = self.replenishment_orders[jd, id, self.time - lead_time]
                else:
                    delivery = 0
                incoming += [delivery]
                self.edges_inv[jd, id, self.time + 1] = (
                    self.edges_inv[jd, id, self.time]
                    - delivery
                    + self.replenishment_orders[jd, id, self.time]
                )
            # update on-hand inventories
            # sort out v at some point
            v = 1
            outgoing = (
                1
                / v
                * np.sum(
                    self.units_sold[id, k, self.time]
                    for k in self.edge_reciever[reciever_idxs]
                )
            )
            self.nodes_inv[id, self.time + 1] = (
                self.nodes_inv[id, self.time] + np.sum(incoming) - outgoing
            )

        # demand is realized
        for id in self.retail:
            for jd in self.market:
                demand = 10  # np.random.poisson(lam=10.0)
                self.demands[0, self.time] = demand
                if self.backlog and self.time >= 1:
                    demand += self.unfufilled_demand[id, jd, self.time - 1]
                    self.demands[0, self.time] = demand

                retail_inv = self.nodes_inv[id, self.time + 1]
                self.units_sold[id, jd, self.time] = min(demand, retail_inv)
                self.nodes_inv[id, self.time + 1] -= self.units_sold[id, jd, self.time]
                self.unfufilled_demand[id, jd, self.time] = (
                    demand - self.units_sold[id, jd, self.time]
                )

        # calculate profit
        # this is for main nodes again we should define this
        a = self.alpha
        for id in self.main_nodes:
            pc = 0
            sr = 0
            hc = 0
            oc = 0
            up = 0
            for jd in range(self.num_nodes):
                sr += self.edge_costs[id, jd] * self.units_sold[id, jd, self.time]
                # is replnishment orders ids correct way arounx
                pc += (
                    self.edge_costs[jd, id]
                    * self.replenishment_orders[jd, id, self.time]
                )
            if id not in self.rawmat:
                # this implies purchaser also pays for pipeline holding costs?
                node_holding_cost = (
                    self.node_hold_cost[id] * self.nodes_inv[id, self.time + 1]
                )
                individual_edge_costs = [
                    self.edge_hold_costs[k, id] * self.edges_inv[k, id, self.time + 1]
                    for k in range(self.num_nodes)
                ]
                edge_holding_cost = np.sum(individual_edge_costs)
                hc += node_holding_cost + edge_holding_cost
            if id in self.factory:
                oc += (
                    self.node_operating_cost[id]
                    / self.node_yield[id]
                    * np.sum([self.units_sold[id, k] for k in range(self.num_nodes)])
                )

            if id in self.retail:
                up += np.sum(
                    [
                        self.edge_backlog_costs[id, k]
                        * self.unfufilled_demand[id, k, self.time]
                        for k in range(self.num_nodes)
                    ]
                )
            self.node_profits[id] = a**self.time * (sr - pc - oc - hc - up)

        total_profit = np.sum(self.node_profits)
        self.time += 1
        if self.time >= self.max_time_length:
            done = True
        else:
            done = False
            self.update_state()

        rewards = np.zeros(self.num_agents)

        dones = np.zeros(self.num_agents)
        idxs = np.sort([key[1] - 1 for i, key in enumerate(self.reorder_links)])
        for i, id in enumerate(idxs):
            rewards[i] = self.node_profits[id]

        truncate = done
        info = {"total_reward": total_profit}

        return self.state, rewards, dones, truncate, info


env = SupplyChain()
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10])
env.reset()
rewards = []
tot_reward = []
for i in range(10):
    o, r, d, _, i = env.step(a)
    rewards.append(r)
    tot_reward.append(i["total_reward"])

print(tot_reward)
