"""
  Copyright 2020 ETH Zurich, Secure, Reliable, and Intelligent Systems Lab

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""


import math
from itertools import product

import torch
from gurobipy import GRB, Model, quicksum

from spatial.t_norm_transformer import TNormTransformer
from spatial.interpolation import interpolate


class TInfNormTransformer(TNormTransformer):

    def add_norm_constraints(self, model, vx, vy):
        pass

    def compute_candidates(self):

        assert not self.candidates[0]

        radius = math.ceil(self.delta)

        for row, col in product(range(-radius, radius), repeat=2):

            lb_row, ub_row = row, row + 1
            lb_col, ub_col = col, col + 1

            if row == -radius:
                lb_row = -self.delta

            if row == radius - 1:
                ub_row = self.delta

            if col == -radius:
                lb_col = -self.delta

            if col == radius - 1:
                ub_col = self.delta

            flows = [
                torch.tensor([lb_col, lb_row]).repeat(
                    self.batch_size, self.height, self.width, 1
                ).float().to(self.device)
            ]

            if row == radius - 1:
                flows.append(
                    torch.tensor([lb_col, ub_row]).repeat(
                        self.batch_size, self.height, self.width, 1
                    ).float().to(self.device)
                )

            if col == radius - 1:
                flows.append(
                    torch.tensor([ub_col, lb_row]).repeat(
                        self.batch_size, self.height, self.width, 1
                    ).float().to(self.device)
                )

            if (row == radius - 1) and (col == radius - 1):
                flows.append(
                    torch.tensor([ub_col, ub_row]).repeat(
                        self.batch_size, self.height, self.width, 1
                    ).float().to(self.device)
                )

            for flow in flows:
                candidate = interpolate(self.images, flow)

                for channel in range(self.channels):
                    self.candidates[channel].append(candidate[:, channel])
                    self.candidate_flows[channel].append(flow)

    def linear_constraints(self):

        if not self.candidates[0]:
            self.box_constraints()

        empty_plane_coord = torch.empty(
            (self.batch_size, self.height, self.width)
        ).float().to(self.device)

        lower_planes = [
            [empty_plane_coord.clone() for _ in range(3)]
            for channel in range(self.channels)
        ]
        upper_planes = [
            [empty_plane_coord.clone() for _ in range(3)]
            for channel in range(self.channels)
        ]

        for channel in range(self.channels):
            candidates = self.candidates[channel]
            candidate_flows = self.candidate_flows[channel]

            for batch in range(self.batch_size):
                for row in range(self.height):
                    for col in range(self.width):
                        lb_a, lb_b, lb_c = self.compute_plane(
                            batch, row, col, candidates, candidate_flows, True
                        )
                        ub_a, ub_b, ub_c = self.compute_plane(
                            batch, row, col, candidates, candidate_flows, False
                        )

                        lower_planes[channel][0][batch, row, col] = lb_a
                        lower_planes[channel][1][batch, row, col] = lb_b
                        lower_planes[channel][2][batch, row, col] = lb_c

                        upper_planes[channel][0][batch, row, col] = ub_a
                        upper_planes[channel][1][batch, row, col] = ub_b
                        upper_planes[channel][2][batch, row, col] = ub_c

        for lower_plane, upper_plane in zip(lower_planes, upper_planes):
            lower_plane[0] -= 1e-5
            upper_plane[0] += 1e-5

        return lower_planes, upper_planes

    def compute_plane(self, batch, row, col, candidates, candidate_flows,
                      is_lower):

        model = Model()

        model.setParam('OutputFlag', False)
        model.setParam('NumericFocus', 2)

        a = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
        b = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
        c = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)

        differences = list()

        for candidate, flow in zip(candidates, candidate_flows):
            (x, y), z = flow[batch, row, col], candidate[batch, row, col]
            x, y, z = x.item(), y.item(), z.item()

            if is_lower:
                model.addConstr(a + b * x + c * y <= z)
                differences.append(z - a + b * x + c * y)

            else:
                model.addConstr(a + b * x + c * y >= z)
                differences.append(a + b * x + c * y - z)

        model.setObjective(quicksum(differences), sense=GRB.MINIMIZE)
        model.optimize()

        if model.status != GRB.OPTIMAL:
            raise ValueError('Gurobi: objective not optimal')

        return a.x, b.x, c.x
