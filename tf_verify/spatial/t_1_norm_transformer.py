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
from spatial.t_norm_transformer import TNormTransformer
from spatial.t_inf_norm_transformer import TInfNormTransformer
from spatial.interpolation import interpolate


class T1NormTransformer(TNormTransformer):

    def add_norm_constraints(self, model, vx, vy):

        model.addConstr(vx + vy <= self.delta)
        model.addConstr(vx - vy <= self.delta)
        model.addConstr(-vx + vy <= self.delta)
        model.addConstr(-vx - vy <= self.delta)

    def compute_candidates(self):

        assert not self.candidates[0]

        radius = math.ceil(self.delta)
        delta_offset = math.modf(self.delta)[0]

        for row, col in product(range(-radius, radius), repeat=2):

            lb_row, ub_row = row, row + 1
            lb_col, ub_col = col, col + 1

            interpolation_region = [[lb_col, ub_col], [lb_row, ub_row]]

            distances_row = sorted((abs(lb_row), abs(ub_row)))
            distances_col = sorted((abs(lb_col), abs(ub_col)))

            # no overlap with adversarial region
            if distances_row[0] + distances_col[0] >= self.delta:
                continue

            # adjust bounds for partial overlap
            if self.delta - (distances_row[0] + distances_col[0]) < 1:
                if row < 0:
                    lb_row = ub_row - delta_offset
                else:
                    ub_row = lb_row + delta_offset

                if col < 0:
                    lb_col = ub_col - delta_offset
                else:
                    ub_col = lb_col + delta_offset

            flows = list()
            flows_by_channel = list()

            # full overlap with interpolation region
            if distances_row[1] + distances_col[1] <= self.delta:
                flows = [
                    torch.tensor([lb_col, lb_row]).repeat(
                        self.batch_size, self.height, self.width, 1
                    ).float().to(self.device),
                    torch.tensor([ub_col, lb_row]).repeat(
                        self.batch_size, self.height, self.width, 1
                    ).float().to(self.device),
                    torch.tensor([lb_col, ub_row]).repeat(
                        self.batch_size, self.height, self.width, 1
                    ).float().to(self.device),
                    torch.tensor([ub_col, ub_row]).repeat(
                        self.batch_size, self.height, self.width, 1
                    ).float().to(self.device)
                ]

            else:
                endpoints_row = [lb_row, ub_row]
                endpoints_col = [lb_col, ub_col]

                # adjust endpoints for overlap with corners
                if 1 < self.delta - (distances_row[0] + distances_col[0]) < 2:
                    if row < 0:
                        endpoints_row[1] -= delta_offset
                    else:
                        endpoints_row[0] += delta_offset

                    if col < 0:
                        endpoints_col[1] -= delta_offset
                    else:
                        endpoints_col[0] += delta_offset

                row_idx = 0 if row < 0 else 1
                col_idx = 0 if col < 0 else 1

                endpoints = [
                    [endpoints_col[col_idx], endpoints_row[1 - row_idx]],
                    [endpoints_col[1 - col_idx], endpoints_row[row_idx]]
                ]

                flows_by_channel = self.compute_extremum_on_line(
                    col=col, row=row, endpoints=endpoints,
                    interpolation_region=interpolation_region
                )

            # partial overlap with interpolation region
            if distances_row[0] + distances_col[1] >= self.delta:
                box_row = sorted((lb_row, ub_row), key=abs)
                box_col = sorted((lb_col, ub_col), key=abs)

                flows.append(
                    torch.tensor([box_col[0], box_row[0]]).repeat(
                        self.batch_size, self.height, self.width, 1
                    ).float().to(self.device)
                )
                flows.append(
                    torch.tensor([box_col[0], box_row[1]]).repeat(
                        self.batch_size, self.height, self.width, 1
                    ).float().to(self.device)
                )
                flows.append(
                    torch.tensor([box_col[1], box_row[0]]).repeat(
                        self.batch_size, self.height, self.width, 1
                    ).float().to(self.device)
                )

            for flow in flows:
                candidate = interpolate(self.images, flow)

                for channel in range(self.channels):
                    self.candidates[channel].append(candidate[:, channel])
                    self.candidate_flows[channel].append(flow)

            for channel, flow in enumerate(flows_by_channel):
                self.candidates[channel].append(
                    interpolate(self.images, flow)[:, channel]
                )
                self.candidate_flows[channel].append(flow)

    def compute_extremum_on_line(self, col, row, endpoints,
                                 interpolation_region):

        box_col, box_row = interpolation_region
        box_col, box_row = sorted(box_col, key=abs), sorted(box_row, key=abs)

        a = interpolate(
            self.images, torch.tensor(
                [box_col[0], box_row[0]], device=self.device
            ).float()
        )
        b = interpolate(
            self.images, torch.tensor(
                [box_col[1], box_row[0]], device=self.device
            ).float()
        )
        c = interpolate(
            self.images, torch.tensor(
                [box_col[0], box_row[1]], device=self.device
            ).float()
        )
        d = interpolate(
            self.images, torch.tensor(
                [box_col[1], box_row[1]], device=self.device
            ).float()
        )

        beta, gamma, delta = b - a, c - a, d + a - b - c

        x1 = abs(endpoints[0][0] - box_col[0])
        x2 = abs(endpoints[1][0] - box_col[0])
        y1 = abs(endpoints[0][1] - box_row[0])
        y2 = abs(endpoints[1][1] - box_row[0])

        slope, offset = (y2 - y1) / (x2 - x1), (x2 * y1 - x1 * y2) / (x2 - x1)

        vx = - (beta + gamma * slope + offset * delta) / (2 * slope * delta)
        vx = torch.where(
            torch.isnan(vx) | (vx == float('inf')) | (vx == float('-inf')),
            torch.zeros_like(vx), vx
        )
        vy = slope * vx + offset

        box_constraint = (0 <= vx) & (vx <= 1) & (0 <= vy) & (vy <= 1)
        vx = torch.where(box_constraint, vx, torch.zeros_like(vx))
        vy = torch.where(box_constraint, vy, torch.zeros_like(vy))

        if col < 0:
            vx *= -1
        if row < 0:
            vy *= -1

        vx += box_col[0]
        vy += box_row[0]

        flows = list()

        for channel in range(self.channels):
            flows.append(torch.stack((vx[:, channel], vy[:, channel]), dim=-1))

        return flows

    def linear_constraints(self):
        return TInfNormTransformer(
            self.images, self.delta
        ).linear_constraints()
