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

import numpy as np
import torch
from mpmath import polyroots

from spatial.t_inf_norm_transformer import TInfNormTransformer
from spatial.t_norm_transformer import TNormTransformer
from spatial.interpolation import interpolate


class T2NormTransformer(TNormTransformer):

    def add_norm_constraints(self, model, vx, vy):

        model.addConstr(vx + vy <= math.sqrt(2) * self.delta)
        model.addConstr(vx - vy <= math.sqrt(2) * self.delta)
        model.addConstr(-vx + vy <= math.sqrt(2) * self.delta)
        model.addConstr(-vx - vy <= math.sqrt(2) * self.delta)

    def compute_candidates(self):

        delta_sqr = self.delta ** 2
        radius = math.ceil(self.delta)

        for row, col in product(range(-radius, radius), repeat=2):
            lb_row, ub_row = row, row + 1
            lb_col, ub_col = col, col + 1

            interpolation_region = [[lb_col, ub_col], [lb_row, ub_row]]

            distances_row = sorted((abs(lb_row), abs(ub_row)))
            distances_col = sorted((abs(lb_col), abs(ub_col)))

            # no overlap with adversarial region
            if distances_row[0] ** 2 + distances_col[0] ** 2 >= delta_sqr:
                continue

            flows = list()
            flows_by_channel = list()

            # full overlap with interpolation region
            if distances_row[1] ** 2 + distances_col[1] ** 2 <= delta_sqr:
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

                if lb_col ** 2 + lb_row ** 2 <= delta_sqr:
                    flows.append(
                        torch.tensor([lb_col, lb_row]).repeat(
                            self.batch_size, self.height, self.width, 1
                        ).float().to(self.device)
                    )

                if ub_col ** 2 + lb_row ** 2 <= delta_sqr:
                    flows.append(
                        torch.tensor([ub_col, lb_row]).repeat(
                            self.batch_size, self.height, self.width, 1
                        ).float().to(self.device)
                    )

                if lb_col ** 2 + ub_row ** 2 <= delta_sqr:
                    flows.append(
                        torch.tensor([lb_col, ub_row]).repeat(
                            self.batch_size, self.height, self.width, 1
                        ).float().to(self.device)
                    )

                if ub_col ** 2 + ub_row ** 2 <= delta_sqr:
                    flows.append(
                        torch.tensor([ub_col, ub_row]).repeat(
                            self.batch_size, self.height, self.width, 1
                        ).float().to(self.device)
                    )

                box_row = sorted((lb_row, ub_row), key=abs)
                box_col = sorted((lb_col, ub_col), key=abs)

                candidates = list()

                row_sign = -1 if row < 0 else 1
                col_sign = -1 if col < 0 else 1

                if box_col[0] ** 2 <= delta_sqr:
                    candidates.append([
                        box_col[0],
                        row_sign * math.sqrt(delta_sqr - box_col[0] ** 2)
                    ])
                if box_col[1] ** 2 <= delta_sqr:
                    candidates.append([
                        box_col[1],
                        row_sign * math.sqrt(delta_sqr - box_col[1] ** 2)
                    ])
                if box_row[0] ** 2 <= delta_sqr:
                    candidates.append([
                        col_sign * math.sqrt(delta_sqr - box_row[0] ** 2),
                        box_row[0]
                    ])
                if box_row[1] ** 2 <= delta_sqr:
                    candidates.append([
                        col_sign * math.sqrt(delta_sqr - box_row[1] ** 2),
                        box_row[1]
                    ])

                endpoints = [
                    candidate for candidate in candidates if self.in_box(
                        candidate, lb_col, ub_col, lb_row, ub_row
                    )
                ]

                for endpoint in endpoints:
                    flows.append(
                        torch.tensor(endpoint).repeat(
                            self.batch_size, self.height, self.width, 1
                        ).float().to(self.device)
                    )

                flows_by_channel = self.compute_extremum_on_arc(
                    col=col, row=row, endpoints=endpoints,
                    interpolation_region=interpolation_region
                )

            for flow in flows:
                candidate = interpolate(self.images, flow)

                for channel in range(self.channels):
                    self.candidates[channel].append(candidate[:, channel])
                    self.candidate_flows[channel].append(flow)

            for channel, flows in enumerate(flows_by_channel):
                for flow in flows:
                    self.candidates[channel].append(
                        interpolate(self.images, flow)[:, channel]
                    )
                    self.candidate_flows[channel].append(flow)

    def in_box(self, point, lb_x, ub_x, lb_y, ub_y):
        return (lb_x <= point[0] <= ub_x) and (lb_y <= point[1] <= ub_y)

    def compute_extremum_on_arc(self, col, row, endpoints,
                                interpolation_region):

        (lb_col, ub_col), (lb_row, ub_row) = interpolation_region

        alpha = interpolate(
            self.images.double(),
            torch.tensor([lb_col, lb_row]).double().to(self.device)
        )
        beta = interpolate(
            self.images.double(),
            torch.tensor([ub_col, lb_row]).double().to(self.device)
        )
        gamma = interpolate(
            self.images.double(),
            torch.tensor([lb_col, ub_row]).double().to(self.device)
        )
        delta = interpolate(
            self.images.double(),
            torch.tensor([ub_col, ub_row]).double().to(self.device)
        )

        # a = torch.add(
        #     alpha * ub_col * ub_row - beta * lb_col * ub_row,
        #     delta * lb_col * lb_row - gamma * ub_col * lb_row
        # )
        b = (beta - alpha) * ub_row + (gamma - delta) * lb_row
        c = (gamma - alpha) * ub_col + (beta - delta) * lb_col
        d = alpha - beta - gamma + delta

        e = - b / (2 * d)
        f = b * b / (4 * d * d)
        g = c / d
        h = e * e + f

        j = (self.delta ** 2 - h) ** 2 - 4 * f * e * e
        k = - 2 * g * ((self.delta ** 2 - h) + 2 * e * e)
        l = g * g - 4 * ((self.delta ** 2 - h) + e * e)
        m = 4 * g
        n = torch.full_like(m, 4).double().to(self.device)

        flows = [
            [
                torch.zeros(
                    self.batch_size, self.height, self.width, 2
                ).float().to(self.device) for _ in range(16)
            ] for channel in range(self.channels)
        ]

        for batch in range(self.batch_size):
            for channel in range(self.channels):
                for height in range(self.height):
                    for width in range(self.width):

                        b_val = b[batch, channel, height, width].item()
                        c_val = c[batch, channel, height, width].item()
                        d_val = d[batch, channel, height, width].item()

                        if math.isclose(d_val, 0, abs_tol=1e-6):

                            if (c_val == 0) or (b_val == 0):
                                continue

                            denominator = math.sqrt(b_val ** 2 + c_val ** 2)
                            x = b_val * self.delta / denominator
                            y = c_val * self.delta / denominator

                            flows[channel][0][batch, height, width, 0] = x
                            flows[channel][0][batch, height, width, 1] = y

                            flows[channel][1][batch, height, width, 0] = x
                            flows[channel][1][batch, height, width, 1] = -y

                            flows[channel][2][batch, height, width, 0] = -x
                            flows[channel][2][batch, height, width, 1] = y

                            flows[channel][3][batch, height, width, 0] = -x
                            flows[channel][3][batch, height, width, 1] = -y

                            continue

                        coeffs = [
                            n[batch, channel, height, width].item(),
                            m[batch, channel, height, width].item(),
                            l[batch, channel, height, width].item(),
                            k[batch, channel, height, width].item(),
                            j[batch, channel, height, width].item()
                        ]
                        roots = polyroots(coeffs, maxsteps=500, extraprec=100)

                        for idx, root in enumerate(roots):

                            root = complex(root)

                            if not math.isclose(root.imag, 0, abs_tol=1e-7):
                                continue

                            x = float(root.real)

                            if self.delta ** 2 < x ** 2:
                                continue

                            y = math.sqrt(self.delta ** 2 - x ** 2)

                            i = 4 * idx

                            flows[channel][i + 0][batch, height, width, 0] = x
                            flows[channel][i + 0][batch, height, width, 1] = y

                            flows[channel][i + 1][batch, height, width, 0] = x
                            flows[channel][i + 1][batch, height, width, 1] = -y

                            flows[channel][i + 2][batch, height, width, 0] = -x
                            flows[channel][i + 2][batch, height, width, 1] = y

                            flows[channel][i + 3][batch, height, width, 0] = -x
                            flows[channel][i + 3][batch, height, width, 1] = -y

        for channel in range(self.channels):
            for idx in range(16):

                vx = flows[channel][idx][:, :, :, 0]
                vy = flows[channel][idx][:, :, :, 1]

                box_col_constraint = (lb_col <= vx) & (vx <= ub_col)
                box_row_constraint = (lb_row <= vy) & (vy <= ub_row)
                box_constraint = box_col_constraint & box_row_constraint

                flows[channel][idx][:, :, :, 0] = torch.where(
                    box_constraint, vx, torch.zeros_like(vx)
                )
                flows[channel][idx][:, :, :, 1] = torch.where(
                    box_constraint, vy, torch.zeros_like(vy)
                )

        return flows

    def linear_constraints(self):
        return TInfNormTransformer(
            self.images, self.delta
        ).linear_constraints()
