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


from abc import ABC, abstractmethod

import torch


class TNormTransformer(ABC):

    def __init__(self, images, delta):

        assert delta > 0

        self.delta = delta
        self.images = images

        self.batch_size, self.channels, self.height, self.width = images.shape
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.candidates = [list() for channel in range(self.channels)]
        self.candidate_flows = [list() for channel in range(self.channels)]
        self.minimum, self.maximum = self.images.clone(), self.images.clone()
        self.minimum_flows = [
            torch.zeros(
                (self.batch_size, self.height, self.width, 2)
            ).float().to(self.device) for channel in range(self.channels)
        ]
        self.maximum_flows = [
            torch.zeros(
                (self.batch_size, self.height, self.width, 2)
            ).float().to(self.device) for channel in range(self.channels)
        ]

    @abstractmethod
    def add_norm_constraints(self, model, vx, vy):
        pass

    @abstractmethod
    def compute_candidates(self):
        pass

    def box_constraints(self):

        if not self.candidates[0]:
            self.compute_candidates()

        for channel in range(self.channels):
            iterator = zip(
                self.candidates[channel], self.candidate_flows[channel]
            )

            for candidate, flow in iterator:
                self.minimum[:, channel] = torch.min(
                    self.minimum[:, channel], candidate
                )
                self.maximum[:, channel] = torch.max(
                    self.maximum[:, channel], candidate
                )

                self.minimum_flows[channel] = torch.where(
                    (self.minimum[:, channel] == candidate).unsqueeze(-1),
                    flow, self.minimum_flows[channel]
                )
                self.maximum_flows[channel] = torch.where(
                    (self.maximum[:, channel] == candidate).unsqueeze(-1),
                    flow, self.maximum_flows[channel]
                )

        lower_bound = torch.clamp(self.minimum - 1.5e-6, min=0, max=1)
        upper_bound = torch.clamp(self.maximum + 1.5e-6, min=0, max=1)

        return lower_bound, upper_bound

    @abstractmethod
    def linear_constraints(self):
        pass

    @property
    def flow_constraint_pairs(self):

        image_indices = torch.arange(
            self.channels * self.height * self.width
        ).reshape(self.height, self.width, self.channels).to(self.device)

        indices_horizontal = image_indices[:, :-1].flatten()
        indices_vertical = image_indices[:-1].flatten()
        neighbors_horizontal = image_indices[:, 1:].flatten()
        neighbors_vertical = image_indices[1:].flatten()

        indices = torch.cat((indices_horizontal, indices_vertical))
        neighbors = torch.cat((neighbors_horizontal, neighbors_vertical))

        return {'indices': indices, 'neighbors': neighbors}
