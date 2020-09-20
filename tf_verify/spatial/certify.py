        # TODO
        config.sparse_n = krelu
        config.refine_neurons = krelu > 0
        config.use_milp = use_milp
        config.timeout_milp = milp_timeout

    def normalize_plane(self, plane, channel, is_constant):
        plane_ = plane.clone()

        if is_constant:
            plane_ -= self.mean[channel]

        plane_ /= self.std[channel]

        return plane_

