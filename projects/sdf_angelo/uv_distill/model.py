import torch
import torch.nn.functional as torch_F


def build_tangent_frame(normals):
    helper = torch.tensor([0.0, 0.0, 1.0], device=normals.device, dtype=normals.dtype).expand_as(normals)
    alt = torch.tensor([1.0, 0.0, 0.0], device=normals.device, dtype=normals.dtype).expand_as(normals)
    use_alt = normals[:, 2].abs() > 0.99
    helper = torch.where(use_alt[:, None], alt, helper)
    tangent = torch_F.normalize(torch.cross(helper, normals, dim=-1), dim=-1)
    bitangent = torch_F.normalize(torch.cross(normals, tangent, dim=-1), dim=-1)
    return tangent, bitangent


class UVResidualStudent(torch.nn.Module):

    def __init__(self, texture_size, latent_dim=8, latent_scale=4,
                 hidden_dim=64, num_layers=3):
        super().__init__()
        texture_size = int(texture_size)
        latent_scale = max(int(latent_scale), 1)
        self.texture_size = texture_size
        self.latent_dim = int(latent_dim)
        self.latent_scale = latent_scale
        self.latent_res = max(texture_size // latent_scale, 1)
        self.latent = torch.nn.Parameter(
            torch.zeros((1, self.latent_dim, self.latent_res, self.latent_res), dtype=torch.float32)
        )
        input_dim = 3 + 3 + self.latent_dim
        layers = []
        in_dim = input_dim
        num_layers = max(int(num_layers), 1)
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, 3))
        self.mlp = torch.nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.latent, mean=0.0, std=0.01)
        for module in self.mlp:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity="relu")
                torch.nn.init.zeros_(module.bias)
        if isinstance(self.mlp[-1], torch.nn.Linear):
            torch.nn.init.zeros_(self.mlp[-1].weight)
            torch.nn.init.zeros_(self.mlp[-1].bias)

    def sample_latent(self, uv):
        grid = uv * 2.0 - 1.0
        grid = grid.view(1, -1, 1, 2)
        sampled = torch_F.grid_sample(
            self.latent,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled[0, :, :, 0].transpose(0, 1).contiguous()

    def encode_view(self, points, normals, camera_center_norm):
        rays = points - camera_center_norm
        rays = torch_F.normalize(rays, dim=-1)
        tangent, bitangent = build_tangent_frame(normals)
        local_x = (rays * tangent).sum(dim=-1, keepdim=True)
        local_y = (rays * bitangent).sum(dim=-1, keepdim=True)
        local_z = (rays * normals).sum(dim=-1, keepdim=True)
        return torch.cat([local_x, local_y, local_z], dim=-1)

    def forward(self, uv, points, normals, base_rgb, camera_center_norm):
        latent = self.sample_latent(uv)
        view_feat = self.encode_view(points, normals, camera_center_norm)
        x = torch.cat([base_rgb, view_feat, latent], dim=-1)
        delta = torch.tanh(self.mlp(x))
        return delta

    def latent_regularization(self):
        return self.latent.square().mean()

    def latent_tv(self):
        loss = self.latent[:, :, 1:, :] - self.latent[:, :, :-1, :]
        loss = loss.abs().mean()
        loss = loss + (self.latent[:, :, :, 1:] - self.latent[:, :, :, :-1]).abs().mean()
        return loss

