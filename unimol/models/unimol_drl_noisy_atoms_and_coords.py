# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from typing import Dict, Any, List


logger = logging.getLogger(__name__)

class Diffusion:
    def __init__(self, args, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def get_noisy_data(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

@register_model("unimol_drl_noisy_atoms_and_coords")
class UniMolModelDRLNoisyAtomsAndCoords(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--diffusion-loss",
            type=float,
            metavar="D",
            help="diffusion loss ratio",
        )
        parser.add_argument(
            "--x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )

        parser.add_argument(
            "--max_noise_scale",
            type=float,
            default=3.
        )

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        print('[***********] NOISE SCALE:', args.max_noise_scale)

        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )

        ########################## Diffusion Part
        self.molecule_pooling = nn.MultiheadAttention(embed_dim=args.encoder_embed_dim, num_heads=8, batch_first = True) 
        self.gen_query = nn.Parameter(torch.randn(1, 1, args.encoder_embed_dim))

        self.diffusion = Diffusion(args)
        self.diffusion_encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )

        # # self.diffusion_coord_head = self.dist_head = NonLinearHead(
        # #         args.encoder_embed_dim, 3, args.activation_fn
        # #     )

        self.diffusion_embed = NonLinearHead(
            args.encoder_embed_dim*2 + 1, args.encoder_embed_dim, args.activation_fn
        )
        self.diffusion_coord_head = self.dist_head = NonLinearHead(
                args.encoder_attention_heads, 3, args.activation_fn
            )
        self.diffusion_atom_head = NonLinearHead(
                args.encoder_embed_dim, args.encoder_embed_dim, args.activation_fn
            )
        self.diffusion_dist_head = NonLinearHead(
                args.encoder_attention_heads, 1, args.activation_fn
            )
        ###########################

        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                args.encoder_attention_heads, 1, args.activation_fn
            )
        if args.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                args.encoder_attention_heads, args.activation_fn
            )

        
        self.classification_heads = nn.ModuleDict()
        self.apply(init_bert_params)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):

        if classification_head_name is not None:
            features_only = True

        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)
        gt_atoms = x.clone().detach()

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        # # ##########
        # # sigma = torch.rand(x.shape[0], 1, 1).cuda() * 3. # B 1 1 from 0 to 10
        # # noise_coord = torch.randn_like(src_coord) * sigma
        # # coord_t = src_coord + noise_coord
        # # coord_t = coord_t.detach()

        # # batch_size, seq_len, _ = coord_t.shape
        # # distance_t = torch.cdist(coord_t.view(batch_size, seq_len, 3), coord_t.view(batch_size, seq_len, 3), p=2)

        # # distance_t = distance_t.half().detach()
        # # ##########

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)

        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        encoder_distance = None
        encoder_coord = None
        logits = None

        if not features_only:
            if self.args.masked_token_loss > 0:
                logits = self.lm_head(encoder_rep, encoder_masked_tokens)
            if self.args.masked_coord_loss > 0:
                coords_emb = src_coord
                if padding_mask is not None:
                    atom_num = torch.sum(1 - padding_mask.type_as(x), dim=1).view(
                        -1, 1, 1, 1
                    )  # consider BOS and EOS as part of the object
                else:
                    atom_num = src_coord.shape[1]
                delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
                coord_update = delta_pos / atom_num * attn_probs
                # Mask padding
                pair_coords_mask = (1 - padding_mask.float()).unsqueeze(-1) * (1 - padding_mask.float()).unsqueeze(1)
                coord_update = coord_update * pair_coords_mask.unsqueeze(-1)
                #
                coord_update = torch.sum(coord_update, dim=2)
                encoder_coord = coords_emb + coord_update
            if self.args.masked_dist_loss > 0:
                encoder_distance = self.dist_head(encoder_pair_rep)


        ########## Diffusion Part     
        # # predicted_coord_noise = self.diffusion_coord_head(encoder_rep)
        ########## 

        padding_mask = src_tokens.eq(self.padding_idx)
        pair_coords_mask = (1 - padding_mask.float()).unsqueeze(-1) * (1 - padding_mask.float()).unsqueeze(1)

        cared_mask = 1 - padding_mask.float()
        cared_pair_mask = 1 - pair_coords_mask

        sigma = torch.rand(x.shape[0], 1, 1).half().cuda() * self.args.max_noise_scale # B 1 1 from 0 to 3
        noise_coord = torch.randn_like(src_coord) * sigma * cared_mask.unsqueeze(-1)
        coord_t = src_coord.clone().detach() + noise_coord
        coord_t = coord_t.half().detach()

        noise_atoms = torch.randn_like(x) * sigma * cared_mask.unsqueeze(-1)
        x_t = x.clone().detach() + noise_atoms
        x_t = x_t.half().detach()


        batch_size, seq_len, _ = coord_t.shape
        distance_t = torch.cdist(coord_t.view(batch_size, seq_len, 3).float(), coord_t.view(batch_size, seq_len, 3).float(), p=2) * cared_pair_mask
        distance_t = distance_t.half().detach()


        noise_distances = (src_distance - distance_t).clone().detach()      


        diff_edge_type = torch.zeros_like(src_edge_type)
        noisy_graph_attn_bias = get_dist_features(distance_t, diff_edge_type)

        input_embedding = torch.cat([x_t, sigma.repeat(1, x.shape[1], 1)], dim=-1)
        # conditioning on pristine molecule rep
        generative_rep, _ = self.molecule_pooling(self.gen_query.repeat(encoder_rep.shape[0], 1, 1), encoder_rep, encoder_rep) # encoder rep will be the representation used for predictions
        input_embedding = torch.cat([input_embedding, generative_rep.repeat(1, x.shape[1], 1)], dim=-1)
        # transform input embedding
        input_embedding = self.diffusion_embed(input_embedding)
        
        (
            predicted_x_noise,
            diff_distance_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.diffusion_encoder(input_embedding, padding_mask=None, attn_mask=noisy_graph_attn_bias)

        predicted_distance_noise = self.diffusion_dist_head(diff_distance_rep).squeeze(-1)
        predicted_coord_noise = self.diffusion_coord_head(delta_encoder_pair_rep.sum(dim=2))
        predicted_atom_noise = self.diffusion_atom_head(predicted_x_noise)

        #########################

        if classification_head_name is not None:
            logits = self.classification_heads[classification_head_name](encoder_rep)
        if self.args.mode == 'infer':
            return encoder_rep, encoder_pair_rep
        else:
            if self.args.masked_token_loss <= 0:
                if logits is None:
                    logits = encoder_rep ##################
                # # logits = encoder_rep
                encoder_distance = None
                encoder_coord = None

            return (
                logits,
                encoder_distance,
                encoder_coord,
                x_norm,
                delta_encoder_pair_rep_norm,
                {'coord_noise': noise_coord, 'coord_pred': predicted_coord_noise,
                'dist_noise': noise_distances, 'dist_pred': predicted_distance_noise,
                'atom_noise': noise_atoms, 'atom_pred': predicted_atom_noise}
            )         
        
            # # return (
            # #     logits,
            # #     encoder_distance,
            # #     encoder_coord,
            # #     x_norm,
            # #     delta_encoder_pair_rep_norm,
            # #     {'x_noise': noise_x, 'x_pred': predicted_x_noise, 'dist_noise': noise_distance, 'dist_pred': predicted_distance_noise}
            # # )         

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )

        if name in ['dipole_ranking', 'homo_ranking', 'lumo_ranking', 'multi-ranking']:
            self.classification_heads[name] = RankingHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )
        elif name in ['multi-ranking-regression']:
            self.classification_heads[name] = RankingRegressionHead(
                input_dim=self.args.encoder_embed_dim,
                inner_dim=inner_dim or self.args.encoder_embed_dim,
                num_classes=num_classes,
                activation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
            )
        
        elif name in ['rank_split_0', 'rank_split_1', 'rank_split_2']:
            self.classification_heads['rank-regressor'] = RankingRegressionHeadFused(
                input_dim=self.args.encoder_embed_dim,
                inner_dim=inner_dim or self.args.encoder_embed_dim,
                num_classes=num_classes,
                activation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
            )
        else:
            self.classification_heads[name] = ClassificationHead(
                input_dim=self.args.encoder_embed_dim,
                inner_dim=inner_dim or self.args.encoder_embed_dim,
                num_classes=num_classes,
                activation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
            )


    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class RankingHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim*2, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features1, features2, **kwargs):
        x1 = features1[:, 0, :]
        x2 = features2[:, 0, :]

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        concat_x = torch.cat([x1, x2], dim=-1)

        x = self.dense(concat_x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RankingRegressionHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim*2, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

        self.dense_reg = nn.Linear(input_dim, inner_dim)
        self.out_proj_reg = nn.Linear(inner_dim, 3)

    def forward(self, features1, features2, **kwargs):
        x1_ori = features1[:, 0, :]
        x2_ori = features2[:, 0, :]

        x1 = self.dropout(x1_ori)
        x2 = self.dropout(x2_ori)

        concat_x = torch.cat([x1, x2], dim=-1)

        x = self.dense(concat_x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        x1_reg = self.out_proj_reg(self.activation_fn(self.dense_reg(x1_ori)))
        x2_reg = self.out_proj_reg(self.activation_fn(self.dense_reg(x2_ori)))
        return {'cls_logits': x, 'reg1': x1_reg, 'reg2': x2_reg}
    
class RankingRegressionHeadFused(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim*2, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

        self.dense_reg = nn.Linear(input_dim, inner_dim)
        self.out_proj_reg = nn.Linear(inner_dim, 1)

    def forward(self, features1, features2, **kwargs):
        if features2 is None:
            x_reg = self.out_proj_reg(self.activation_fn(self.dense_reg(features1[:, 0, :])))
            return x_reg
        else:
            x1_ori = features1[:, 0, :]
            x2_ori = features2[:, 0, :]

            x1 = self.dropout(x1_ori)
            x2 = self.dropout(x2_ori)

            concat_x = torch.cat([x1, x2], dim=-1)

            x = self.dense(concat_x)
            x = self.activation_fn(x)
            x = self.dropout(x)
            x = self.out_proj(x)

            return x

class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


@register_model_architecture("unimol_drl_noisy_atoms_and_coords", "unimol_drl_noisy_atoms_and_coords")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    args.diffusion_loss = getattr(args, "diffusion_loss", -1.0)
    args.max_noise_scale = getattr(args, "max_noise_scale", 3.0)


@register_model_architecture("unimol_drl_noisy_atoms_and_coords", "unimol_drl_noisy_atoms_and_coords_base")
def unimol_base_architecture(args):
    base_architecture(args)
