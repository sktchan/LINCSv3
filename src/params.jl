# dataset
# data_path = "data/lincs_trt_untrt_data.jld2"
# dataset = "trt"
data_path = "data/lincs_untrt_data.jld2"
dataset = "untrt"

# params
batch_size = 64
n_epochs = 10
embed_dim = 128
drop_prob = 0.05
lr = 0.001
mask_ratio = 0.1

# tf
hidden_dim = 256
n_heads = 2
n_layers = 4

# ae
latent_1 = 734
latent_2 = 490
latent_3 = 246

# latent_1 = 300
# latent_2 = 225
# latent_3 = 150

# notes
gpu_info = "kraken"
additional_notes = "test run for nmacros"

# matrix types
const IntMatrix2DType = Union{Array{Int64}, CuArray{Int32, 2}, CuMatrix{Int64}, DenseCuMatrix{Int64}}
const Float32Matrix2DType = Union{Array{Float32}, CuArray{Float32, 2}}
const Float32Matrix3DType = Union{Array{Float32}, CuArray{Float32, 3}}
