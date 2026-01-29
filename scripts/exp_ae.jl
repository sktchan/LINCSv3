using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using LincsProject, DataFrames, Dates, StatsBase, JLD2, MLUtils
using Flux, Random, ProgressBars, CUDA, Statistics, CairoMakie, LinearAlgebra

include("../src/params.jl")
include("../src/fxns.jl")
include("../src/plot.jl")
include("../src/save.jl")

CUDA.device!(0)

start_time = now()
data = JLD2.load(data_path)["filtered_data"]

X = data.expr

n_genes = size(X, 1)
n_classes = 1 
MASK_VALUE = -1.0f0

X_train, X_test, train_indices, test_indices = split_data(X, 0.2)
X_train_masked, y_train_masked = mask_input(X_train, mask_ratio, NaN32, MASK_VALUE)
X_test_masked, y_test_masked = mask_input(X_test, mask_ratio, NaN32, MASK_VALUE)


### model ###

struct Mask
    mask_ratio::Float64
    mask_value::Float32
end

Flux.@functor Mask ()

### changed to this masking function since masking is done on gpu within the model fxn, not cpu beforehand (as prev in tf file)
function (m::Mask)(x::AbstractMatrix{Float32})
    # random boolean mask for entire matrix for what to mask
        # rand_like: all element of the new array will be set to a random value. 
        # .< mask_ratio: if value is < mask ratio, then it is marked as true
    mask_indices = rand_like(x) .< m.mask_ratio

    # place mask on data
        # if true, places mask_value into the new matrix
        # if false, it copies the original value from x
    X_masked = ifelse.(mask_indices, m.mask_value, x)

    # get masked labels
        # if true, gets original value from x
        # if false, it puts NaN32 to skip over in loss
    mask_labels = ifelse.(mask_indices, x, NaN32)

    return X_masked, mask_labels
end

### encoder

struct Encoder
    noise::Mask
    compress::Flux.Chain
end

function Encoder(
    num_genes::Int,
    embed_dim::Int,
    latent_1::Int,
    latent_2::Int,
    latent_3::Int,
    mask_ratio::Float64,
    mask_value::Float32
    )

    noise = Mask(mask_ratio, mask_value)

    compress = Flux.Chain(
        Flux.Dense(num_genes => latent_1, relu),
        Flux.Dense(latent_1 => latent_2, relu),
        Flux.Dense(latent_2 => latent_3, relu),
        Flux.Dense(latent_3 => embed_dim)
    )

    return Encoder(noise, compress)
end

Flux.@functor Encoder (compress,)

function (enc::Encoder)(input::Float32Matrix2DType)
    noised, labels = enc.noise(input)
    compressed = enc.compress(noised)
    return compressed, labels
end

### decoder

struct Decoder
    reconstruct::Flux.Chain
end

function Decoder(
    embed_dim::Int,
    num_genes::Int,
    latent_1::Int,
    latent_2::Int,
    latent_3::Int,
    )

    reconstruct = Flux.Chain(
        Flux.Dense(embed_dim => latent_3, relu),
        Flux.Dense(latent_3 => latent_2, relu),
        Flux.Dense(latent_2 => latent_1, relu),
        Flux.Dense(latent_1 => num_genes)
    )

    return Decoder(reconstruct)
end

Flux.@functor Decoder

function (dec::Decoder)(input::Float32Matrix2DType)
    return dec.reconstruct(input)
end

### full model

struct Model
    # mlp::Flux.Chain
    encoder::Encoder
    decoder::Decoder
    # mlp_head::Flux.Chain
end

function Model(;
    num_genes::Int,
    embed_dim::Int,
    latent_1::Int,
    latent_2::Int,
    latent_3::Int,
    mask_ratio::Float64,
    mask_value::Float32
    )

    encoder = Encoder(num_genes, embed_dim, latent_1, latent_2, latent_3, mask_ratio, mask_value)

    decoder = Decoder(embed_dim, num_genes, latent_1, latent_2, latent_3)

    return Model(encoder, decoder)
end

Flux.@functor Model

function (model::Model)(input::AbstractMatrix{Float32})
    # embedding = model.mlp(input)
    latent, labels = model.encoder(input)
    recon_embed = model.decoder(latent)
    return recon_embed, labels
end


### training ###

model = Model(
    num_genes=n_genes,
    embed_dim=embed_dim,
    latent_1=latent_1,
    latent_2=latent_2,
    latent_3=latent_3,
    mask_ratio=mask_ratio,
    mask_value=MASK_VALUE
) |> gpu

opt = Flux.setup(Adam(lr), model)

function loss(model::Model, x, mode::String)
    preds, trues = model(x)
    preds_flat = vec(preds)
    trues_flat = vec(trues)

    mask = .!isnan.(trues_flat)
    
    preds_masked = preds_flat[mask]
    trues_masked = trues_flat[mask]
    
    error = Flux.mse(preds_masked, trues_masked)

    if mode == "train"
        return error
    end
    if mode == "test"
        return error, preds_masked, trues_masked, trues
    end
end

train_losses = Float32[]
test_losses = Float32[]
all_preds = Float32[]
all_trues = Float32[]
all_gene_indices = Int[]
all_column_indices = Int[]

for epoch in ProgressBar(1:n_epochs)
    train_epoch_losses = Float32[]
    for start_idx in 1:batch_size:size(X_train, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_train, 2))
        x_gpu = gpu(X_train[:, start_idx:end_idx])

        loss_val, grads = Flux.withgradient(model) do m
            loss(m, x_gpu, "train")
        end
        Flux.update!(opt, model, grads[1])
        loss_val = loss(model, x_gpu, "train")
        push!(train_epoch_losses, loss_val)
    end
    push!(train_losses, mean(train_epoch_losses))

    test_epoch_losses = Float32[]
    for start_idx in 1:batch_size:size(X_test, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_test, 2))
        x_gpu = gpu(X_test[:, start_idx:end_idx])

        test_loss_val, preds_masked, trues_masked, trues_full = loss(model, x_gpu, "test")
        push!(test_epoch_losses, test_loss_val)

        if epoch == n_epochs
            append!(all_preds, cpu(preds_masked))
            append!(all_trues, cpu(trues_masked))

            trues_cpu = cpu(trues_full)
            masked_indices = findall(!isnan, trues_cpu)
            batch_gene_indices = [idx[1] for idx in masked_indices]
            append!(all_gene_indices, batch_gene_indices)

            batch_col_indices = start_idx:end_idx
            pred_col_indices = [batch_col_indices[idx[2]] for idx in masked_indices]
            append!(all_column_indices, pred_col_indices)
        end
    end
    push!(test_losses, mean(test_epoch_losses))
end

correlation = cor(all_trues, all_preds)
absolute_errors = abs.(all_trues .- all_preds)
df_gene_errors = DataFrame(gene_index = all_gene_indices, absolute_error = absolute_errors)


### plot/save ###

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", dataset, "exp_ae", timestamp)
mkpath(save_dir)

plot_loss(n_epochs, train_losses, test_losses, save_dir, "mse")
plot_hexbin(all_trues, all_preds, "expression", save_dir)
plot_prediction_error(all_gene_indices, absolute_errors, save_dir)

avg_errors = plot_mean_prediction_error(all_gene_indices, absolute_errors, save_dir)
plot_sorted_mean_rediction_error(avg_errors, all_gene_indices, absolute_errors, save_dir)

ranked_preds, ranked_trues = convert_exp_to_rank(X_test, all_trues, all_preds)
cs, cp = plot_ranked_heatmap(ranked_trues, ranked_preds, save_dir, false)

log_model(model, save_dir)
# embeddings = get_profile_embeddings(X, model)

# log run info
end_time = now()
run_time = end_time - start_time
total_minutes = div(run_time.value, 60000)
run_hours = div(total_minutes, 60)
run_minutes = rem(total_minutes, 60)

log_info(train_indices, test_indices, nothing, n_epochs, 
                    train_losses, test_losses, all_preds, all_trues, 
                    nothing, nothing, nothing, 
                    X_test_masked, y_test_masked, X_test)


log_ae_params(gpu_info, dataset, mask_ratio, batch_size, n_epochs, 
                    embed_dim, hidden_dim, n_heads, n_layers, lr, drop_prob, 
                    additional_notes, run_hours, run_minutes)




# # here we will test nmacros :DDDDD
# note: i cannot get @nmacros to work after jubox import :(

# using JuBox

# @nscatter begin
#     y1 = train_losses
#     y2 = test_losses
#     x = n_epochs
# end