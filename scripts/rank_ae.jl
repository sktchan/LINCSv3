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
data = load(data_path)["filtered_data"]

gene_medians = vec(median(data.expr, dims=2)) .+ 1e-10
X = Matrix{Int32}(rank_genes(data.expr, gene_medians))

n_features = size(X, 1) + 2
n_classes = size(X, 1)
n_genes = size(X, 1)
mask_ratio = 0.15f0 
# MASK_ID = (n_classes + 1)
# CLS_ID = n_genes + 2
# CLS_VECTOR = fill(CLS_ID, (1, size(X, 2)))
# X = Int32.(vcat(CLS_VECTOR, X))

X_train, X_test, train_indices, test_indices = split_data(X, 0.2)
# X_train_masked, y_train_masked = mask_input(X_train, mask_ratio, -100, MASK_ID)
# X_test_masked, y_test_masked = mask_input(X_test, mask_ratio, -100, MASK_ID)

MASK_TOKEN_ID = Int32(n_genes + 1)
VOCAB_SIZE = n_genes + 1 

### model ###

### masking

struct Mask
    mask_ratio::Float32
    mask_token_id::Int32
end

function (m::Mask)(x::AbstractMatrix{Int32})
    mask_indices = rand_like(x) .< m.mask_ratio
    X_masked = ifelse.(mask_indices, m.mask_token_id, x)
    mask_labels = ifelse.(mask_indices, x, Int32(0))

    return X_masked, mask_labels
end

### encoder

struct Encoder
    noise::Mask
    embedding::Flux.Embedding
    compress::Flux.Chain
end

function Encoder(
    vocab_size::Int,
    seq_len::Int,
    embed_dim::Int,
    latent_1::Int,
    latent_2::Int,
    latent_3::Int,
    mask_ratio::Float32,
    mask_token_id::Int32
    )

    noise = Mask(mask_ratio, mask_token_id)

    embedding = Flux.Embedding(vocab_size => embed_dim)

    input_to_dense = seq_len * embed_dim

    compress = Flux.Chain(
        Flux.Dense(input_to_dense => latent_1, relu),
        Flux.Dense(latent_1 => latent_2, relu),
        Flux.Dense(latent_2 => latent_3, relu),
        Flux.Dense(latent_3 => embed_dim)
    )

    return Encoder(noise, embedding, compress)
end

Flux.@functor Encoder

function (enc::Encoder)(input::IntMatrix2DType)
    noised_indices, labels = enc.noise(input) 
    embedded = enc.embedding(noised_indices)
    embedded_flat = Flux.flatten(embedded)
    compressed = enc.compress(embedded_flat)
    return compressed, labels
end

### decoder

struct Decoder
    reconstruct::Flux.Chain
    n_genes::Int
end

function Decoder(
    embed_dim::Int,
    n_genes::Int, # This is seq_len
    vocab_size::Int,
    latent_1::Int,
    latent_2::Int,
    latent_3::Int,
    )

    output_dim = n_genes * vocab_size

    reconstruct = Flux.Chain(
        Flux.Dense(embed_dim => latent_3, relu),
        Flux.Dense(latent_3 => latent_2, relu),
        Flux.Dense(latent_2 => latent_1, relu),
        Flux.Dense(latent_1 => output_dim) 
    )

    return Decoder(reconstruct, n_genes)
end

Flux.@functor Decoder

function (dec::Decoder)(input::Float32Matrix2DType)
    return dec.reconstruct(input)
end

### full model

struct Model
    encoder::Encoder
    decoder::Decoder
end

function Model(;
    n_genes::Int,
    embed_dim::Int,
    latent_1::Int,
    latent_2::Int,
    latent_3::Int,
    mask_ratio::Float32,
    mask_token_id::Int32
    )

    vocab_size = n_genes + 1 # genes + mask token

    encoder = Encoder(vocab_size, n_genes, embed_dim, latent_1, latent_2, latent_3, mask_ratio, mask_token_id)
    decoder = Decoder(embed_dim, n_genes, vocab_size, latent_1, latent_2, latent_3)

    return Model(encoder, decoder)
end

Flux.@functor Model

function (model::Model)(input::AbstractMatrix{Int32})
    latent, labels = model.encoder(input)
    logits_flat = model.decoder(latent)
    return logits_flat, labels
end


### training ###

model = Model(
    n_genes=n_genes,
    embed_dim=embed_dim,
    latent_1=latent_1,
    latent_2=latent_2,
    latent_3=latent_3,
    mask_ratio=mask_ratio,
    mask_token_id=MASK_TOKEN_ID
) |> gpu

opt = Flux.setup(Adam(lr), model)

function loss(model::Model, x, mode::String)
    logits_flat, labels = model(x) 
    labels_flat = vec(labels)
    mask_locs = labels_flat .> 0
    
    if count(mask_locs) == 0
        return 0.0f0
    end

    targets = labels_flat[mask_locs]

    vocab_size = size(model.encoder.embedding.weight, 2)
    logits_reshaped = reshape(logits_flat, vocab_size, :)

    logits_masked = logits_reshaped[:, mask_locs]
    targets_oh = Flux.onehotbatch(targets, 1:vocab_size)
    error = Flux.logitcrossentropy(logits_masked, targets_oh)

    if mode == "train"
        return error
    end
    if mode == "test"
        return error, cpu(logits_masked), cpu(targets)
    end
end

train_losses = Float32[]
test_losses = Float32[]
test_rank_errors = Float32[]

all_preds = Int32[] 
all_trues = Int32[] 
all_original_ranks = Int32[]
all_prediction_errors = Int32[]

for epoch in ProgressBar(1:n_epochs)
    train_epoch_losses = Float32[]
    for start_idx in 1:batch_size:size(X_train, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_train, 2))
        x_gpu = gpu(X_train[:, start_idx:end_idx])

        loss_val, grads = Flux.withgradient(model) do m
            loss(m, x_gpu, "train")
        end
        Flux.update!(opt, model, grads[1])
        push!(train_epoch_losses, loss_val)
    end
    push!(train_losses, mean(train_epoch_losses))

    test_epoch_losses = Float32[]
    epoch_rank_errors = Float32[]

    for start_idx in 1:batch_size:size(X_test, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_test, 2))
        x_gpu = gpu(X_test[:, start_idx:end_idx])

        test_loss_val, logits_masked, targets = loss(model, x_gpu, "test")
        
        if isnothing(logits_masked) continue end

        push!(test_epoch_losses, test_loss_val)

        for i in 1:length(targets)
            true_token = targets[i]
            pred_logits = logits_masked[:, i]
            true_logit_val = pred_logits[true_token]
            rank = count(x -> x > true_logit_val, pred_logits) + 1
            rank_error = rank - 1

            push!(epoch_rank_errors, rank_error)

            if epoch == n_epochs
                push!(all_original_ranks, true_token)
                push!(all_prediction_errors, rank_error)
                push!(all_trues, true_token)
                push!(all_preds, argmax(pred_logits))
            end
        end
    end

    push!(test_losses, mean(test_epoch_losses))
    
    if !isempty(epoch_rank_errors)
        push!(test_rank_errors, mean(epoch_rank_errors))
    else
        push!(test_rank_errors, NaN32)
    end
end


### plot/save ###

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", dataset, "rank_ae", timestamp)
mkpath(save_dir)

plot_loss(n_epochs, train_losses, test_losses, save_dir, "logit-ce")
plot_rank_error(n_epochs, test_rank_errors, save_dir)
plot_boxplot(n_classes, all_trues, all_preds, save_dir)
plot_hexbin(all_trues, all_preds, "gene id", save_dir)
plot_prediction_error(all_original_ranks, all_prediction_errors, save_dir)

avg_errors = plot_mean_prediction_error(all_original_ranks, all_prediction_errors, save_dir)
cs, cp = plot_ranked_heatmap(all_trues, all_preds, save_dir, true)


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
                    all_original_ranks, all_prediction_errors, 
                    avg_errors, nothing, nothing, X_test)


log_ae_params(gpu_info, dataset, mask_ratio, batch_size, n_epochs, 
                    embed_dim, hidden_dim, n_heads, n_layers, lr, drop_prob, 
                    additional_notes, run_hours, run_minutes)