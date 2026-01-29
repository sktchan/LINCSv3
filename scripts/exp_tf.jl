using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using LincsProject, DataFrames, Dates, StatsBase, JLD2
using Flux, Random, ProgressBars, CUDA, Statistics, CairoMakie, LinearAlgebra

include("../src/params.jl")
include("../src/fxns.jl")
include("../src/plot.jl")
include("../src/save.jl")

CUDA.device!(0)

start_time = now()
data = load(data_path)["filtered_data"]

X = data.expr

n_genes = size(X, 1)
n_classes = 1 
MASK_VALUE = -1.0f0

X_train, X_test, train_indices, test_indices = split_data(X, 0.2)
X_train_masked, y_train_masked = mask_input(X_train, mask_ratio, NaN32, MASK_VALUE)
X_test_masked, y_test_masked = mask_input(X_test, mask_ratio, NaN32, MASK_VALUE)


### model ###

### positional encoder

struct PosEnc
    pe_matrix::Float32Matrix2DType
end

#!# uses n_genes as max_len directly
function PosEnc(embed_dim::Int, max_len::Int) # max_len is number of genes
    pe_matrix = Matrix{Float32}(undef, embed_dim, max_len)
    for pos in 1:max_len, i in 1:embed_dim
        angle = pos / (10000^(2*(div(i-1,2))/embed_dim))
        if mod(i, 2) == 1
            pe_matrix[i,pos] = sin(angle) # odd indices
        else
            pe_matrix[i,pos] = cos(angle) # even indices
        end
    end
    return PosEnc(cu(pe_matrix))
end

Flux.@functor PosEnc

function (pe::PosEnc)(input::Float32Matrix3DType)
    seq_len = size(input,2)
    return input .+ pe.pe_matrix[:,1:seq_len] # adds positional encoding to input embeddings
end

### building transformer section

struct Transf
    mha::Flux.MultiHeadAttention
    att_dropout::Flux.Dropout
    att_norm::Flux.LayerNorm # this is the normalization aspect
    mlp::Flux.Chain
    mlp_norm::Flux.LayerNorm
end

function Transf(
    embed_dim::Int, 
    hidden_dim::Int; 
    n_heads::Int, 
    dropout_prob::Float64
    )

    mha = Flux.MultiHeadAttention((embed_dim, embed_dim, embed_dim) => (embed_dim, embed_dim) => embed_dim, 
                                    nheads=n_heads, 
                                    dropout_prob=dropout_prob
                                    )

    att_dropout = Flux.Dropout(dropout_prob)
    
    att_norm = Flux.LayerNorm(embed_dim)
    
    mlp = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(dropout_prob),
        Flux.Dense(hidden_dim => embed_dim),
        Flux.Dropout(dropout_prob)
        )
    mlp_norm = Flux.LayerNorm(embed_dim)

    return Transf(mha, att_dropout, att_norm, mlp, mlp_norm)
end

Flux.@functor Transf

function (tf::Transf)(input::Float32Matrix3DType) # input shape: embed_dim × seq_len × batch_size
    normed = tf.att_norm(input)
    atted = tf.mha(normed, normed, normed)[1] # outputs a tuple (a, b)
    att_dropped = tf.att_dropout(atted)
    residualed = input + att_dropped
    res_normed = tf.mlp_norm(residualed)

    embed_dim, seq_len, batch_size = size(res_normed)
    reshaped = reshape(res_normed, embed_dim, seq_len * batch_size) # dense layers expect 2D inputs
    mlp_out = tf.mlp(reshaped)
    mlp_out_reshaped = reshape(mlp_out, embed_dim, seq_len, batch_size)
    
    tf_output = residualed + mlp_out_reshaped
    return tf_output
end

#!# full model for raw value regression

struct Model
    projection::Flux.Dense #!# replace embedding w/ dense layer for cont's input
    pos_encoder::PosEnc
    pos_dropout::Flux.Dropout
    transformer::Flux.Chain
    classifier::Flux.Chain
end

function Model(;
    seq_len::Int, #!# changed from input_size
    embed_dim::Int,
    n_layers::Int,
    n_classes::Int, #!# 1 for regression
    n_heads::Int,
    hidden_dim::Int,
    dropout_prob::Float64
    )

    #!# project the single raw expression value to the embedding dimension
    projection = Flux.Dense(1 => embed_dim)

    pos_encoder = PosEnc(embed_dim, seq_len)

    pos_dropout = Flux.Dropout(dropout_prob)

    transformer = Flux.Chain(
        [Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...
        )

    #!# classifier preds a singular cont's val
    classifier = Flux.Chain(
        Flux.Dense(embed_dim => embed_dim, gelu),
        Flux.LayerNorm(embed_dim),
        Flux.Dense(embed_dim => 1, softplus) #!# 1 value returned
        )

    return Model(projection, pos_encoder, pos_dropout, transformer, classifier)
end

Flux.@functor Model

function (model::Model)(input::Float32Matrix2DType)
    seq_len, batch_size = size(input)

    #!# reshape for dense projection: (seq_len, batch_size) -> (1, seq_len * batch_size)
    input_reshaped = reshape(input, 1, :)
    #!# output is (embed_dim, seq_len * batch_size) -> (embed_dim, seq_len, batch_size)
    embedded = reshape(model.projection(input_reshaped), :, seq_len, batch_size)
    
    encoded = model.pos_encoder(embedded)
    encoded_dropped = model.pos_dropout(encoded)
    transformed = model.transformer(encoded_dropped)
    
    regression_output = model.classifier(transformed)
    return regression_output
end


### training ###

model = Model(
    seq_len=n_genes,
    embed_dim=embed_dim,
    n_layers=n_layers,
    n_classes=n_classes, # n_classes is 1
    n_heads=n_heads,
    hidden_dim=hidden_dim,
    dropout_prob=drop_prob
) |> gpu

opt = Flux.setup(Adam(lr), model)

#!# loss is now mse for regression on masked values

function loss(model::Model, x, y, mode::String)
    preds = model(x)  # (1, seq_len, batch_size)
    preds_flat = vec(preds)
    y_flat = vec(y)

    mask = .!isnan.(y_flat)

    if sum(mask) == 0
        return 0.0f0
    end
    
    preds_masked = preds_flat[mask]
    y_masked = y_flat[mask]
    
    regression_loss = Flux.mse(preds_masked, y_masked)

    if mode == "train"
        return regression_loss
    end
    if mode == "test"
        return regression_loss, preds_masked, y_masked
    end
end

train_losses = Float32[]
test_losses = Float32[]

# Profile.Allocs.@profile sample_rate=1 begin
for epoch in ProgressBar(1:n_epochs)

    epoch_losses = Float32[]

    # # dynamic masking here (optional, kept as is)
    # X_train_masked = copy(X_train)
    # y_train_masked = mask_input_dyn(X_train_masked)

    for start_idx in 1:batch_size:size(X_train_masked, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_train_masked, 2))
        x_gpu = gpu(X_train_masked[:, start_idx:end_idx])
        y_gpu = gpu(y_train_masked[:, start_idx:end_idx])
        
        loss_val, grads = Flux.withgradient(model) do m
            loss(m, x_gpu, y_gpu, "train")
        end
        Flux.update!(opt, model, grads[1])
        loss_val = loss(model, x_gpu, y_gpu, "train")
        push!(epoch_losses, loss_val)
    end

    push!(train_losses, mean(epoch_losses))

    test_epoch_losses = Float32[]
    
    for start_idx in 1:batch_size:size(X_test_masked, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_test_masked, 2))
        x_gpu = gpu(X_test_masked[:, start_idx:end_idx])
        y_gpu = gpu(y_test_masked[:, start_idx:end_idx])

        test_loss_val, _, _ = loss(model, x_gpu, y_gpu, "test")
        push!(test_epoch_losses, test_loss_val)

    end

    push!(test_losses, mean(test_epoch_losses))
end

all_preds = Float32[]
all_trues = Float32[]
all_gene_indices = Int[]
all_column_indices = Int[]

for start_idx in 1:batch_size:size(X_test_masked, 2)
    end_idx = min(start_idx + batch_size - 1, size(X_test_masked, 2))
    x_gpu = gpu(X_test_masked[:, start_idx:end_idx])
    y_gpu = gpu(y_test_masked[:, start_idx:end_idx])
    _, preds_masked, y_masked = loss(model, x_gpu, y_gpu, "test")

    # for calculating predicted error per gene
    y_cpu = cpu(y_gpu)
    masked_indices = findall(!isnan, y_cpu)
    batch_gene_indices = [idx[1] for idx in masked_indices]
    append!(all_gene_indices, batch_gene_indices)

    append!(all_preds, cpu(preds_masked))
    append!(all_trues, cpu(y_masked))

    batch_col_indices = start_idx:end_idx
    pred_col_indices = [batch_col_indices[idx[2]] for idx in masked_indices]
    append!(all_column_indices, pred_col_indices)
end

correlation = cor(all_trues, all_preds)
absolute_errors = abs.(all_trues .- all_preds)
df_gene_errors = DataFrame(gene_index = all_gene_indices, absolute_error = absolute_errors)



### plot/save ###

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", dataset, "exp_tf", timestamp)
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


log_tf_params(gpu_info, dataset, mask_ratio, batch_size, n_epochs, 
                    embed_dim, hidden_dim, n_heads, n_layers, lr, drop_prob, 
                    additional_notes, run_hours, run_minutes)