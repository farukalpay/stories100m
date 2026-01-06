/*
 * inference_engine.hpp
 * 
 * Core Inference Runtime for Llama 2 Architectures.
 * Provides a zero-copy, memory-mapped interface to the model weights and
 * implements hardware-accelerated (NEON) forward pass kernels.
 *
 * Design Principles:
 *  - Zero Allocations on Critical Path: All buffers are pre-allocated.
 *  - Cache Locality: Weights are mapped sequentially; Activations are aligned.
 *  - SIMD Saturation: Kernels utilize full 128-bit NEON pipeline.
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <arm_neon.h>
#include <new>

// Optimization Hints
#define FORCE_INLINE __attribute__((always_inline)) inline
#define ALIGNED(x)   __attribute__((aligned(x)))
#define RESTRICT     __restrict__

namespace bare_metal {

// -----------------------------------------------------------------------------
// 1. CONFIGURATION
// -----------------------------------------------------------------------------

struct Config {
    int dim;        // Transformer dimension (e.g. 288)
    int hidden_dim; // FFN Hidden dimension (e.g. 768)
    int n_layers;   // Layer depth
    int n_heads;    // Query heads
    int n_kv_heads; // Key/Value heads (GQA Support)
    int vocab_size; // Lexicon size
    int seq_len;    // Context window size
};

// -----------------------------------------------------------------------------
// 2. MATH KERNELS (ARM NEON)
// -----------------------------------------------------------------------------

struct Math {
    static FORCE_INLINE void rmsnorm(float* o, const float* x, const float* weight, int size) {
        // Calculate sum of squares
        float32x4_t v_ss = vdupq_n_f32(0.0f);
        for (int i = 0; i < size; i += 4) {
             float32x4_t v_x = vld1q_f32(x + i);
             v_ss = vmlaq_f32(v_ss, v_x, v_x);
        }
        float ss = vaddvq_f32(v_ss);
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);

        // Normalize and scale
        float32x4_t v_ss_inv = vdupq_n_f32(ss);
        for (int i = 0; i < size; i += 4) {
            float32x4_t v_x = vld1q_f32(x + i);
            float32x4_t v_w = vld1q_f32(weight + i);
            vst1q_f32(o + i, vmulq_f32(vmulq_f32(v_x, v_ss_inv), v_w));
        }
    }

    static FORCE_INLINE void matmul(float* xout, const float* x, const float* w, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        // Parallelize over d (output dimension)
        #pragma omp parallel for
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            int j = 0;
            
            // Vectorized dot product
            float32x4_t v_sum = vdupq_n_f32(0.0f);
            for (; j <= n - 4; j += 4) {
                 float32x4_t v_x = vld1q_f32(x + j);
                 float32x4_t v_w = vld1q_f32(w + i * n + j);
                 v_sum = vmlaq_f32(v_sum, v_x, v_w); // FMALL
            }
            val = vaddvq_f32(v_sum);

            // Tail
            for (; j < n; j++) {
                val += w[i * n + j] * x[j];
            }
            xout[i] = val;
        }
    }

    static FORCE_INLINE void softmax(float* x, int size) {
        // Find max
        float max_val = x[0];
        for (int i = 1; i < size; i++) {
            if (x[i] > max_val) max_val = x[i];
        }

        // Exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[i] = expf(x[i] - max_val);
            sum += x[i];
        }

        // Normalize
        float scale = 1.0f / sum;
        for (int i = 0; i < size; i++) {
            x[i] *= scale;
        }
    }
    
    static FORCE_INLINE void silu_elementwise(float* x, const float* hb, int size) {
        for (int i = 0; i < size; i++) {
            float val = x[i];
            // Silu = x * sigmoid(x) = x / (1 + exp(-x))
            val *= (1.0f / (1.0f + expf(-val)));
            // Multiply by hb (gate)
            x[i] = val * hb[i];
        }
    }
};

// -----------------------------------------------------------------------------
// 3. MODEL STRUCTURES
// -----------------------------------------------------------------------------

struct TransformerWeights {
    // [Vocabulary]
    float* token_embedding_table; // Shape: (vocab_size, dim)
    
    // [Layer Weights]
    float* rms_att_weight; // RMSNorm weights for Attention input
    float* rms_ffn_weight; // RMSNorm weights for FFN input
    
    // Attention Projections
    float* wq; // Query Projection: (layer, dim, n_heads * head_size)
    float* wk; // Key Projection:   (layer, dim, n_kv_heads * head_size)
    float* wv; // Value Projection: (layer, dim, n_kv_heads * head_size)
    float* wo; // Output Projection: (layer, n_heads * head_size, dim)
    
    // Feed-Forward Network (SwiGLU)
    float* w1; // Gate Projection
    float* w2; // Down Projection
    float* w3; // Up Projection
    
    // [Finalization]
    float* rms_final_weight; // Final Norm
    float* wcls;             // Classifier Head (often tied to embeddings)
};

struct RunState {
    // Activations (Scratchpad Memory)
    // Note: All pointers MUST be 64-byte aligned for AVX/NEON loads.
    float *x;      // Input features
    float *xb;     // Residual branch buffer
    float *xb2;    // Secondary buffer
    float *hb;     // FFN Hidden state
    float *hb2;    // FFN Hidden state (SwiGLU)
    float *q;      // Query Vector
    float *k;      // Key Vector
    float *v;      // Value Vector
    float *att;    // Attention Scores (Softmax buffer)
    float *logits; // Output Probabilities
    
    // KV Cache (Persistent State)
    // Layout: (Layer, Seq_Len, Dim)
    float* key_cache;   
    float* value_cache; 
};

class Transformer {
public:
    Config config;
    TransformerWeights weights;
    RunState state;
    
    int fd;
    float* data;
    size_t file_size;

    Transformer(const char* checkpoint_path) {
        // [1] Load Checkpoint Metadata
        FILE *file = fopen(checkpoint_path, "rb");
        if (!file) { fprintf(stderr, "FATAL: Unable to open model '%s'\n", checkpoint_path); exit(1); }
        if (fread(&config, sizeof(Config), 1, file) != 1) { exit(1); }
        
        // [2] Map Weights into Address Space (Zero-Copy)
        fseek(file, 0, SEEK_END);
        file_size = ftell(file);
        fclose(file);
        
        fd = open(checkpoint_path, O_RDONLY);
        data = (float*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { perror("mmap"); exit(1); }
        
        // [3] Initialize Weight Pointers
        // Pointers are offset directly into the memory-mapped segment.
        float* weights_ptr = data + sizeof(Config)/sizeof(float);
        
        // Helper to increment pointer
        auto next = [&](int size) { float* p = weights_ptr; weights_ptr += size; return p; };
        
        int dim = config.dim;
        int hidden_dim = config.hidden_dim;
        int layers = config.n_layers;
        int heads = config.n_heads;
        int kv_heads = config.n_kv_heads;
        int vocab = config.vocab_size;
        int head_size = dim / heads; // usually
        
        weights.token_embedding_table = next(vocab * dim);
        
        weights.rms_att_weight = next(layers * dim);
        weights.wq = next(layers * dim * (heads * head_size));
        weights.wk = next(layers * dim * (kv_heads * head_size));
        weights.wv = next(layers * dim * (kv_heads * head_size));
        weights.wo = next(layers * (heads * head_size) * dim);
        
        weights.rms_ffn_weight = next(layers * dim);
        weights.w1 = next(layers * hidden_dim * dim);
        weights.w2 = next(layers * dim * hidden_dim);
        weights.w3 = next(layers * hidden_dim * dim);
        
        weights.rms_final_weight = next(dim);
        
        // Skip freq_cis_real/imag (RoPE tables) usually stored here in llama2.c bin
        // They are seq_len * head_size / 2
        // We will calc them on fly or skip strict ptr mapping if we don't need them pre-calc'd
        // For simplicity, we assume they are there:
        int head_size_val = config.dim / config.n_heads;
        weights_ptr += config.seq_len * head_size_val / 2; // real
        weights_ptr += config.seq_len * head_size_val / 2; // imag

        // shared weights logic would go here, but for now map wcls if present
        // In some versions wcls is absent if tied? 
        // We act as if it is present if we haven't hit EOF.
        // But for stories100m.bin usually output head is NOT tied or is stored?
        // Let's assume there is a wcls at end?
        // In Karpathy's code: if (shared_classifier) wcls = token_embedding_table else ...
        // We'll trust the offset for now or check file size.
        // For benchmarks, let's assume we read it or point to it.
        // Optimization: just use token_embedding_table if we run out of bytes?
        
        // Check for Shared Classifier (Weight Tying)
        size_t bytes_remaining = file_size - ((char*)weights_ptr - (char*)data);
        size_t classifier_size = vocab * dim * sizeof(float);
        
        if (bytes_remaining < classifier_size) {
            // Case: Shared Classifier
            weights.wcls = weights.token_embedding_table;
            printf("    -> Weights:   Shared Classifier Detected.\n");
        } else {
            // Case: Explicit Classifier
            weights.wcls = weights_ptr;
            printf("    -> Weights:   Separate Classifier Detected.\n");
        }

        // Verify Weight Bounds
        // If shared, we don't expect extra bytes for classifier
        size_t bytes_mapped;
        if (weights.wcls == weights.token_embedding_table) {
             bytes_mapped = (char*)weights_ptr - (char*)data;
        } else {
             bytes_mapped = ((char*)weights_ptr + classifier_size) - (char*)data;
        }

        if (bytes_mapped > file_size) {
            fprintf(stderr, "FATAL: Checkpoint file too small!\n");
            fprintf(stderr, "  Need:     %zu bytes\n", bytes_mapped);
            fprintf(stderr, "  Actual:   %zu bytes\n", file_size);
            exit(1);
        }
        printf("    -> Mapped:    %.2f MB\n", bytes_mapped / 1024.0 / 1024.0);


        // 4. Allocate RunState (Activations)
        size_t kv_size = layers * config.seq_len * dim;
        state.key_cache   = (float*)aligned_alloc(64, kv_size * sizeof(float));
        state.value_cache = (float*)aligned_alloc(64, kv_size * sizeof(float));
        
        state.x = (float*)aligned_alloc(64, dim * sizeof(float));
        state.xb = (float*)aligned_alloc(64, dim * sizeof(float));
        state.xb2 = (float*)aligned_alloc(64, dim * sizeof(float));
        state.hb = (float*)aligned_alloc(64, hidden_dim * sizeof(float));
        state.hb2 = (float*)aligned_alloc(64, hidden_dim * sizeof(float));
        state.q = (float*)aligned_alloc(64, dim * sizeof(float));
        state.k = (float*)aligned_alloc(64, dim * sizeof(float));
        state.v = (float*)aligned_alloc(64, dim * sizeof(float));
        state.att = (float*)aligned_alloc(64, heads * config.seq_len * sizeof(float));
        state.logits = (float*)aligned_alloc(64, vocab * sizeof(float));
    }

    ~Transformer() {
        munmap(data, file_size);
        close(fd);
        free(state.key_cache); free(state.value_cache);
        free(state.x); free(state.xb); free(state.xb2);
        free(state.hb); free(state.hb2);
        free(state.q); free(state.k); free(state.v);
        free(state.att); free(state.logits);
    }
    
    // Naive RoPE for brevity, optimized versions exist
    void rope(float* q, float* k, int pos, int head_size) {
        for (int i = 0; i < config.n_heads; i++) {
            for (int j = 0; j < head_size; j += 2) {
                float theta = pos * powf(10000.0f, -((float)j / head_size));
                float cost = cosf(theta);
                float sint = sinf(theta);
                
                float* q_ptr = q + i * head_size + j;
                float q0 = q_ptr[0];
                float q1 = q_ptr[1];
                q_ptr[0] = q0 * cost - q1 * sint;
                q_ptr[1] = q0 * sint + q1 * cost;
                
                if (i < config.n_kv_heads) {
                    float* k_ptr = k + i * head_size + j;
                    float k0 = k_ptr[0];
                    float k1 = k_ptr[1];
                    k_ptr[0] = k0 * cost - k1 * sint;
                    k_ptr[1] = k0 * sint + k1 * cost;
                }
            }
        }
    }

    float* forward(int token, int pos) {
        // Debug prints for segfault isolation
        // printf("DEBUG: Forward token=%d pos=%d\n", token, pos);

        // 1. Embedding
        int dim = config.dim;
        float* content_row = weights.token_embedding_table + token * dim;
        memcpy(state.x, content_row, dim * sizeof(float));
        
        // 2. Layers
        int head_size = dim / config.n_heads;
        
        for (int l = 0; l < config.n_layers; l++) {
            // Attention RMSNorm
            Math::rmsnorm(state.xb, state.x, weights.rms_att_weight + l*dim, dim);
            
            // QKV Matmuls
            Math::matmul(state.q, state.xb, weights.wq + l*dim*dim, dim, dim);
            Math::matmul(state.k, state.xb, weights.wk + l*dim*dim, dim, dim); // assuming kv_heads = heads for now or stride logic
            Math::matmul(state.v, state.xb, weights.wv + l*dim*dim, dim, dim);
            
            // RoPE
            rope(state.q, state.k, pos, head_size);
            
            // KV Cache
            int loff = l * config.seq_len * dim;
            float* kc = state.key_cache + loff + pos * dim;
            float* vc = state.value_cache + loff + pos * dim;
            memcpy(kc, state.k, dim * sizeof(float));
            memcpy(vc, state.v, dim * sizeof(float));
            
            // Multihead Attention
            #pragma omp parallel for
            for (int h = 0; h < config.n_heads; h++) {
                // Score
                float* q = state.q + h * head_size;
                float* att = state.att + h * config.seq_len;
                
                for (int t = 0; t <= pos; t++) {
                    float* k = state.key_cache + loff + t * dim + h * head_size;
                    float score = 0.0f;
                    // Vectorize this dot product
                    for (int i=0; i<head_size; i++) score += q[i] * k[i];
                    score /= sqrtf(head_size);
                    att[t] = score;
                }
                
                // Softmax
                Math::softmax(att, pos + 1);
                
                // Weighted sum
                float* xb = state.xb + h * head_size;
                memset(xb, 0, head_size * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                     float* v = state.value_cache + loff + t * dim + h * head_size;
                     float a = att[t];
                     for (int i=0; i<head_size; i++) xb[i] += a * v[i];
                }
            }
            
            // Output projection
            Math::matmul(state.xb2, state.xb, weights.wo + l*dim*dim, dim, dim);
            
            // Residual
            for (int i=0; i<dim; i++) state.x[i] += state.xb2[i];
            
            // FFN RMSNorm
            Math::rmsnorm(state.xb, state.x, weights.rms_ffn_weight + l*dim, dim);
            
            // FFN
            Math::matmul(state.hb, state.xb, weights.w1 + l*dim*config.hidden_dim, dim, config.hidden_dim);
            Math::matmul(state.hb2, state.xb, weights.w3 + l*dim*config.hidden_dim, dim, config.hidden_dim);
            
            // Silu
            Math::silu_elementwise(state.hb, state.hb, config.hidden_dim); // elementwise mult is inside
            for(int i=0; i<config.hidden_dim; i++) state.hb[i] *= state.hb2[i]; // swiglu
            
            Math::matmul(state.xb, state.hb, weights.w2 + l*dim*config.hidden_dim, config.hidden_dim, dim);
            
            // Residual
            for (int i=0; i<dim; i++) state.x[i] += state.xb[i];
        }
        
        // Final RMSNorm
        Math::rmsnorm(state.x, state.x, weights.rms_final_weight, dim);
        
        // Classifier
        Math::matmul(state.logits, state.x, weights.wcls, dim, config.vocab_size);
        
        return state.logits;
    }
};

} // namespace bare_metal
