/*
 * main.cpp 
 * Evaluation Driver for the Inference Engine.
 * Measures token generation latency with microsecond precision.
 */

#include "inference_engine.hpp"
#include <ctime>
#include <vector>
#include <string>

// Simple Tokenizer (Minimal implementation for benchmark)
struct Tokenizer {
    char** vocab;
    int vocab_size;

    Tokenizer(const char* path, int n) : vocab_size(n) {
        vocab = (char**)malloc(n * sizeof(char*));
        FILE* f = fopen(path, "rb");
        if (f) {
            for(int i=0; i<n; i++) {
                int len;
                if(fread(&len, sizeof(int), 1, f) != 1) break;
                vocab[i] = (char*)malloc(len + 1);
                if(fread(vocab[i], len, 1, f) != 1) break;
                vocab[i][len] = '\0';
                
                // Read score float
                float score;
                fread(&score, sizeof(float), 1, f);
            }
            fclose(f);
        } else {
            // Fallback if missing
            for(int i=0; i<n; i++) {
                vocab[i] = (char*)malloc(8);
                snprintf(vocab[i], 8, "[%d]", i);
            }
        }
    }
};

long time_in_us() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000000 + t.tv_nsec / 1000;
}

int sample(float* logits, int size) {
    // Greedy argmax for benchmark stability
    int max_i = 0;
    float max_p = logits[0];
    for (int i=1; i<size; i++) {
        if (logits[i] > max_p) {
            max_p = logits[i];
            max_i = i;
        }
    }
    return max_i;
}

int main(int argc, char** argv) {
    const char* model_path = (argc > 1) ? argv[1] : "stories110M.bin";
    const char* tok_path   = (argc > 2) ? argv[2] : "tokenizer.bin";
    const int STEPS = 256;

    printf("==========================================\n");
    printf(" Bare-Metal Inference Engine (NEON+SoA)\n");
    printf(" Model:     %s\n", model_path);
    printf(" Tokenizer: %s\n", tok_path);
    printf("==========================================\n");

    // Check file existence
    if (access(model_path, F_OK) == -1) {
        fprintf(stderr, "Error: Model file '%s' not found.\n", model_path);
        fprintf(stderr, "Download it from: https://huggingface.co/karpathy/tinyllamas/resolve/main/stories100m.bin\n");
        return 1;
    }

    // 1. Load Model
    printf("[*] Mapping High-Dimensional Tensors... "); fflush(stdout);
    bare_metal::Transformer transformer(model_path);
    printf("OK.\n");
    printf("    -> Dimension: %d\n", transformer.config.dim);
    printf("    -> Layers:    %d\n", transformer.config.n_layers);
    printf("    -> Heads:     %d\n", transformer.config.n_heads);

    // 2. Initialize Tokenizer (optional for perf, good for demo)
    // Tokenizer tokenizer(tok_path, transformer.config.vocab_size);

    // 3. Benchmark Loop
    printf("[*] Generating %d tokens...\n", STEPS);
    
    // Output file for latency plot
    FILE* fp = fopen("benchmark_results.csv", "w");
    fprintf(fp, "token_id,latency_us\n");
    
    int token = 1; // BOS
    long start_total = time_in_us();
    
    for (int i = 0; i < STEPS; i++) {
        long t0 = time_in_us();
        
        float* logits = transformer.forward(token, i);
        token = sample(logits, transformer.config.vocab_size);
        
        long t1 = time_in_us();
        
        fprintf(fp, "%d,%ld\n", i, t1 - t0);
        if (i % 10 == 0) { printf("."); fflush(stdout); }
    }
    
    long end_total = time_in_us();
    fclose(fp);
    
    printf("\n------------------------------------------\n");
    printf("Total Time: %.2f ms\n", (end_total - start_total) / 1000.0);
    printf("Tokens/Sec: %.2f\n", (double)STEPS / ((end_total - start_total) / 1000000.0));
    printf("------------------------------------------\n");

    return 0;
}
