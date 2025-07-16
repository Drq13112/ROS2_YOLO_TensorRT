#include <cuda_runtime.h>
#include <stdint.h>

/*
 * Kernel para el preprocesado de la imagen.
 * Convierte una imagen de entrada (HWC, uint8) a un tensor de salida (CHW, float32),
 * aplicando normalización (resta la media y divide por la desviación estándar).
 */
__global__ void preprocess_kernel_impl(const uint8_t* src, float* dst, int width, int height, const float3 mean, const float3 std) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Índice de píxel en el eje X
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Índice de píxel en el eje Y

    if (x >= width || y >= height) {
        return; // Si el índice está fuera de los límites de la imagen, salir del kernel
    }

    int hwc_idx = (y * width + x) * 3; // Índice para formato HWC (B, G, R)
    
    // OpenCV usa BGR, pero el modelo espera RGB. El colormap en C++ también está en RGB.
    // Asumimos que la entrada `src` es BGR y la convertimos a RGB mientras normalizamos.
    uint8_t b = src[hwc_idx + 0];
    uint8_t g = src[hwc_idx + 1];
    uint8_t r = src[hwc_idx + 2];

    // Normalización y conversión a float
    float r_norm = (static_cast<float>(r) / 255.0f - mean.x) / std.x;
    float g_norm = (static_cast<float>(g) / 255.0f - mean.y) / std.y;
    float b_norm = (static_cast<float>(b) / 255.0f - mean.z) / std.z;

    // Escritura en formato CHW
    int channel_size = width * height;
    dst[0 * channel_size + y * width + x] = r_norm; // Canal R
    dst[1 * channel_size + y * width + x] = g_norm; // Canal G
    dst[2 * channel_size + y * width + x] = b_norm; // Canal B
}

/*
 * Kernel para el post-procesado (Argmax).
 * Para cada píxel, encuentra la clase con la máxima probabilidad.
 * La entrada `src` es el tensor de salida del modelo (CHW, float32).
 * La salida `dst` es un mapa de segmentación 2D (HW, uint8).
 */
__global__ void argmax_kernel_impl(const float* src, uint8_t* dst, int width, int height, int num_classes) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Índice de píxel en el eje X
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Índice de píxel en el eje Y

    if (x >= width || y >= height) {
        return;
    }

    float max_val = -1.0e6f; // Un número negativo muy grande
    uint8_t max_idx = 0;
    int pixel_offset = y * width + x;
    int channel_size = width * height;

    // Iterar sobre todas las clases para encontrar la de mayor valor para este píxel
    for (int c = 0; c < num_classes; ++c) {
        float val = src[c * channel_size + pixel_offset];
        if (val > max_val) {
            max_val = val;
            max_idx = static_cast<uint8_t>(c);
        }
    }

    // Escribir el índice de la clase ganadora en el mapa de salida
    dst[pixel_offset] = max_idx;
}

/*
 * Kernel para el post-procesado (Argmax con Confianza Empaquetada).
 * Para cada píxel, encuentra la clase con la máxima probabilidad y el valor de esa probabilidad.
 * La confianza (float) se reescala a uint8_t.
 * La salida `dst` es un mapa 2D de 2 canales (HW, uint8_t x 2), donde:
 *   - Canal 0: ID de la clase (uint8)
 *   - Canal 1: Confianza reescalada (uint8)
 */
__global__ void argmax_with_confidence_kernel_impl(const float* src, uchar2* dst, int width, int height, int num_classes) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float max_val = -1.0e6f; // Un número negativo muy grande
    uint8_t max_idx = 0;
    int pixel_offset = y * width + x;
    int channel_size = width * height;

    // Iterar sobre todas las clases para encontrar la de mayor valor para este píxel
    for (int c = 0; c < num_classes; ++c) {
        float val = src[c * channel_size + pixel_offset];
        if (val > max_val) {
            max_val = val;
            max_idx = static_cast<uint8_t>(c);
        }
    }

    // Reescalar la confianza de float a uint8.
    // Usamos una función sigmoide para mapear el logit (que puede ser cualquier float)
    // a un rango [0, 1] y luego lo escalamos a [0, 100].
    // Esto maneja bien tanto valores positivos como negativos.
    float confidence_prob = 1.0f / (1.0f + expf(-max_val));
    uint8_t confidence_uint8 = static_cast<uint8_t>(confidence_prob * 100.0f);

    // Escribir el par (clase, confianza) en la memoria de salida usando uchar2.
    // Esto es más eficiente que hacer dos escrituras separadas.
    dst[pixel_offset] = make_uchar2(max_idx, confidence_uint8);
}


// --- Funciones Wrapper ---
// Estas son las funciones que se llaman desde el código C++.
// Se encargan de configurar y lanzar los kernels.

extern "C" void preprocess_kernel(const uint8_t* src, float* dst, int width, int height, const float3 mean, const float3 std) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    preprocess_kernel_impl<<<grid, block>>>(src, dst, width, height, mean, std);
}

extern "C" void argmax_kernel(const float* src, uint8_t* dst, int width, int height, int num_classes) {
    dim3 block(16, 16); // Definir el tamaño del bloque multiplo de 32 para aprovechar la arquitectura CUDA
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y); // Calcular el número de bloques necesarios para cubrir toda la imagen
    // Se redondea hacia arriba el número de bloques para asegurarse de que cubrimos toda la imagen
    // Lanzar el kernel de Argmax
    argmax_kernel_impl<<<grid, block>>>(src, dst, width, height, num_classes);
}

extern "C" void argmax_with_confidence_kernel(const float* src, uchar2* dst, int width, int height, int num_classes) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    argmax_with_confidence_kernel_impl<<<grid, block>>>(src, dst, width, height, num_classes);
}