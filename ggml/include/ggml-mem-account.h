// Opt-in, model-independent accounting of ggml backend buffers and shared backings.
// This is diagnostic bookkeeping, not an allocator or an RSS measurement.
#pragma once

#include "ggml.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Register a backend buffer type whose buffers are suballocations of externally
// owned physical memory. Must be called before the first buffer of that type.
GGML_API void ggml_mem_account_external_buffer_type(const void * buft);

// A backing is allocated exactly once, regardless of the number of mapped
// views or child backend buffers. 'token' is stable until pool_destroy.
GGML_API void ggml_mem_account_pool_create(const void * token, const char * name, size_t allocated_bytes);
GGML_API void ggml_mem_account_pool_destroy(const void * token);

// Register a child backend buffer in a previously registered physical backing.
GGML_API void ggml_mem_account_buffer_attach(const void * buffer, const void * token, size_t offset, size_t bytes);

// Driver-side mapping registrations (NOT distinct physical allocations).
GGML_API void ggml_mem_account_mapping_change(const void * token, int delta);

// Observe the allocator's high-water mark; bytes may include temporary mirrors
// that are not represented as persistent ggml_backend_buffer_t objects.
GGML_API void ggml_mem_account_pool_high_water(const void * token, size_t bytes);

// Print an accounting snapshot. No-op unless GGML_MEM_ACCOUNTING=1.
GGML_API void ggml_mem_account_snapshot(const char * stage);

#ifdef __cplusplus
}
#endif
