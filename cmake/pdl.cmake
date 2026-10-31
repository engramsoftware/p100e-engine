# pdl = raw-Ethernet all-reduce protocol + zero-copy GPU rings. Host code only in p100e_pdl; CUDA in p100e_pdl_gpu.
# p100e_pdl also holds net_link (TCP), the transport of tp_model, so it comes from one library: p100e_core links it
# (CMakeLists.txt). One copy of net_link.cpp avoids a p100e_core <-> p100e_pdl static-link cycle on Linux.
add_library(p100e_pdl STATIC
    src/net_link.cpp
    src/pdl/pdl_util.cpp
    src/pdl/pdl_proto.cpp
    src/pdl/pdl_link.cpp
)
target_include_directories(p100e_pdl PUBLIC src)
find_package(Threads REQUIRED)
target_link_libraries(p100e_pdl PUBLIC Threads::Threads)
if(WIN32)
    target_link_libraries(p100e_pdl PUBLIC ws2_32)
endif()

add_library(p100e_pdl_gpu STATIC src/pdl/pdl_gpu.cu)
target_include_directories(p100e_pdl_gpu PUBLIC src)
target_link_libraries(p100e_pdl_gpu PUBLIC p100e_pdl CUDA::cudart)
