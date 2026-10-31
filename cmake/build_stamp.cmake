# p100e-server /props build_info: this tree's git commit (+ "-dirty"), stamped at configure time (build.cmd always
# reconfigures). Only server_context.cpp sees the define, so a new commit recompiles that one file.
if(TARGET p100e_server_obj)
    execute_process(COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse --short HEAD
                    OUTPUT_VARIABLE P100E_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(COMMAND git -C "${CMAKE_SOURCE_DIR}" status --porcelain --untracked-files=no
                    OUTPUT_VARIABLE P100E_GIT_DIRTY OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT P100E_GIT_COMMIT)
        set(P100E_GIT_COMMIT "unknown")
    endif()
    if(P100E_GIT_DIRTY)
        set(P100E_GIT_COMMIT "${P100E_GIT_COMMIT}-dirty")
    endif()
    set_source_files_properties(server/server_context.cpp PROPERTIES COMPILE_DEFINITIONS "P100E_GIT_COMMIT=\"${P100E_GIT_COMMIT}\"")
    message(STATUS "p100e-server build_info commit: ${P100E_GIT_COMMIT}")
endif()
