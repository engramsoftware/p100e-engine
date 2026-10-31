# p100e-server: check that the llama.cpp fork tree is the commit the server was verified against.
#
# cmake/server.cmake runs this at configure time (include) and before every link of p100e-server (cmake -P, target
# p100e_fork_check), so a fork tree that moves after the configure is caught too. Inputs:
#   FORK_DIR     the fork source tree (P100E_FORK_DIR)
#   FORK_COMMIT  required commit prefix (P100E_FORK_COMMIT, 98ab7e5 = the commit the server was verified against)
# Fails (FATAL_ERROR) unless
#   - git -C FORK_DIR rev-parse HEAD starts with FORK_COMMIT, and
#   - git status shows nothing under the fork paths the server build compiles or reads (below): no staged or unstaged
#     change to a tracked file and no untracked file (src/CMakeLists.txt globs src/models/*.cpp, so a stray new file
#     there would be compiled). Ignored files (build trees) are not reported by git status and do not count.
# Escape hatch for a deliberate fork change: -DP100E_FORK_CHECK=OFF (the server's "same behaviour as llama-server" claims then
# no longer hold), or -DP100E_SERVER=OFF to build the engine without the server.

if(NOT FORK_DIR OR NOT FORK_COMMIT)
    message(FATAL_ERROR "check_fork.cmake: FORK_DIR and FORK_COMMIT are required")
endif()

# what add_subdirectory(<fork>) with the server's options compiles (ggml CPU, libllama, libcommon, mtmd, vendor/*),
# the fork server sources server.cmake lists, and the license file the fork's CMake embeds
set(_p100e_fork_paths CMakeLists.txt cmake ggml src include common vendor tools/server tools/mtmd licenses LICENSE)

find_program(P100E_GIT_EXE git)
if(NOT P100E_GIT_EXE)
    message(FATAL_ERROR "p100e-server: git not found, cannot check the fork commit of ${FORK_DIR} "
                        "(need ${FORK_COMMIT}); configure with -DP100E_FORK_CHECK=OFF to skip the check")
endif()

execute_process(COMMAND "${P100E_GIT_EXE}" -C "${FORK_DIR}" rev-parse HEAD
                OUTPUT_VARIABLE _p100e_fork_head ERROR_VARIABLE _p100e_fork_err RESULT_VARIABLE _p100e_fork_rc
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _p100e_fork_rc EQUAL 0)
    message(FATAL_ERROR "p100e-server: git rev-parse HEAD failed in ${FORK_DIR}: ${_p100e_fork_err}"
                        "configure with -DP100E_FORK_CHECK=OFF to skip the check")
endif()

string(LENGTH "${FORK_COMMIT}" _p100e_n)
string(SUBSTRING "${_p100e_fork_head}" 0 ${_p100e_n} _p100e_head_prefix)
if(NOT _p100e_head_prefix STREQUAL FORK_COMMIT)
    message(FATAL_ERROR "p100e-server: the fork at ${FORK_DIR} is at ${_p100e_fork_head}, not ${FORK_COMMIT}. "
                        "p100e-server compiles the fork's tokenizer / templates / parsers / server sources and its "
                        "behaviour was verified against ${FORK_COMMIT}. Check out "
                        "${FORK_COMMIT} there, point -DP100E_FORK_DIR at a tree at that commit, or configure with "
                        "-DP100E_FORK_CHECK=OFF (deliberate fork change) / -DP100E_SERVER=OFF (no server).")
endif()

execute_process(COMMAND "${P100E_GIT_EXE}" -C "${FORK_DIR}" status --porcelain --untracked-files=normal -- ${_p100e_fork_paths}
                OUTPUT_VARIABLE _p100e_fork_status ERROR_VARIABLE _p100e_fork_err RESULT_VARIABLE _p100e_fork_rc
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _p100e_fork_rc EQUAL 0)
    message(FATAL_ERROR "p100e-server: git status failed in ${FORK_DIR}: ${_p100e_fork_err}"
                        "configure with -DP100E_FORK_CHECK=OFF to skip the check")
endif()
if(NOT _p100e_fork_status STREQUAL "")
    message(FATAL_ERROR "p100e-server: the fork at ${FORK_DIR} (${FORK_COMMIT}) has local changes in files the server "
                        "build compiles:\n${_p100e_fork_status}\n"
                        "Revert them, or configure with -DP100E_FORK_CHECK=OFF (deliberate fork change) / "
                        "-DP100E_SERVER=OFF (no server).")
endif()

if(NOT CMAKE_SCRIPT_MODE_FILE)
    message(STATUS "p100e-server: fork ${FORK_DIR} at ${_p100e_fork_head}, clean")
endif()
