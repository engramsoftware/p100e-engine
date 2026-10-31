// p100e-server --p100e-selftest: after the model is loaded, send a scripted request set to the server itself over
// localhost, print every response (status, content type, body or raw SSE stream) and a summary, then the server exits.
// Lets one command line (one exe, output = stdout) exercise the whole HTTP + engine path on the P100s.
#pragma once

#include <string>

// spec: "builtin" (the request set in selftest.cpp) or a JSON file: [{"name", "method", "path", "body",
// "expect_status", "cancel_after_chunks"}]. Returns 0 when every request got its expected status.
int p100e_selftest(const std::string & host, int port, const std::string & spec);
