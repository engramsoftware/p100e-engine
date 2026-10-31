// pdl: the raw Ethernet frame port the protocol (pdl_proto.h) runs on. A frame here is dst MAC .. payload (no FCS: the
// hardware adds and checks it). Poll mode, single thread: nothing blocks.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace p100e::pdl {

class frame_port {
public:
    virtual ~frame_port() = default;
    virtual size_t max_frame() const = 0;           // largest frame in bytes (14-byte Ethernet header + MTU)
    virtual const uint8_t * mac() const = 0;        // this port's unicast MAC (6 bytes)

    // TX: room for one frame of len bytes (len <= max_frame()) in the current transmit batch, written in place by the
    // caller (zero copy into the USB transfer buffer). Frames shorter than 60 bytes are zero padded by the port.
    // nullptr = every transmit buffer is in flight (poll, then retry). Frames go out on tx_flush() (or when a batch
    // buffer fills up).
    virtual uint8_t * tx_frame(size_t len) = 0;
    virtual void tx_flush() = 0;
    virtual bool tx_idle() = 0;                     // no transmit transfer in flight

    // RX: reap completed transfers, hand every received frame to fn (the pointer is valid during the call only),
    // re-post the receive buffers. Returns the number of frames delivered; never blocks.
    using rx_fn = void (*)(void * ctx, const uint8_t * frame, size_t len);
    virtual int rx_poll(rx_fn fn, void * ctx) = 0;

    virtual bool link_up() const = 0;               // last known link state (refresh: the driver's poll_link)
    virtual std::string describe() const = 0;
};

} // namespace p100e::pdl
