import Foundation
import os

final class ServingChannelWritabilityGate: Sendable {
    enum GateError: Error {
        case backpressureTimeout
    }

    private let writable: OSAllocatedUnfairLock<Bool>

    init(initiallyWritable: Bool) {
        writable = OSAllocatedUnfairLock(initialState: initiallyWritable)
    }

    func update(isWritable: Bool) {
        writable.withLock { $0 = isWritable }
    }

    func waitUntilWritable(timeout: Duration) async throws {
        guard !isWritable else {
            return
        }

        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        while !isWritable {
            try Task.checkCancellation()
            guard clock.now < deadline else {
                throw GateError.backpressureTimeout
            }
            try await Task.sleep(for: .milliseconds(1))
        }
    }

    private var isWritable: Bool {
        writable.withLock { $0 }
    }
}
