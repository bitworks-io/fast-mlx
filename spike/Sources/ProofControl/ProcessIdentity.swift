import CryptoKit
import Darwin
import Foundation

public enum ProcessIdentityError: Error, Equatable, Sendable {
    case bootTimeUnavailable
    case processInfoUnavailable(pid: Int32)
    case executablePathUnavailable(pid: Int32)
    case executableUnreadable(pid: Int32)
}

/// OS-authenticated local process identity facts collected exclusively through
/// kernel-reported interfaces (`sysctl KERN_BOOTTIME`, `proc_pidinfo`,
/// `proc_pidpath`). These are observation primitives only: they grant no run
/// authority, and a caller-supplied pid proves nothing about ancestry — a
/// trust chain must observe a process it spawned itself.
public enum ProcessIdentity {
    public static func bootTimeUnixSeconds() throws -> Int {
        var value = timeval()
        var size = MemoryLayout<timeval>.stride
        var mib: [Int32] = [CTL_KERN, KERN_BOOTTIME]
        let result = mib.withUnsafeMutableBufferPointer {
            sysctl($0.baseAddress, u_int($0.count), &value, &size, nil, 0)
        }
        guard result == 0 else {
            throw ProcessIdentityError.bootTimeUnavailable
        }
        return Int(value.tv_sec)
    }

    public static func processStartUptimeNanoseconds(pid: pid_t) throws -> UInt64 {
        var info = proc_bsdinfo()
        let result = proc_pidinfo(
            pid,
            PROC_PIDTBSDINFO,
            0,
            &info,
            Int32(MemoryLayout<proc_bsdinfo>.stride))
        guard result == Int32(MemoryLayout<proc_bsdinfo>.stride) else {
            throw ProcessIdentityError.processInfoUnavailable(pid: pid)
        }
        let boot = try bootTimeUnixSeconds()
        let seconds = max(1, Int(info.pbi_start_tvsec) - boot)
        return UInt64(seconds) * 1_000_000_000 + UInt64(info.pbi_start_tvusec) * 1_000
    }

    /// Kernel-reported parent process ID for `pid` (`proc_bsdinfo.pbi_ppid`).
    /// A trust chain that spawned a child itself uses this to REQUIRE the
    /// observed parent to be its own pid — a caller-supplied or recycled pid
    /// whose kernel-reported parent is someone else must fail closed, because
    /// evidence minted for it would describe a process the chain never
    /// spawned.
    public static func parentProcessID(pid: pid_t) throws -> pid_t {
        var info = proc_bsdinfo()
        let result = proc_pidinfo(
            pid,
            PROC_PIDTBSDINFO,
            0,
            &info,
            Int32(MemoryLayout<proc_bsdinfo>.stride))
        guard result == Int32(MemoryLayout<proc_bsdinfo>.stride) else {
            throw ProcessIdentityError.processInfoUnavailable(pid: pid)
        }
        return pid_t(info.pbi_ppid)
    }

    public static func executablePath(pid: pid_t) throws -> String {
        // proc_pidpath fails with EOVERFLOW for buffer sizes LARGER than
        // PROC_PIDPATHINFO_MAXSIZE (4 * MAXPATHLEN = 4096) — the kernel
        // treats the limit as an exact upper bound, not a minimum.
        var buffer = [CChar](repeating: 0, count: 4_096)
        let length = proc_pidpath(pid, &buffer, UInt32(buffer.count))
        guard length > 0 else {
            throw ProcessIdentityError.executablePathUnavailable(pid: pid)
        }
        let pathBytes = buffer.prefix(Int(length)).map { UInt8(bitPattern: $0) }
        return String(decoding: pathBytes, as: UTF8.self)
    }

    public static func executableSHA256(pid: pid_t) throws -> String {
        let path = try executablePath(pid: pid)
        let data: Data
        do {
            data = try Data(contentsOf: URL(fileURLWithPath: path))
        } catch {
            throw ProcessIdentityError.executableUnreadable(pid: pid)
        }
        return sha256Hex(data)
    }

    public static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
