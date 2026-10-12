import CryptoKit
import Darwin
import Foundation

/// A `fastmlx pull` receipt, loaded once from its sibling `<dir>.pull-receipt.json`.
///
/// Same validity rule as `scripts/fastmlx_launch.py` `_load_pull_receipt`: a missing, unreadable,
/// non-JSON, or non-object file, or one whose recorded `dest` resolves to a different directory,
/// reads as "no receipt"; a receipt with no `dest` is trusted (the sibling naming ties it to the path).
struct PackReceipt {
    let url: URL
    let object: [String: Any]

    static func load(for directory: URL) -> PackReceipt? {
        let absolute = URL(fileURLWithPath: directory.path)
        let receiptURL = URL(fileURLWithPath: absolute.path + ".pull-receipt.json")
        guard let data = try? Data(contentsOf: receiptURL),
            let json = try? JSONSerialization.jsonObject(with: data),
            let object = json as? [String: Any]
        else { return nil }
        if let recordedDest = object["dest"] as? String {
            let recorded = URL(fileURLWithPath: recordedDest).resolvingSymlinksInPath().path
            guard recorded == absolute.resolvingSymlinksInPath().path else { return nil }
        }
        return PackReceipt(url: receiptURL, object: object)
    }

    /// Non-empty `repo_id`, else `nil`.
    var repo: String? { (object["repo_id"] as? String).flatMap { $0.isEmpty ? nil : $0 } }
    /// Non-empty `revision`, else `nil`.
    var revision: String? { (object["revision"] as? String).flatMap { $0.isEmpty ? nil : $0 } }
}

/// Checks a pack directory's bytes against the per-file `files` map of its pull receipt. The port of
/// `scripts/fastmlx_launch.py` `verify_receipt_binding` / `judge_receipt_binding`; the shared
/// conformance table `Fixtures/receipt-binding-conformance-v1.json` holds the two to one rule.
///
/// Every listed file must exist (symlinks followed) with the recorded size -- and, when that size is
/// at most `hashMaxBytes`, the recorded sha256 -- and the directory must hold no unlisted path
/// outside dot-components. Listed files are checked first, in sorted order, then the unlisted-path
/// walk in sorted order; the first difference is returned. A larger file is checked by size only and
/// is never opened: an accident guard, not an adversary guard.
enum PackReceiptBinding {
    /// Files at or under this size are sha256-compared; the Python `RECEIPT_HASH_MAX_BYTES`.
    static let defaultHashMaxBytes: Int64 = PackAdmission.defaultReceiptHashMaxBytes

    struct Difference: Equatable {
        enum Kind: String, Equatable {
            case missing, size, sha256, unreadable, unlisted, unlistable
        }
        let kind: Kind
        /// The receipt-relative name the difference is about (raw, not quoted).
        let path: String
        /// The one-line human description, e.g. `'a.bin' size differs (receipt 3, found 4)`.
        let detail: String
    }

    enum Outcome: Equatable {
        case ok
        case mismatch(Difference)
        /// The receipt has no usable `files` map; the associated text is the reason.
        case uncheckable(String)
    }

    enum Judgement: Equatable {
        case none
        /// A complete notice line WITHOUT the `fastmlx-serve: ` prefix.
        case notice(String)
        /// The refusal `detail=` text.
        case refuse(detail: String)
    }

    // MARK: verify

    /// A receipt-relative name compared by its UTF-8 bytes. Python compares `str` by code point; a
    /// Swift `String` compares by canonical equivalence (NFC "caf\u{E9}" == NFD "cafe\u{301}"), so a
    /// `String` key would merge two names Python keeps apart.
    private struct Name: Hashable {
        let bytes: [UInt8]

        init(_ text: String) { bytes = Array(text.utf8) }
        init(bytes: [UInt8]) { self.bytes = bytes }

        var text: String { String(decoding: bytes, as: UTF8.self) }
        var startsWithDot: Bool { bytes.first == 0x2E }

        static func precedes(_ lhs: Name, _ rhs: Name) -> Bool {
            lhs.bytes.lexicographicallyPrecedes(rhs.bytes)
        }

        /// Slash-separated parts, empty parts kept (Python `str.split("/")`).
        var parts: [[UInt8]] {
            bytes.split(separator: 0x2F, omittingEmptySubsequences: false).map(Array.init)
        }

        func appending(_ child: Name) -> Name {
            bytes.isEmpty ? child : Name(bytes: bytes + [0x2F] + child.bytes)
        }
    }

    private struct Listed {
        let sha256: String
        let size: Int64
        /// The size as the receipt wrote it, for messages (a size beyond `Int64` clamps in `size`).
        let sizeText: String
    }

    static func verify(
        directory: URL, receipt: [String: Any], hashMaxBytes: Int64 = defaultHashMaxBytes
    ) -> Outcome {
        let listed: [Name: Listed]
        switch listedFiles(receipt) {
        case .failure(let why): return .uncheckable(why.text)
        case .success(let value): listed = value
        }
        let root = directory.path
        for key in listed.keys.sorted(by: Name.precedes) {
            guard let expected = listed[key] else { continue }
            let name = key.text
            let path = root + "/" + name
            let label = quoted(name, 120)
            var info = stat()
            if fileStatus(Array(path.utf8), followingSymlinks: true, into: &info) != 0 {
                let code = errno
                if code == ENOENT || code == ENOTDIR || code == ELOOP {
                    return .mismatch(
                        Difference(
                            kind: .missing, path: name,
                            detail: "\(label) is listed in the receipt but missing"))
                }
                return .mismatch(unreadable(name, label, code))
            }
            guard (info.st_mode & S_IFMT) == S_IFREG else {
                return .mismatch(
                    Difference(
                        kind: .missing, path: name,
                        detail: "\(label) is listed in the receipt but missing"))
            }
            let observed = Int64(info.st_size)
            if observed != expected.size {
                return .mismatch(
                    Difference(
                        kind: .size, path: name,
                        detail: "\(label) size differs (receipt \(expected.sizeText), found \(observed))"
                    ))
            }
            if expected.size <= hashMaxBytes {
                switch sha256Hex(ofFileAt: path) {
                case .failure(let code): return .mismatch(unreadable(name, label, code.code))
                case .success(let digest):
                    if digest != expected.sha256 {
                        return .mismatch(
                            Difference(
                                kind: .sha256, path: name,
                                detail:
                                    "\(label) sha256 differs (receipt \(expected.sha256.prefix(12))..., found \(digest.prefix(12))...)"
                            ))
                    }
                }
            }
        }

        var listedPrefixes: Set<Name> = []
        for key in listed.keys {
            let parts = key.parts
            if parts.count > 1 {
                for depth in 1..<parts.count {
                    listedPrefixes.insert(
                        Name(bytes: Array(parts[0..<depth].joined(separator: [0x2F]))))
                }
            }
        }
        if let extra = firstUnlisted(
            directory: Array(root.utf8), prefix: Name(bytes: []), listed: listed,
            listedPrefixes: listedPrefixes)
        {
            return .mismatch(extra)
        }
        return .ok
    }

    private static func unreadable(_ name: String, _ label: String, _ code: Int32) -> Difference {
        Difference(
            kind: .unreadable, path: name,
            detail: "\(label) could not be read (\(String(cString: strerror(code))))")
    }

    /// The child names of `directory` (no `.` / `..`), or the `errno` that stopped the listing.
    private static func childNames(of directory: [UInt8]) -> Swift.Result<[Name], ReadFailure> {
        guard let handle = withCPath(directory, { opendir($0) }) else {
            return .failure(ReadFailure(code: errno))
        }
        defer { closedir(handle) }
        var names: [Name] = []
        errno = 0
        while let entry = readdir(handle) {
            let bytes = withUnsafePointer(to: &entry.pointee.d_name) { pointer in
                pointer.withMemoryRebound(to: UInt8.self, capacity: Int(entry.pointee.d_namlen)) {
                    Array(UnsafeBufferPointer(start: $0, count: Int(entry.pointee.d_namlen)))
                }
            }
            if bytes == [0x2E] || bytes == [0x2E, 0x2E] { continue }
            names.append(Name(bytes: bytes))
            errno = 0
        }
        if errno != 0 { return .failure(ReadFailure(code: errno)) }
        return .success(names)
    }

    private static func firstUnlisted(
        directory: [UInt8], prefix: Name, listed: [Name: Listed], listedPrefixes: Set<Name>
    ) -> Difference? {
        let children: [Name]
        switch childNames(of: directory) {
        case .failure(let failure):
            let shown = prefix.bytes.isEmpty ? "." : prefix.text
            return Difference(
                kind: .unlistable, path: shown,
                detail:
                    "\(oneLine(shown)) could not be listed (\(String(cString: strerror(failure.code))))")
        case .success(let names):
            children = names.sorted(by: Name.precedes)
        }
        for child in children {
            if child.startsWithDot { continue }
            let relative = prefix.appending(child)
            if listed[relative] != nil { continue }
            let path = directory + [0x2F] + child.bytes
            if listedPrefixes.contains(relative) || isDirectory(path, followingSymlinks: false) {
                if isDirectory(path, followingSymlinks: true) {
                    if let found = firstUnlisted(
                        directory: path, prefix: relative, listed: listed,
                        listedPrefixes: listedPrefixes)
                    {
                        return found
                    }
                    continue
                }
            }
            return Difference(
                kind: .unlisted, path: relative.text,
                detail: "\(quoted(relative.text, 120)) is in the pack but not listed in the receipt")
        }
        return nil
    }

    private static func isDirectory(_ path: [UInt8], followingSymlinks: Bool) -> Bool {
        var info = stat()
        return fileStatus(path, followingSymlinks: followingSymlinks, into: &info) == 0
            && (info.st_mode & S_IFMT) == S_IFDIR
    }

    /// Runs `body` over `bytes` as a NUL-terminated C path (exact bytes, no normalization).
    private static func withCPath<T>(_ bytes: [UInt8], _ body: (UnsafePointer<CChar>) -> T) -> T {
        var terminated = bytes.map { CChar(bitPattern: $0) }
        terminated.append(0)
        return terminated.withUnsafeBufferPointer { body($0.baseAddress!) }
    }

    /// `stat(2)` / `lstat(2)`; `errno` is left set on failure.
    private static func fileStatus(
        _ path: [UInt8], followingSymlinks: Bool, into info: inout stat
    ) -> Int32 {
        var local = stat()
        let result = withCPath(path) { followingSymlinks ? stat($0, &local) : lstat($0, &local) }
        info = local
        return result
    }

    private struct ReadFailure: Error {
        let code: Int32
    }

    private static func sha256Hex(ofFileAt path: String) -> Swift.Result<String, ReadFailure> {
        let descriptor = open(path, O_RDONLY)
        guard descriptor >= 0 else { return .failure(ReadFailure(code: errno)) }
        defer { close(descriptor) }
        var hasher = SHA256()
        var buffer = [UInt8](repeating: 0, count: 1 << 20)
        while true {
            let count = buffer.withUnsafeMutableBytes { read(descriptor, $0.baseAddress, $0.count) }
            if count < 0 {
                if errno == EINTR { continue }
                return .failure(ReadFailure(code: errno))
            }
            if count == 0 { break }
            buffer.withUnsafeBytes { hasher.update(bufferPointer: UnsafeRawBufferPointer(rebasing: $0.prefix(count))) }
        }
        return .success(hasher.finalize().map { String(format: "%02x", $0) }.joined())
    }

    // MARK: files map

    private struct Unusable: Error {
        let text: String
    }

    private static func listedFiles(_ receipt: [String: Any]) -> Swift.Result<[Name: Listed], Unusable> {
        // NSDictionary, not [String: Any]: the Swift bridge merges canonically equivalent keys.
        guard let files = receipt["files"] as? NSDictionary, files.count > 0 else {
            return .failure(Unusable(text: "it records no files map"))
        }
        var keyed: [(Name, Any)] = []
        for key in files.allKeys {
            guard let text = key as? String else {
                return .failure(Unusable(text: "it lists a file with an unusable name"))
            }
            keyed.append((Name(text), key))
        }
        keyed.sort { Name.precedes($0.0, $1.0) }
        var listed: [Name: Listed] = [:]
        for (name, key) in keyed {
            let label = quoted(name.text, 80)
            if name.bytes.isEmpty {
                return .failure(Unusable(text: "it lists a file with an unusable name"))
            }
            if name.bytes.first == 0x2F
                || name.parts.contains(where: { $0.isEmpty || $0 == [0x2E] || $0 == [0x2E, 0x2E] })
            {
                return .failure(Unusable(text: "it lists a file with an unusable name \(label)"))
            }
            guard let entry = files[key] as? [String: Any] else {
                return .failure(Unusable(text: "its entry for \(label) is malformed"))
            }
            guard let size = integerSize(entry["size"]) else {
                return .failure(Unusable(text: "its entry for \(label) has no integer size"))
            }
            guard let sha = entry["sha256"] as? String, isSHA256Hex(sha) else {
                return .failure(Unusable(text: "its entry for \(label) has no sha256"))
            }
            listed[name] = Listed(sha256: sha.lowercased(), size: size.value, sizeText: size.text)
        }
        return .success(listed)
    }

    /// A JSON integer that is not a boolean, not a float, and not negative (Python: `isinstance(size,
    /// int)`, not `bool`, `>= 0`). `value` clamps an integer beyond `Int64` to `Int64.max`: no real
    /// file has that size, so it can only read as a size difference, as in Python. Foundation hands an
    /// integer beyond 64 bits back as an `NSDecimalNumber`, which `CFNumberIsFloatType` reports as a
    /// float, so it is recognised by class and an integral, non-negative value. (A decimal-point
    /// spelling of such a huge value, e.g. `1000000000000000000000000000000.0`, is indistinguishable
    /// from the integer after parsing and reads as a size difference where Python says "no integer
    /// size" -- a distinction that costs nothing, as both refuse or notice on a corrupt receipt.)
    private static func integerSize(_ value: Any?) -> (value: Int64, text: String)? {
        guard let number = value as? NSNumber else { return nil }
        if CFGetTypeID(number) == CFBooleanGetTypeID() { return nil }
        if number is NSDecimalNumber {
            let decimal = number.decimalValue
            var rounded = Decimal()
            var copy = decimal
            NSDecimalRound(&rounded, &copy, 0, .down)
            guard !decimal.isNaN, decimal >= 0, rounded == decimal else { return nil }
        } else if CFNumberIsFloatType(number) {
            return nil
        }
        if number.compare(NSNumber(value: 0)) == .orderedAscending { return nil }
        if number.compare(NSNumber(value: Int64.max)) == .orderedDescending {
            return (Int64.max, number.stringValue)
        }
        return (number.int64Value, number.stringValue)
    }

    private static func isSHA256Hex(_ value: String) -> Bool {
        value.utf8.count == 64
            && value.utf8.allSatisfy {
                (0x30...0x39).contains($0) || (0x41...0x46).contains($0) || (0x61...0x66).contains($0)
            }
    }

    // MARK: rendering

    /// A single-quoted, single-line rendering bounded to `maxLength` characters of body: printable
    /// ASCII is kept (backslash and quote escaped), every other scalar is escaped, and a longer value
    /// is cut and marked with `...`. Not an exact Python `repr`; a name can never break a notice
    /// across lines.
    static func quoted(_ value: String, _ maxLength: Int) -> String {
        var body = ""
        var truncated = false
        for scalar in value.unicodeScalars {
            let piece: String
            switch scalar.value {
            case 0x5C: piece = "\\\\"
            case 0x27: piece = "\\'"
            case 0x0A: piece = "\\n"
            case 0x0D: piece = "\\r"
            case 0x09: piece = "\\t"
            case 0x20...0x7E: piece = String(scalar)
            default: piece = "\\u{" + String(scalar.value, radix: 16) + "}"
            }
            if body.count + piece.count > maxLength {
                truncated = true
                break
            }
            body += piece
        }
        return "'" + body + (truncated ? "..." : "") + "'"
    }

    /// A filesystem path on one line: control characters become `?`; nothing is truncated.
    static func oneLine(_ value: String) -> String {
        var out = String.UnicodeScalarView()
        for scalar in value.unicodeScalars {
            let isControl =
                scalar.value < 0x20 || scalar.value == 0x7F || (0x80...0x9F).contains(scalar.value)
                || scalar.value == 0x2028 || scalar.value == 0x2029
            out.append(isControl ? "?" : scalar)
        }
        return String(out)
    }

    // MARK: judgement

    /// Judge the pack by its receipt only while the receipt still describes the pack's bytes. A
    /// mismatch is a `refuse` when the selected card matches the receipt's own identity (the receipt
    /// is then that card's evidence), otherwise a `notice`. A receipt that cannot be checked is a
    /// `notice` only; a matching pack is `none`. Pure: prints and throws nothing.
    static func judge(
        directory: URL, receipt: PackReceipt, card: QualityCard?,
        hashMaxBytes: Int64 = defaultHashMaxBytes
    ) -> Judgement {
        let receiptPath = oneLine(receipt.url.path)
        switch verify(directory: directory, receipt: receipt.object, hashMaxBytes: hashMaxBytes) {
        case .ok:
            return .none
        case .uncheckable(let why):
            return .notice(
                "pull receipt \(receiptPath) cannot be checked against the pack's bytes (\(why))")
        case .mismatch(let difference):
            if let card, cardMatchesReceiptIdentity(card, receipt) {
                return .refuse(
                    detail:
                        "the pack at \(oneLine(directory.path)) no longer matches its pull receipt \(receiptPath): \(difference.detail); its quality card describes the bytes that were pulled, not these -- re-run `fastmlx pull` for this directory, or remove the stale receipt so the pack is judged as unidentified"
                )
            }
            return .notice(
                "pull receipt \(receiptPath) does not match the pack's bytes (\(difference.detail)); the receipt is not evidence for any quality card here"
            )
        }
    }

    /// Mirrors Python `card_matches_model_identity` over the receipt's `repo_id` / `revision`: equal
    /// repos, or the card's `hfPin` prefix-matching a full-hex revision.
    private static func cardMatchesReceiptIdentity(_ card: QualityCard, _ receipt: PackReceipt) -> Bool {
        if let cardRepo = card.model.repo, let receiptRepo = receipt.repo, cardRepo == receiptRepo {
            return true
        }
        return QualityCardStore.hfPinMatchesRevision(hfPin: card.model.hfPin, revision: receipt.revision)
    }
}
