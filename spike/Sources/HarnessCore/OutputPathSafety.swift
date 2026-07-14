import Foundation

/// Symlink output destinations are intentionally unsupported. In particular, Foundation does
/// not resolve a dangling final symlink, so an alias check can otherwise change its answer after
/// another configured output creates the symlink target.
public func outputPathIsSymbolicLink(_ path: String) -> Bool {
    let standardizedPath = URL(fileURLWithPath: path).standardizedFileURL.path
    return (try? FileManager.default.destinationOfSymbolicLink(atPath: standardizedPath)) != nil
}

/// Returns true when two configured output paths can resolve to the same filesystem object.
///
/// The inode check covers existing hard links and symlinks. The volume-aware comparison also
/// covers case variants that do not exist yet on the default case-insensitive APFS filesystem;
/// waiting until both files exist is too late for append-only evidence and atomic progress files.
public func outputPathsReferToSameFile(_ left: String, _ right: String) -> Bool {
    let leftURL = URL(fileURLWithPath: left).standardizedFileURL.resolvingSymlinksInPath()
    let rightURL = URL(fileURLWithPath: right).standardizedFileURL.resolvingSymlinksInPath()
    if leftURL == rightURL { return true }

    let leftCaseSensitive = volumeSupportsCaseSensitiveNames(containing: leftURL)
    let rightCaseSensitive = volumeSupportsCaseSensitiveNames(containing: rightURL)
    if leftCaseSensitive == false, rightCaseSensitive == false,
        leftURL.path.compare(rightURL.path, options: [.caseInsensitive]) == .orderedSame
    {
        return true
    }

    let fileManager = FileManager.default
    guard let leftAttributes = try? fileManager.attributesOfItem(atPath: leftURL.path),
        let rightAttributes = try? fileManager.attributesOfItem(atPath: rightURL.path),
        let leftDevice = leftAttributes[.systemNumber] as? NSNumber,
        let rightDevice = rightAttributes[.systemNumber] as? NSNumber,
        let leftInode = leftAttributes[.systemFileNumber] as? NSNumber,
        let rightInode = rightAttributes[.systemFileNumber] as? NSNumber
    else {
        return false
    }
    return leftDevice == rightDevice && leftInode == rightInode
}

private func volumeSupportsCaseSensitiveNames(containing url: URL) -> Bool? {
    let fileManager = FileManager.default
    var candidate = url
    while !fileManager.fileExists(atPath: candidate.path) {
        let parent = candidate.deletingLastPathComponent()
        guard parent.path != candidate.path else { return nil }
        candidate = parent
    }
    return try? candidate.resourceValues(
        forKeys: [.volumeSupportsCaseSensitiveNamesKey]
    ).volumeSupportsCaseSensitiveNames
}
