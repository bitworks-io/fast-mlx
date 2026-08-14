import Foundation
import ProofControl

let status: [String: Any] = [
    "schema": ProofControl.schema,
    "status": "NOT_IMPLEMENTED",
    "promotable": false,
]
let data = try JSONSerialization.data(withJSONObject: status, options: [.sortedKeys])
FileHandle.standardError.write(data)
FileHandle.standardError.write(Data([0x0a]))
exit(78)
