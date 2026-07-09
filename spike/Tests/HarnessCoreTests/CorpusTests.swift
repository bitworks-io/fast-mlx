import XCTest
@testable import HarnessCore

final class CorpusTests: XCTestCase {
  func testUniversalInvariantsHoldAcrossCorpus() throws {
    for e in HarnessCorpus.entries {
      let out = HarnessCorpus.process(e.raw)         // pure: strip think-tags, parse tool calls
      XCTAssertFalse(out.visibleText.contains("<|"), "control-tag leak in \(e.name)")
      if let args = out.toolArgsJSON { XCTAssertNoThrow(try JSONSerialization.jsonObject(with: Data(args.utf8))) }
      if let exp = e.expectedVisible { XCTAssertEqual(out.visibleText, exp, e.name) }
    }
  }
}
