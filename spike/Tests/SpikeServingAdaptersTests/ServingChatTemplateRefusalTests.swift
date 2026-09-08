import Foundation
import XCTest

import Jinja
import ServingCore
import Tokenizers
@testable import SpikeServingAdapters

/// Proves `ServingChatTemplateRefusal.translated(_:)` — the single place a chat template's own
/// `raise_exception(...)` refusal (surfaced to Swift as `Jinja.TemplateException`) is converted
/// into a client-facing `OpenAIServingError.invalidRequestWithCode` instead of falling through to
/// the serving layer's generic 500.
final class ServingChatTemplateRefusalTests: XCTestCase {
    /// A faithful minimal excerpt of the served model's own guard: a system message that is not
    /// first in the list raises. This mirrors the real template closely enough to provoke a
    /// GENUINE `Jinja.TemplateException` — `TemplateException`'s memberwise init is internal, so a
    /// real render through a real template is the only way to obtain one.
    private static let nonLeadingSystemGuardTemplate = try! Template(
        """
        {%- for message in messages %}
        {%- if message.role == "system" %}
        {%- if not loop.first %}
        {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
        {%- endif %}
        {%- endfor %}
        """)

    private func messagesValue(roles: [String]) throws -> Value {
        try Value(any: roles.map { ["role": $0, "content": "x"] })
    }

    // MARK: A — decisive reproduction

    func testTranslatesGenuineNonLeadingSystemTemplateExceptionToInvalidRequest() throws {
        let messages = try messagesValue(roles: ["user", "system", "user"])

        var caught: (any Error)?
        do {
            _ = try Self.nonLeadingSystemGuardTemplate.render(["messages": messages])
            XCTFail("expected the non-leading system message to raise")
        } catch {
            caught = error
        }

        let genuineException = try XCTUnwrap(caught)
        XCTAssertTrue(
            genuineException is TemplateException,
            "test fixture must provoke a genuine Jinja.TemplateException, got \(type(of: genuineException))")

        let translated = ServingChatTemplateRefusal.translated(genuineException)
        let servingError = try XCTUnwrap(translated as? OpenAIServingError)

        // Exact equality, not `contains` — a `contains` check would still pass if extraction
        // returned the whole raw `TemplateException(message: Optional("..."))` reflected
        // description (which also contains the template's text as a substring), so it would not
        // actually prove the extraction is clean.
        XCTAssertEqual(
            servingError,
            .invalidRequestWithCode(
                "The model's chat template rejected this request: "
                    + "System message must be at the beginning.",
                param: "messages",
                code: "chat_template_rejected"))
    }

    /// Regression for the substring-parsing bug the `Mirror`-based extractor replaced: a template
    /// message that itself contains a double quote used to get truncated to whatever fell between
    /// the FIRST and LAST quote in the printed description, silently dropping the rest of the
    /// message. Proves the full message, quote included, survives extraction intact.
    func testTemplateExceptionMessageContainingQuoteIsExtractedInFull() throws {
        let template = try Template(
            """
            {{- raise_exception('Use "tools" instead.') }}
            """)

        var caught: (any Error)?
        do {
            _ = try template.render([:])
            XCTFail("expected raise_exception(...) to raise")
        } catch {
            caught = error
        }

        let genuineException = try XCTUnwrap(caught)
        XCTAssertTrue(genuineException is TemplateException)

        let translated = ServingChatTemplateRefusal.translated(genuineException)
        let servingError = try XCTUnwrap(translated as? OpenAIServingError)

        XCTAssertEqual(
            servingError,
            .invalidRequestWithCode(
                "The model's chat template rejected this request: Use \"tools\" instead.",
                param: "messages",
                code: "chat_template_rejected"))
    }

    /// Proves the excerpt is discriminating — it does not raise unconditionally — so test A above
    /// is exercising the actual guard condition, not an always-throwing fixture.
    func testLeadingSystemMessageDoesNotRaise() throws {
        let messages = try messagesValue(roles: ["system", "user"])

        XCTAssertNoThrow(try Self.nonLeadingSystemGuardTemplate.render(["messages": messages]))
    }

    // MARK: B — anti-misclassification

    private struct DistinctiveMarkerError: Error, Equatable {
        let marker: String
    }

    func testCancellationErrorPassesThroughUnchanged() {
        let original: any Error = CancellationError()
        let translated = ServingChatTemplateRefusal.translated(original)
        XCTAssertTrue(translated is CancellationError)
    }

    func testDistinctiveCustomErrorPassesThroughUnchanged() {
        let original = DistinctiveMarkerError(marker: "not-a-template-refusal")
        let translated = ServingChatTemplateRefusal.translated(original)
        let recovered = try? XCTUnwrap(translated as? DistinctiveMarkerError)
        XCTAssertEqual(recovered, original)
    }

    /// A `TokenizerError` (e.g. a missing chat template entirely) is a SERVER misconfiguration,
    /// not a client request-shape refusal, and must stay a 500 — prove it is not reclassified.
    func testTokenizerErrorPassesThroughUnchanged() {
        let original: any Error = TokenizerError.missingChatTemplate
        let translated = ServingChatTemplateRefusal.translated(original)
        guard case .missingChatTemplate = translated as? TokenizerError else {
            XCTFail("expected TokenizerError.missingChatTemplate to pass through unchanged, got \(translated)")
            return
        }
    }

    /// `Jinja.JinjaError` is thrown from the SAME `template.render` frame as `TemplateException`,
    /// but it represents a swift-jinja capability gap or a corrupt template — a SERVER fault, not
    /// a template author's own refusal — so it must stay a 500. This is the error type most likely
    /// to be wrongly swept in by a future widening of the `translated(_:)` guard: proves it passes
    /// through with both its type AND its case completely unchanged. Provoked genuinely (an
    /// unrecognized filter name), rather than constructed directly, so the test also proves this
    /// build's swift-jinja still raises `.runtime` for that condition.
    func testJinjaErrorPassesThroughUnchanged() throws {
        let template = try Template(
            """
            {{- "x" | totally_unrecognized_filter }}
            """)

        var caught: (any Error)?
        do {
            _ = try template.render([:])
            XCTFail("expected an unrecognized filter to raise")
        } catch {
            caught = error
        }

        let genuineError = try XCTUnwrap(caught)
        guard case .runtime(let message) = genuineError as? JinjaError else {
            XCTFail("expected a genuine Jinja.JinjaError.runtime, got \(type(of: genuineError))")
            return
        }

        let translated = ServingChatTemplateRefusal.translated(genuineError)
        guard case .runtime(let translatedMessage) = translated as? JinjaError else {
            XCTFail("expected JinjaError to pass through unchanged, got \(translated)")
            return
        }
        XCTAssertEqual(translatedMessage, message)
    }

    // MARK: C — message-extraction fallback

    func testMessagelessTemplateExceptionFallsBackToSaneMessage() throws {
        let template = try Template(
            """
            {{- raise_exception() }}
            """)

        var caught: (any Error)?
        do {
            _ = try template.render([:])
            XCTFail("expected raise_exception() with no argument to raise")
        } catch {
            caught = error
        }

        let genuineException = try XCTUnwrap(caught)
        XCTAssertTrue(genuineException is TemplateException)

        let translated = ServingChatTemplateRefusal.translated(genuineException)
        let servingError = try XCTUnwrap(translated as? OpenAIServingError)

        XCTAssertEqual(servingError.openAIError.type, .invalidRequest)
        XCTAssertEqual(servingError.openAIError.code, "chat_template_rejected")
        XCTAssertFalse(servingError.openAIError.message.isEmpty)
    }
}
