import Foundation
import os
import XCTest

import HarnessCore
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

final class ContinuousServingBackendTests: XCTestCase {
    func testDynamicAdmissionSingleHeldExpiryStartsSoloPLDWithSpeculation()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99])

        let start = Task {
            try await backend.start(request(text: "solo", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }

        await backend.diagnosticExpireAdmissionCoalescingWindow()
        let handle = try await start.value

        XCTAssertEqual(handle.route, .soloPLD)
        try await runUntilDecode(coordinator, recorder: recorder)
        XCTAssertEqual(
            recorder.decodeActions.first,
            .solo(BatchRequestID(1), speculationAllowed: true))

        _ = await handle.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    /// Sampling on the continuous route (cycle 276, slice 2): the policy maps to the submission's
    /// `ContinuousBatchRowSampling` exactly as the scalar route's `MLXDecoder.setSampling` builds its
    /// `TopPSampler` (Float values, absent top-k/min-p as 0, the Int64 seed by bit pattern).
    func testRowSamplingMappingMatchesTheScalarRouteTopPSamplerArguments() {
        XCTAssertNil(ContinuousServingBackend.rowSampling(for: .greedy))

        XCTAssertEqual(
            ContinuousServingBackend.rowSampling(
                for: .sampled(temperature: 0.8, topP: 1, topK: nil, minP: nil, seed: nil)),
            ContinuousBatchRowSampling(
                temperature: Float(0.8), topP: 1, topK: 0, minP: 0, seed: nil))

        XCTAssertEqual(
            ContinuousServingBackend.rowSampling(
                for: .sampled(temperature: 0.7, topP: 0.9, topK: 40, minP: 0.05, seed: 7)),
            ContinuousBatchRowSampling(
                temperature: Float(0.7), topP: Float(0.9), topK: 40, minP: Float(0.05), seed: 7))

        let negativeSeed = ContinuousServingBackend.rowSampling(
            for: .sampled(temperature: 1, topP: 1, topK: nil, minP: nil, seed: -1))
        XCTAssertEqual(negativeSeed?.seed, 18_446_744_073_709_551_615)
    }

    /// A sampled request (temperature > 0) is ADMITTED on the continuous route and the coordinator
    /// receives the mapped sampling; a greedy request carries none. Replaces the former
    /// `testContinuousBatchRouteRejectsSampledRequestInsteadOfSilentGreedy`.
    func testContinuousBatchRouteAdmitsSampledRequestAndCarriesSamplingIntoTheSubmission()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99])

        var sampled = request(text: "solo", maxTokens: 2)
        sampled.temperature = 0.8
        sampled.seed = 7
        let handle = try await backend.start(sampled)
        XCTAssertEqual(handle.route, .continuousBatchNoSpec)
        XCTAssertEqual(recorder.submissions.count, 1)
        XCTAssertEqual(
            recorder.submissions.first?.sampling,
            ContinuousBatchRowSampling(
                temperature: Float(0.8), topP: 1, topK: 0, minP: 0, seed: 7))
        _ = await handle.lease.cancel(.clientDisconnected)

        let greedy = request(text: "solo", maxTokens: 2)
        let greedyHandle = try await backend.start(greedy)
        XCTAssertEqual(recorder.submissions.count, 2)
        XCTAssertNil(recorder.submissions.last?.sampling)
        _ = await greedyHandle.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    /// Invalid sampling values are still 400s on the continuous route, before any admission. (The
    /// penalty 400 that used to sit beside them is gone: penalties are served, see the penalty tests
    /// below.)
    func testContinuousBatchRouteStillRefusesInvalidSamplingValues()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99])

        // A penalized request with an invalid sampling value is still refused on the sampling param.
        var invalid = request(text: "solo", maxTokens: 2)
        invalid.temperature = 3.0
        invalid.presencePenalty = 1.5
        do {
            _ = try await backend.start(invalid)
            XCTFail("temperature 3.0 is out of range and must be refused")
        } catch let error as OpenAIServingError {
            guard case .invalidRequest(_, let param) = error else {
                XCTFail("expected invalidRequest, got \(error)")
                await backend.shutdown()
                return
            }
            XCTAssertEqual(param, "temperature")
        }
        XCTAssertEqual(recorder.submissions, [], "a refused request must never reach the coordinator")
        await backend.shutdown()
    }

    /// The request's penalty fields map through the scalar route's `DecoderPenalties` bridge: the
    /// neutral set (`DecoderPenalties.isEmpty`) is `nil` (the unpenalized graph is untouched);
    /// otherwise each field is a Float, with a neutral repetition value or a zero presence/frequency
    /// as `nil`.
    func testRowPenaltiesMappingMatchesTheDecoderPenaltiesBridge() {
        func mapped(
            presence: Double? = nil, frequency: Double? = nil, repetition: Double? = nil
        ) -> ContinuousBatchRowPenalties? {
            ContinuousServingBackend.rowPenalties(
                for: DecoderPenalties(
                    presencePenalty: presence, frequencyPenalty: frequency,
                    repetitionPenalty: repetition))
        }
        XCTAssertNil(mapped())
        XCTAssertNil(mapped(presence: 0, frequency: 0, repetition: 1))
        XCTAssertNil(mapped(repetition: 0))
        XCTAssertEqual(
            mapped(presence: 1.5, frequency: 0.5, repetition: 1.1),
            ContinuousBatchRowPenalties(
                presence: Float(1.5), frequency: Float(0.5), repetition: Float(1.1)))
        XCTAssertEqual(
            mapped(presence: 1.5, repetition: 1.0),
            ContinuousBatchRowPenalties(presence: Float(1.5), frequency: nil, repetition: nil))
        XCTAssertEqual(
            mapped(presence: 0, frequency: -0.5),
            ContinuousBatchRowPenalties(presence: nil, frequency: Float(-0.5), repetition: nil))
        XCTAssertEqual(
            mapped(repetition: 1.1),
            ContinuousBatchRowPenalties(presence: nil, frequency: nil, repetition: Float(1.1)))
    }

    /// P3: a neutral penalty set (`repetition_penalty: 1.0`, `presence_penalty: 0`,
    /// `frequency_penalty: 0`) reaches the coordinator as NO penalties, exactly like a request that
    /// never sent the fields.
    func testContinuousBatchRouteSubmitsNoPenaltiesForTheNeutralSet() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99])

        var neutral = request(text: "solo", maxTokens: 2)
        neutral.repetitionPenalty = 1.0
        neutral.presencePenalty = 0
        neutral.frequencyPenalty = 0
        let neutralHandle = try await backend.start(neutral)
        XCTAssertEqual(recorder.submissions.count, 1)
        XCTAssertNil(recorder.submissions.last?.penalties)
        _ = await neutralHandle.lease.cancel(.clientDisconnected)

        let absent = try await backend.start(request(text: "solo", maxTokens: 2))
        XCTAssertEqual(recorder.submissions.count, 2)
        XCTAssertNil(recorder.submissions.last?.penalties)
        _ = await absent.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    /// A real penalty set is ADMITTED (greedy or sampled) and the coordinator receives exactly the
    /// request's Float values; a neutral repetition beside a real presence penalty stays `nil`.
    func testContinuousBatchRouteAdmitsPenaltiesAndCarriesThemIntoTheSubmission() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99])

        for temperature in [0.0, 0.8] {
            var full = request(text: "solo", maxTokens: 2)
            full.temperature = temperature
            full.presencePenalty = 1.5
            full.repetitionPenalty = 1.1
            full.frequencyPenalty = 0.5
            let handle = try await backend.start(full)
            XCTAssertEqual(handle.route, .continuousBatchNoSpec)
            XCTAssertEqual(
                recorder.submissions.last?.penalties,
                ContinuousBatchRowPenalties(
                    presence: Float(1.5), frequency: Float(0.5), repetition: Float(1.1)),
                "T=\(temperature)")
            _ = await handle.lease.cancel(.clientDisconnected)
        }

        var presenceOnly = request(text: "solo", maxTokens: 2)
        presenceOnly.presencePenalty = 1.5
        presenceOnly.repetitionPenalty = 1.0
        let presenceHandle = try await backend.start(presenceOnly)
        XCTAssertEqual(
            recorder.submissions.last?.penalties,
            ContinuousBatchRowPenalties(presence: Float(1.5), frequency: nil, repetition: nil))
        _ = await presenceHandle.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    /// The HF Instruct preset (T 0.7, top_p 0.8, top_k 20, presence 1.5, repetition 1.0) is admitted
    /// with BOTH its sampling and its penalties carried; the neutral repetition 1.0 adds nothing.
    func testContinuousBatchRouteAdmitsTheHFInstructPreset() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99])

        var preset = request(text: "solo", maxTokens: 2)
        preset.temperature = 0.7
        preset.topP = 0.8
        preset.topK = 20
        preset.presencePenalty = 1.5
        preset.repetitionPenalty = 1.0
        let handle = try await backend.start(preset)
        XCTAssertEqual(handle.route, .continuousBatchNoSpec)
        XCTAssertEqual(recorder.submissions.count, 1)
        XCTAssertEqual(
            recorder.submissions.first?.sampling,
            ContinuousBatchRowSampling(
                temperature: Float(0.7), topP: Float(0.8), topK: 20, minP: 0, seed: nil))
        XCTAssertEqual(
            recorder.submissions.first?.penalties,
            ContinuousBatchRowPenalties(presence: Float(1.5), frequency: nil, repetition: nil))
        _ = await handle.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    /// Invalid penalty values keep their existing 400s, raised at request parse (ServingCore) with the
    /// offending field as `param`; none of them can reach this route.
    func testInvalidPenaltyValuesAreStillRefusedWithTheirOwnParam() throws {
        let cases: [(String, String)] = [
            (#""presence_penalty":3"#, "presence_penalty"),
            (#""presence_penalty":-3"#, "presence_penalty"),
            (#""frequency_penalty":2.1"#, "frequency_penalty"),
            (#""repetition_penalty":-1"#, "repetition_penalty"),
            (#""repetition_penalty":0"#, "repetition_penalty"),
        ]
        for (field, expectedParam) in cases {
            let body =
                #"{"model":"fixture","messages":[{"role":"user","content":"hi"}],"# + field + "}"
            do {
                _ = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
                XCTFail("\(field) must be refused")
            } catch let error as OpenAIServingError {
                guard case .invalidRequest(_, let param) = error else {
                    XCTFail("expected invalidRequest for \(field), got \(error)")
                    continue
                }
                XCTAssertEqual(param, expectedParam, field)
            }
        }
    }

    /// Each invalid sampling field is a 400 naming ITS OWN `param` (the scalar route's mapping, shared
    /// byte for byte), not a blanket `temperature`.
    func testContinuousBatchRouteNamesTheOffendingSamplingParameter() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99])

        let cases: [(String, (inout OpenAIChatCompletionRequest) -> Void, OpenAIServingError)] = [
            (
                "temperature 3.0", { $0.temperature = 3.0 },
                .invalidRequest("temperature must be finite and in (0, 2]", param: "temperature")
            ),
            (
                "top_p 0", { $0.temperature = 0.8; $0.topP = 0 },
                .invalidRequest("top_p must be finite and in (0, 1]", param: "top_p")
            ),
            (
                "top_k 0", { $0.temperature = 0.8; $0.topK = 0 },
                .invalidRequest("top_k must be greater than zero", param: "top_k")
            ),
            (
                "min_p 2", { $0.temperature = 0.8; $0.minP = 2 },
                .invalidRequest("min_p must be finite and in [0, 1]", param: "min_p")
            ),
        ]
        for (label, configure, expected) in cases {
            var invalid = request(text: "solo", maxTokens: 2)
            configure(&invalid)
            do {
                _ = try await backend.start(invalid)
                XCTFail("\(label) must be refused")
            } catch let error as OpenAIServingError {
                XCTAssertEqual(error, expected, label)
            }
        }
        XCTAssertEqual(recorder.submissions, [], "a refused request must never reach the coordinator")
        await backend.shutdown()
    }

    /// On the solo-PLD route a sampled request is submitted WITHOUT speculation (the runtime refuses
    /// sampling + speculation) while carrying its sampling; the control, an unconstrained greedy
    /// request on the same route, still requests speculation.
    func testDynamicAdmissionSoloPLDSampledRequestIsSubmittedWithoutSpeculation()
        async throws
    {
        func runSolo(temperature: Double) async throws -> (
            ServingExecutionRoute, [ContinuousBatchSubmission], [BatchDecodeAction]
        ) {
            let recorder = ContinuousRuntimeRecorder()
            let coordinator = ContinuousBatchCoordinator(
                configuration: try configuration(active: 2, queued: 4),
                runtime: FixtureContinuousRuntime(
                    scriptsByPromptHead: [10: [1, 99]],
                    recorder: recorder,
                    allowsSpeculation: true),
                automaticDrive: false,
                publicationCapacity: 1,
                traceLimit: 32)
            let backend = makeDynamicBackend(
                coordinator: coordinator,
                promptByText: ["solo": [10]],
                pieces: [1: "s"],
                stopTokenIDs: [99])
            var solo = request(text: "solo", maxTokens: 2)
            solo.temperature = temperature
            solo.seed = 7
            let start = Task { try await backend.start(solo) }
            await waitUntil {
                await backend.diagnosticPendingAdmissionRequestCount() == 1
            }
            await backend.diagnosticExpireAdmissionCoalescingWindow()
            let handle = try await start.value
            try await runUntilDecode(coordinator, recorder: recorder)
            let result = (handle.route, recorder.submissions, recorder.decodeActions)
            _ = await handle.lease.cancel(.clientDisconnected)
            await backend.shutdown()
            return result
        }

        let sampled = try await runSolo(temperature: 0.8)
        XCTAssertEqual(sampled.0, .soloPLD)
        XCTAssertEqual(sampled.1.count, 1)
        XCTAssertEqual(
            sampled.1.first?.sampling,
            ContinuousBatchRowSampling(
                temperature: Float(0.8), topP: 1, topK: 0, minP: 0, seed: 7))
        XCTAssertEqual(sampled.1.first?.requestsSpeculation, false)
        XCTAssertEqual(
            sampled.2.first, .solo(BatchRequestID(1), speculationAllowed: false))

        let greedy = try await runSolo(temperature: 0)
        XCTAssertEqual(greedy.0, .soloPLD)
        XCTAssertNil(greedy.1.first?.sampling)
        XCTAssertEqual(greedy.1.first?.requestsSpeculation, true)
        XCTAssertEqual(
            greedy.2.first, .solo(BatchRequestID(1), speculationAllowed: true))
    }

    /// On the solo-PLD route a PENALIZED greedy request is submitted WITHOUT speculation (the runtime
    /// refuses penalties + speculation) while carrying its penalties; the control, an unpenalized
    /// greedy request on the same route, still requests speculation.
    func testDynamicAdmissionSoloPLDPenalizedRequestIsSubmittedWithoutSpeculation()
        async throws
    {
        func runSolo(presencePenalty: Double?) async throws -> (
            ServingExecutionRoute, [ContinuousBatchSubmission], [BatchDecodeAction]
        ) {
            let recorder = ContinuousRuntimeRecorder()
            let coordinator = ContinuousBatchCoordinator(
                configuration: try configuration(active: 2, queued: 4),
                runtime: FixtureContinuousRuntime(
                    scriptsByPromptHead: [10: [1, 99]],
                    recorder: recorder,
                    allowsSpeculation: true),
                automaticDrive: false,
                publicationCapacity: 1,
                traceLimit: 32)
            let backend = makeDynamicBackend(
                coordinator: coordinator,
                promptByText: ["solo": [10]],
                pieces: [1: "s"],
                stopTokenIDs: [99])
            var solo = request(text: "solo", maxTokens: 2)
            solo.presencePenalty = presencePenalty
            let start = Task { try await backend.start(solo) }
            await waitUntil {
                await backend.diagnosticPendingAdmissionRequestCount() == 1
            }
            await backend.diagnosticExpireAdmissionCoalescingWindow()
            let handle = try await start.value
            try await runUntilDecode(coordinator, recorder: recorder)
            let result = (handle.route, recorder.submissions, recorder.decodeActions)
            _ = await handle.lease.cancel(.clientDisconnected)
            await backend.shutdown()
            return result
        }

        let penalized = try await runSolo(presencePenalty: 1.5)
        XCTAssertEqual(penalized.0, .soloPLD)
        XCTAssertEqual(penalized.1.count, 1)
        XCTAssertEqual(
            penalized.1.first?.penalties,
            ContinuousBatchRowPenalties(presence: Float(1.5), frequency: nil, repetition: nil))
        XCTAssertEqual(penalized.1.first?.requestsSpeculation, false)
        XCTAssertEqual(
            penalized.2.first, .solo(BatchRequestID(1), speculationAllowed: false))

        let plain = try await runSolo(presencePenalty: nil)
        XCTAssertEqual(plain.0, .soloPLD)
        XCTAssertNil(plain.1.first?.penalties)
        XCTAssertEqual(plain.1.first?.requestsSpeculation, true)
        XCTAssertEqual(
            plain.2.first, .solo(BatchRequestID(1), speculationAllowed: true))
    }

    /// `logit_bias` (A7): the capability is declared exactly when the backend knows the model's
    /// vocabulary width (so every bias id can be bounded before admission). An unknown width fails
    /// CLOSED: the capability is false and a non-empty map that reaches `start` anyway is refused with
    /// the generic 400 (`param: "logit_bias"`) before any admission.
    func testContinuousBatchRouteDeclaresLogitBiasCapabilityOnlyWithKnownVocabulary()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let known = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99],
            logitBiasVocabularySize: 128)
        XCTAssertTrue(known.supportsLogitBias, "a known vocabulary width enables logit_bias")

        let unknown = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99])
        XCTAssertFalse(
            unknown.supportsLogitBias, "an unknown vocabulary width must keep logit_bias off")

        var biased = request(text: "solo", maxTokens: 2)
        biased.logitBias = [1: -100]
        do {
            _ = try await unknown.start(biased)
            XCTFail("an unknown vocabulary must refuse a logit_bias request")
        } catch let error as OpenAIServingError {
            guard case .invalidRequest(let message, let param) = error else {
                XCTFail("expected invalidRequest, got \(error)")
                return
            }
            XCTAssertEqual(param, "logit_bias")
            XCTAssertEqual(
                message, "logit_bias is not supported by the loaded model's decoding route")
        }
        XCTAssertEqual(recorder.admissionBatchSizes, [], "a refused request must never be admitted")
        XCTAssertEqual(recorder.submissions, [], "a refused request must never reach the coordinator")
        XCTAssertEqual(recorder.decodeActions, [])
        await known.shutdown()
        await unknown.shutdown()
    }

    /// `logit_bias` (A8): an id at or above the vocabulary is a 400 (`param: "logit_bias"`, the scalar
    /// backend's text) BEFORE any prompt render, admission or coordinator work; the boundary id
    /// `vocabulary - 1` is admissible. Negative ids are the request decoder's job and not tested here.
    func testContinuousBatchRouteRefusesOutOfVocabularyLogitBiasWithoutAdmittingAnything()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99],
            logitBiasVocabularySize: 128)

        for outOfVocabulary in [128, 129, 1_000_000] {
            var biased = request(text: "solo", maxTokens: 2)
            biased.logitBias = [1: -100, outOfVocabulary: 5]
            do {
                _ = try await backend.start(biased)
                XCTFail("id \(outOfVocabulary) is outside the vocabulary and must be refused")
            } catch let error as OpenAIServingError {
                guard case .invalidRequest(let message, let param) = error else {
                    XCTFail("expected invalidRequest, got \(error)")
                    await backend.shutdown()
                    return
                }
                XCTAssertEqual(param, "logit_bias")
                XCTAssertEqual(
                    message,
                    "logit_bias token ids must be smaller than the model's vocabulary size (128)")
            }
        }
        XCTAssertEqual(recorder.admissionBatchSizes, [], "a refused request must never be admitted")
        XCTAssertEqual(recorder.submissions, [], "a refused request must never reach the coordinator")
        XCTAssertEqual(recorder.decodeActions, [], "a refused request must never reach a decode step")

        // Control: the last in-vocabulary id (and the neutral empty map) are admissible.
        var edge = request(text: "solo", maxTokens: 2)
        edge.logitBias = [127: 3]
        let handle = try await backend.start(edge)
        XCTAssertEqual(handle.route, .continuousBatchNoSpec)
        XCTAssertEqual(recorder.submissions.count, 1)
        _ = await handle.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    /// Replaces the former `testContinuousBatchRouteRefusesNonEmptyLogitBiasWithoutAdmittingAnything`
    /// (the route no longer refuses a bias; the runtime applies it per row). A non-empty in-vocabulary
    /// map is admitted and carried verbatim into the coordinator submission; the neutral map is
    /// carried as empty.
    func testContinuousBatchRouteAdmitsInVocabularyLogitBiasAndCarriesItIntoTheSubmission()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99],
            logitBiasVocabularySize: 128)

        var biased = request(text: "solo", maxTokens: 2)
        biased.logitBias = [1: -100, 7: 2.5]
        let handle = try await backend.start(biased)
        XCTAssertEqual(handle.route, .continuousBatchNoSpec)
        XCTAssertEqual(recorder.admissionBatchSizes, [1])
        XCTAssertEqual(recorder.submissions.count, 1)
        XCTAssertEqual(recorder.submissions.first?.logitBias, [1: -100, 7: 2.5])
        XCTAssertEqual(recorder.submissions.first?.requestsSpeculation, false)
        _ = await handle.lease.cancel(.clientDisconnected)

        var neutral = request(text: "solo", maxTokens: 2)
        neutral.logitBias = [:]
        let neutralHandle = try await backend.start(neutral)
        XCTAssertEqual(recorder.submissions.count, 2)
        XCTAssertEqual(recorder.submissions.last?.logitBias, [:])
        _ = await neutralHandle.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    /// `logit_bias` (A10 / B3): on the solo-PLD route a biased request is submitted WITHOUT speculation
    /// (the runtime refuses bias + speculation, and the verify step must never see a biased row) while
    /// carrying its bias map, and the decode action is a non-speculative solo step. The control: an
    /// unbiased request on the same route still requests speculation.
    func testDynamicAdmissionSoloPLDBiasedRequestIsSubmittedWithoutSpeculation()
        async throws
    {
        func runSolo(bias: [Int: Float]) async throws -> (
            ServingExecutionRoute, [ContinuousBatchSubmission], [BatchDecodeAction]
        ) {
            let recorder = ContinuousRuntimeRecorder()
            let coordinator = ContinuousBatchCoordinator(
                configuration: try configuration(active: 2, queued: 4),
                runtime: FixtureContinuousRuntime(
                    scriptsByPromptHead: [10: [1, 99]],
                    recorder: recorder,
                    allowsSpeculation: true),
                automaticDrive: false,
                publicationCapacity: 1,
                traceLimit: 32)
            let backend = makeDynamicBackend(
                coordinator: coordinator,
                promptByText: ["solo": [10]],
                pieces: [1: "s"],
                stopTokenIDs: [99],
                logitBiasVocabularySize: 128)
            var solo = request(text: "solo", maxTokens: 2)
            solo.logitBias = bias
            let start = Task { try await backend.start(solo) }
            await waitUntil {
                await backend.diagnosticPendingAdmissionRequestCount() == 1
            }
            await backend.diagnosticExpireAdmissionCoalescingWindow()
            let handle = try await start.value
            try await runUntilDecode(coordinator, recorder: recorder)
            let result = (handle.route, recorder.submissions, recorder.decodeActions)
            _ = await handle.lease.cancel(.clientDisconnected)
            await backend.shutdown()
            return result
        }

        let biased = try await runSolo(bias: [1: -100])
        XCTAssertEqual(biased.0, .soloPLD)
        XCTAssertEqual(biased.1.count, 1)
        XCTAssertEqual(biased.1.first?.logitBias, [1: -100])
        XCTAssertEqual(biased.1.first?.requestsSpeculation, false)
        XCTAssertEqual(
            biased.2.first, .solo(BatchRequestID(1), speculationAllowed: false))

        let unbiased = try await runSolo(bias: [:])
        XCTAssertEqual(unbiased.0, .soloPLD)
        XCTAssertEqual(unbiased.1.first?.logitBias, [:])
        XCTAssertEqual(unbiased.1.first?.requestsSpeculation, true)
        XCTAssertEqual(
            unbiased.2.first, .solo(BatchRequestID(1), speculationAllowed: true))
    }

    // MARK: - Named / required `tool_choice` and `response_format` (cycle 274, slice 2)

    /// Replaces the former unconditional refusal. WITHOUT constraint support (the default
    /// configuration, or a checkpoint whose tokenizer cannot back one) a named or `"required"`
    /// `tool_choice` is still refused with a 400 on `tool_choice` before any admission or decode --
    /// defense in depth behind the HTTP capability gate. `auto`/`none` are unaffected.
    func testContinuousBatchRouteRefusesNamedAndRequiredToolChoiceWithoutConstraintSupport()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99])
        XCTAssertFalse(backend.supportsJSONSchemaResponseFormat)
        XCTAssertFalse(backend.supportsJSONObjectResponseFormat)
        XCTAssertFalse(backend.enforcesForcingToolChoice)

        for choice: OpenAIToolChoice in [.function("get_weather"), .required] {
            var forced = request(text: "solo", maxTokens: 2, tools: [continuousWeatherTool])
            forced.toolChoice = choice
            do {
                _ = try await backend.start(forced)
                XCTFail("continuous route must refuse \(choice) instead of answering in prose")
            } catch let error as OpenAIServingError {
                guard case .invalidRequest(_, let param) = error else {
                    XCTFail("expected invalidRequest, got \(error)")
                    await backend.shutdown()
                    return
                }
                XCTAssertEqual(param, "tool_choice")
            }
        }
        XCTAssertEqual(recorder.admissionBatchSizes, [], "a refused request must never be admitted")
        XCTAssertEqual(recorder.decodeActions, [], "a refused request must never reach a decode step")

        // `response_format` without its support is refused the same way, with its own param.
        for format in [ServingResponseFormat.jsonObject, .jsonSchema(try booleanSchemaFormat())] {
            var constrained = request(text: "solo", maxTokens: 2)
            constrained.responseFormat = format
            do {
                _ = try await backend.start(constrained)
                XCTFail("continuous route must refuse \(format) without constraint support")
            } catch OpenAIServingError.invalidRequest(_, let param) {
                XCTAssertEqual(param, "response_format")
            }
        }
        XCTAssertEqual(recorder.admissionBatchSizes, [])

        // Control: `auto` with tools is still a normal, admissible request.
        let handle = try await backend.start(
            request(text: "solo", maxTokens: 2, tools: [continuousWeatherTool]))
        XCTAssertEqual(handle.route, .continuousBatchNoSpec)
        XCTAssertNil(recorder.submissions.first?.tokenMask, "an unconstrained request carries no mask")
        _ = await handle.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    /// The capability flags flip with the loaded support exactly as on the scalar route:
    /// json_object <- object support, json_schema <- schema support, forcing tool_choice <- schema.
    func testConstraintCapabilityFlagsFollowTheConfiguredSupport() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]], recorder: recorder),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        func backend(
            object: ScalarServingJSONObjectConstraintSupport?,
            schema: ScalarServingJSONSchemaConstraintSupport?
        ) -> ContinuousServingBackend {
            makeBackend(
                coordinator: coordinator, promptByText: ["solo": [10]], pieces: [1: "s"],
                stopTokenIDs: [99], jsonObjectConstraintSupport: object,
                jsonSchemaConstraintSupport: schema)
        }
        let both = backend(object: continuousObjectSupport(), schema: continuousSchemaSupport())
        XCTAssertTrue(both.supportsJSONObjectResponseFormat)
        XCTAssertTrue(both.supportsJSONSchemaResponseFormat)
        XCTAssertTrue(both.enforcesForcingToolChoice)

        let objectOnly = backend(object: continuousObjectSupport(), schema: nil)
        XCTAssertTrue(objectOnly.supportsJSONObjectResponseFormat)
        XCTAssertFalse(objectOnly.supportsJSONSchemaResponseFormat)
        XCTAssertFalse(objectOnly.enforcesForcingToolChoice, "forcing needs the json_schema mask")

        let schemaOnly = backend(object: nil, schema: continuousSchemaSupport())
        XCTAssertFalse(schemaOnly.supportsJSONObjectResponseFormat)
        XCTAssertTrue(schemaOnly.supportsJSONSchemaResponseFormat)
        XCTAssertTrue(schemaOnly.enforcesForcingToolChoice)
    }

    /// (a) A named `tool_choice` is ADMITTED when the json_schema support is configured; the
    /// submission carries a live mask (the REAL constraint: `{` is the only legal first byte of an
    /// object) and does not request speculation. (e) `"required"` likewise.
    func testContinuousBatchRouteAdmitsNamedAndRequiredToolChoiceWithATokenMask() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99],
            jsonSchemaConstraintSupport: continuousSchemaSupport())
        XCTAssertTrue(backend.enforcesForcingToolChoice)

        let named = try strictRequest(toolChoice: Self.namedChoice("get_weather"))
        let namedHandle = try await backend.start(named)
        XCTAssertEqual(namedHandle.route, .continuousBatchNoSpec)
        XCTAssertEqual(recorder.admissionBatchSizes, [1])
        let namedMask = try XCTUnwrap(recorder.submissions.first?.tokenMask, "named: mask carried")
        XCTAssertEqual(recorder.submissions.first?.requestsSpeculation, false)
        let namedBits = try XCTUnwrap(namedMask.allowedTokenBitset(), "active from the first token")
        XCTAssertTrue(continuousBit(Int(UInt8(ascii: "{")), in: namedBits))
        XCTAssertFalse(continuousBit(Int(UInt8(ascii: "x")), in: namedBits))
        _ = await namedHandle.lease.cancel(.clientDisconnected)

        let required = try strictRequest(toolChoice: #""required""#)
        let requiredHandle = try await backend.start(required)
        XCTAssertEqual(recorder.submissions.count, 2)
        let requiredMask = try XCTUnwrap(recorder.submissions.last?.tokenMask, "required: mask carried")
        XCTAssertEqual(recorder.submissions.last?.requestsSpeculation, false)
        let requiredBits = try XCTUnwrap(requiredMask.allowedTokenBitset())
        XCTAssertTrue(continuousBit(Int(UInt8(ascii: "{")), in: requiredBits))
        XCTAssertFalse(continuousBit(Int(UInt8(ascii: "x")), in: requiredBits))
        XCTAssertFalse(namedMask === requiredMask, "one mask per request")
        _ = await requiredHandle.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    /// (f) `response_format` json_object and json_schema are admitted with a mask.
    func testContinuousBatchRouteAdmitsJSONObjectAndJSONSchemaResponseFormatsWithATokenMask()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]], recorder: recorder),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let object = continuousObjectSupport()
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99],
            jsonObjectConstraintSupport: object,
            jsonSchemaConstraintSupport: loadScalarServingJSONSchemaConstraintSupport(sharing: object))

        var objectRequest = request(text: "solo", maxTokens: 2)
        objectRequest.responseFormat = .jsonObject
        let objectHandle = try await backend.start(objectRequest)
        let objectMask = try XCTUnwrap(recorder.submissions.last?.tokenMask, "json_object: mask carried")
        let objectBits = try XCTUnwrap(objectMask.allowedTokenBitset())
        XCTAssertTrue(continuousBit(Int(UInt8(ascii: "{")), in: objectBits))
        XCTAssertFalse(continuousBit(Int(UInt8(ascii: "x")), in: objectBits))
        _ = await objectHandle.lease.cancel(.clientDisconnected)

        var schemaRequest = request(text: "solo", maxTokens: 2)
        schemaRequest.responseFormat = .jsonSchema(try booleanSchemaFormat())
        let schemaHandle = try await backend.start(schemaRequest)
        XCTAssertEqual(recorder.submissions.count, 2)
        let schemaMask = try XCTUnwrap(recorder.submissions.last?.tokenMask, "json_schema: mask carried")
        let schemaBits = try XCTUnwrap(schemaMask.allowedTokenBitset())
        XCTAssertTrue(continuousBit(Int(UInt8(ascii: "t")), in: schemaBits))
        XCTAssertFalse(continuousBit(Int(UInt8(ascii: "{")), in: schemaBits), "boolean schema: no object")
        _ = await schemaHandle.lease.cancel(.clientDisconnected)

        // json_object support alone does not admit json_schema (and vice versa is covered by flags).
        let objectOnly = makeBackend(
            coordinator: coordinator, promptByText: ["solo": [10]], pieces: [1: "s"],
            stopTokenIDs: [99], jsonObjectConstraintSupport: continuousObjectSupport())
        do {
            _ = try await objectOnly.start(schemaRequest)
            XCTFail("json_schema without schema support must be refused")
        } catch OpenAIServingError.invalidRequest(_, let param) {
            XCTAssertEqual(param, "response_format")
        }
        XCTAssertEqual(recorder.submissions.count, 2, "the refused request was never admitted")
        await backend.shutdown()
    }

    /// (c) A `response_format` together with a named or `"required"` `tool_choice` is refused on
    /// `tool_choice` (the scalar route's rule), before any admission.
    func testResponseFormatTogetherWithForcingToolChoiceIsRefusedBeforeAdmission() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]], recorder: recorder),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let object = continuousObjectSupport()
        let backend = makeBackend(
            coordinator: coordinator, promptByText: ["solo": [10]], pieces: [1: "s"],
            stopTokenIDs: [99], jsonObjectConstraintSupport: object,
            jsonSchemaConstraintSupport: loadScalarServingJSONSchemaConstraintSupport(sharing: object))

        for choice in [Self.namedChoice("get_weather"), #""required""#] {
            for format in [ServingResponseFormat.jsonObject, .jsonSchema(try booleanSchemaFormat())] {
                var conflicting = try strictRequest(toolChoice: choice)
                conflicting.responseFormat = format
                do {
                    _ = try await backend.start(conflicting)
                    XCTFail("expected a refusal for \(choice) + \(format)")
                } catch OpenAIServingError.invalidRequest(_, let param) {
                    XCTAssertEqual(param, "tool_choice")
                }
            }
        }
        XCTAssertEqual(recorder.admissionBatchSizes, [])
        XCTAssertTrue(recorder.submissions.isEmpty)
        await backend.shutdown()
    }

    /// (d) A thinking request has no `activeFromStart` mask: without a resolvable `</think>` token
    /// the constraint could never activate (or would corrupt the reasoning block), so every
    /// constrained form is refused per request, before admission. WITH a think-end id the same request
    /// is admitted and its mask is INACTIVE (nil) until the think-end token is emitted.
    func testThinkingRequestWithoutAThinkEndTokenIsRefusedAndWithOneStartsInactive() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]], recorder: recorder),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let noThinkEnd = continuousObjectSupport(thinkEndTokenID: nil)
        let refusing = makeBackend(
            coordinator: coordinator, promptByText: ["solo": [10]], pieces: [1: "s"],
            stopTokenIDs: [99], thinksByDefault: true, jsonObjectConstraintSupport: noThinkEnd,
            jsonSchemaConstraintSupport: loadScalarServingJSONSchemaConstraintSupport(sharing: noThinkEnd))

        var attempts: [(OpenAIChatCompletionRequest, String)] = []
        attempts.append((try strictRequest(toolChoice: Self.namedChoice("get_weather")), "tool_choice"))
        attempts.append((try strictRequest(toolChoice: #""required""#), "tool_choice"))
        var objectRequest = request(text: "solo", maxTokens: 2, enableThinking: true)
        objectRequest.responseFormat = .jsonObject
        attempts.append((objectRequest, "response_format"))
        var schemaRequest = request(text: "solo", maxTokens: 2, enableThinking: true)
        schemaRequest.responseFormat = .jsonSchema(try booleanSchemaFormat())
        attempts.append((schemaRequest, "response_format"))
        for (attempt, expectedParam) in attempts {
            var thinking = attempt
            thinking.enableThinking = true
            do {
                _ = try await refusing.start(thinking)
                XCTFail("a thinking request with no </think> token must be refused (\(expectedParam))")
            } catch OpenAIServingError.invalidRequest(_, let param) {
                XCTAssertEqual(param, expectedParam)
            }
        }
        XCTAssertEqual(recorder.admissionBatchSizes, [])
        XCTAssertTrue(recorder.submissions.isEmpty)

        // With a resolvable think-end id the same named request is admitted; its mask is inactive
        // until the think-end token, then answers the constraint's first-byte set.
        let withThinkEnd = continuousObjectSupport(thinkEndTokenID: continuousThinkEndID)
        let accepting = makeBackend(
            coordinator: coordinator, promptByText: ["solo": [10]], pieces: [1: "s"],
            stopTokenIDs: [99], thinksByDefault: true,
            jsonSchemaConstraintSupport: loadScalarServingJSONSchemaConstraintSupport(sharing: withThinkEnd))
        var thinkingNamed = try strictRequest(toolChoice: Self.namedChoice("get_weather"))
        thinkingNamed.enableThinking = true
        let handle = try await accepting.start(thinkingNamed)
        XCTAssertTrue(handle.separatesReasoning)
        let mask = try XCTUnwrap(recorder.submissions.first?.tokenMask)
        XCTAssertNil(mask.allowedTokenBitset(), "thinking phase: unconstrained")
        mask.advance(token: continuousThinkEndID)
        let bits = try XCTUnwrap(mask.allowedTokenBitset(), "after </think>: constrained")
        XCTAssertTrue(continuousBit(Int(UInt8(ascii: "{")), in: bits))
        _ = await handle.lease.cancel(.clientDisconnected)
        await accepting.shutdown()
    }

    /// A masked request on the solo-PLD route is submitted WITHOUT speculation (the runtime refuses
    /// mask + speculation) and decodes with a non-speculative solo step; an unmasked request on the
    /// same route still requests speculation.
    func testDynamicAdmissionSoloPLDMaskedRequestIsSubmittedWithoutSpeculation() async throws {
        func runSolo(masked: Bool) async throws -> (
            ServingExecutionRoute, [ContinuousBatchSubmission], [BatchDecodeAction]
        ) {
            let recorder = ContinuousRuntimeRecorder()
            let coordinator = ContinuousBatchCoordinator(
                configuration: try configuration(active: 2, queued: 4),
                runtime: FixtureContinuousRuntime(
                    scriptsByPromptHead: [10: [1, 99]],
                    recorder: recorder,
                    allowsSpeculation: true),
                automaticDrive: false,
                publicationCapacity: 1,
                traceLimit: 32)
            let backend = makeDynamicBackend(
                coordinator: coordinator,
                promptByText: ["solo": [10]],
                pieces: [1: "s"],
                stopTokenIDs: [99],
                jsonSchemaConstraintSupport: continuousSchemaSupport())
            let solo = masked
                ? try strictRequest(toolChoice: Self.namedChoice("get_weather"))
                : request(text: "solo", maxTokens: 2)
            let start = Task { try await backend.start(solo) }
            await waitUntil {
                await backend.diagnosticPendingAdmissionRequestCount() == 1
            }
            await backend.diagnosticExpireAdmissionCoalescingWindow()
            let handle = try await start.value
            try await runUntilDecode(coordinator, recorder: recorder)
            let result = (handle.route, recorder.submissions, recorder.decodeActions)
            _ = await handle.lease.cancel(.clientDisconnected)
            await backend.shutdown()
            return result
        }

        let masked = try await runSolo(masked: true)
        XCTAssertEqual(masked.0, .soloPLD)
        XCTAssertEqual(masked.1.count, 1)
        XCTAssertNotNil(masked.1.first?.tokenMask)
        XCTAssertEqual(masked.1.first?.requestsSpeculation, false)
        XCTAssertEqual(masked.2.first, .solo(BatchRequestID(1), speculationAllowed: false))

        let unmasked = try await runSolo(masked: false)
        XCTAssertEqual(unmasked.0, .soloPLD)
        XCTAssertNil(unmasked.1.first?.tokenMask)
        XCTAssertEqual(unmasked.1.first?.requestsSpeculation, true)
        XCTAssertEqual(unmasked.2.first, .solo(BatchRequestID(1), speculationAllowed: true))
    }

    // MARK: Enforced output shaping

    /// (g) Named `tool_choice`: the generated text IS the call's arguments. The backend never runs
    /// the family tool parser; it streams a `.toolCallStart(name)` with the first text and then
    /// `.toolCallArguments` fragments (byte for byte the generated text), no `.text`, no
    /// `.toolCalls`. The finish reason is `stop`, exactly as on the scalar route (the HTTP layer
    /// assembles the single call). The fixture runtime advances the REAL constraint with every
    /// emitted token, so a script the grammar rejects would fail the request instead.
    func testNamedToolChoiceStreamsCallHeadThenArgumentFragmentsAndNoContent() async throws {
        let json = #"{"city":"Paris"}"#
        let result = try await runEnforced(
            toolChoice: Self.namedChoice("get_weather"), script: continuousByteIDs(json))

        XCTAssertNil(result.failure)
        guard case .toolCallStart(let head)? = result.events.first else {
            return XCTFail("expected the call head first: \(result.events)")
        }
        XCTAssertEqual(head.function.name, "get_weather")
        XCTAssertEqual(head.function.arguments, "")
        XCTAssertFalse(head.id.isEmpty)
        XCTAssertEqual(result.events.filter { if case .toolCallStart = $0 { true } else { false } }.count, 1)
        XCTAssertEqual(result.arguments, json, "argument fragments are the generated text, in order")
        XCTAssertTrue(result.texts.isEmpty, "no content for an enforced call: \(result.texts)")
        XCTAssertTrue(result.toolCallBatches.isEmpty, "the family parser must not run")
        XCTAssertEqual(result.completion?.finishReason, .stop)
        XCTAssertEqual(result.completion?.usage.completionTokens, json.utf8.count)
    }

    /// `"required"`: the union document's frame (`{"name": ..., "arguments": `) is stripped; the head
    /// carries the chosen tool and the fragments concatenate to the arguments value only.
    func testRequiredToolChoiceStripsTheUnionFrameAndReportsTheChosenTool() async throws {
        let document = #"{"name": "get_weather", "arguments": {"city":"Paris"}}"#
        let result = try await runEnforced(
            toolChoice: #""required""#, script: continuousByteIDs(document))

        XCTAssertNil(result.failure)
        guard case .toolCallStart(let head)? = result.events.first(where: {
            if case .toolCallStart = $0 { true } else { false }
        }) else {
            return XCTFail("expected a call head: \(result.events)")
        }
        XCTAssertEqual(head.function.name, "get_weather")
        XCTAssertEqual(result.arguments, #"{"city":"Paris"}"#)
        XCTAssertTrue(result.texts.isEmpty)
        XCTAssertTrue(result.toolCallBatches.isEmpty)
        XCTAssertEqual(result.completion?.finishReason, .stop)
    }

    /// A thinking request: reasoning (through the first `</think>`) is shown as content and does not
    /// advance the constraint; only the bytes after it are the call's arguments.
    func testNamedToolChoiceOnAThinkingRequestSeparatesReasoningFromArguments() async throws {
        let json = #"{"city":"Rome"}"#
        let script = [Int(UInt8(ascii: "h")), Int(UInt8(ascii: "m")), continuousThinkEndID]
            + continuousByteIDs(json)
        let result = try await runEnforced(
            toolChoice: Self.namedChoice("get_weather"), script: script, thinking: true)

        XCTAssertNil(result.failure, "reasoning tokens must not be fed to the constraint")
        XCTAssertEqual(result.texts.joined(), "hm</think>")
        XCTAssertEqual(result.arguments, json)
        guard case .toolCallStart(let head)? = result.events.first(where: {
            if case .toolCallStart = $0 { true } else { false }
        }) else {
            return XCTFail("expected a call head")
        }
        XCTAssertEqual(head.function.name, "get_weather")
        // The head follows the reasoning text in stream order.
        let firstTextIndex = result.events.firstIndex { if case .text = $0 { true } else { false } }
        let headIndex = result.events.firstIndex { if case .toolCallStart = $0 { true } else { false } }
        XCTAssertLessThan(try XCTUnwrap(firstTextIndex), try XCTUnwrap(headIndex))
    }

    /// `response_format` output is plain content (the mask makes it JSON); no tool events.
    func testResponseFormatOutputIsPlainContent() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: continuousByteIDs("true")],
                recorder: recorder, advancesTokenMasks: true),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator, promptByText: ["solo": [10]], pieces: continuousBytePieces(),
            stopTokenIDs: [continuousEOSID], mailboxCapacity: .init(maxDeltas: 64, maxBytes: 65_536),
            jsonSchemaConstraintSupport: continuousSchemaSupport())
        var constrained = request(text: "solo", maxTokens: 16)
        constrained.responseFormat = .jsonSchema(try booleanSchemaFormat())
        let handle = try await backend.start(constrained)
        let events = try await collect(handle.mailbox)

        let texts = events.compactMap { event -> String? in
            if case .text(let value) = event { value } else { nil }
        }
        XCTAssertEqual(texts.joined(), "true")
        XCTAssertFalse(events.contains {
            switch $0 {
            case .toolCallStart, .toolCallArguments, .toolCalls: true
            default: false
            }
        })
        guard case .completion(let completion)? = events.last else {
            return XCTFail("expected a completion last")
        }
        XCTAssertEqual(completion.finishReason, .stop)
        await backend.shutdown()
    }

    /// An enforced request that exhausts its budget mid-arguments finishes `length` with the partial
    /// arguments streamed (scalar parity).
    func testEnforcedToolCallThatHitsTheBudgetFinishesWithLength() async throws {
        let result = try await runEnforced(
            toolChoice: Self.namedChoice("get_weather"),
            script: continuousByteIDs(#"{"city":"Paris"}"#), maxTokens: 5)
        XCTAssertEqual(result.arguments, #"{"cit"#)
        XCTAssertEqual(result.completion?.finishReason, .length)
    }

    // MARK: Failure surfacing

    /// (h) A constraint failure recorded by the mask (here: the script emits a byte the grammar
    /// forbids) fails THAT request with a backend error before the offending token reaches the
    /// client, releases its slot, and leaves an unconstrained request on the same coordinator
    /// untouched.
    func testRecordedMaskFailureFailsOnlyThatRequestBeforePublishingTheToken() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let valid = Int(UInt8(ascii: "{"))
        let invalid = Int(UInt8(ascii: "x"))
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [valid, invalid, valid], 11: [1, 2]],
                recorder: recorder, advancesTokenMasks: true),
            publicationCapacity: 1,
            traceLimit: 32)
        var pieces = continuousBytePieces()
        pieces[1] = "plain-1"
        pieces[2] = "plain-2"
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10], "other": [11]],
            pieces: pieces,
            stopTokenIDs: [continuousEOSID],
            mailboxCapacity: .init(maxDeltas: 64, maxBytes: 65_536),
            jsonSchemaConstraintSupport: continuousSchemaSupport())

        let failing = try await backend.start(
            try strictRequest(toolChoice: Self.namedChoice("get_weather")))
        let plain = try await backend.start(request(text: "other", maxTokens: 8))

        var received: [ServingResponseDelta] = []
        var failure: ServingMailboxError?
        do {
            while let event = try await failing.mailbox.next() { received.append(event) }
            XCTFail("the constrained request must fail, not complete")
        } catch let error as ServingMailboxError {
            failure = error
        }
        guard case .backend? = failure else {
            return XCTFail("expected a backend failure, got \(String(describing: failure))")
        }
        let args = received.compactMap { event -> String? in
            if case .toolCallArguments(let value) = event { value } else { nil }
        }
        XCTAssertEqual(args.joined(), "{", "only the legal first token was published")
        XCTAssertFalse(received.contains { if case .completion = $0 { true } else { false } })

        let plainEvents = try await collect(plain.mailbox)
        XCTAssertEqual(
            plainEvents.compactMap { event -> String? in
                if case .text(let value) = event { value } else { nil }
            },
            ["plain-1", "plain-2"], "an unconstrained neighbour is unaffected")
        await waitUntil { await backend.snapshot().activeRequests == 0 }
        await waitUntil { await backend.snapshot().coordinatorSlots == 0 }
        await backend.shutdown()
    }

    // MARK: Helpers

    private static func namedChoice(_ name: String) -> String {
        #"{"type":"function","function":{"name":"\#(name)"}}"#
    }

    /// A request decoded the way HTTP does (the forced tool's parameters need the client's key
    /// order, which only the strict decode preserves).
    private func strictRequest(toolChoice: String) throws -> OpenAIChatCompletionRequest {
        let weather =
            #"{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}}"#
        let payload =
            #"{"model":"fixture","messages":[{"role":"user","content":"solo"}],"tools":[\#(weather)],"tool_choice":\#(toolChoice),"max_completion_tokens":64,"temperature":0,"stream":true}"#
        return try OpenAIChatCompletionRequest.decodeStrict(from: Data(payload.utf8))
    }

    private struct EnforcedRun {
        var events: [ServingResponseDelta] = []
        var failure: ServingMailboxError?

        var texts: [String] {
            events.compactMap { if case .text(let value) = $0 { value } else { nil } }
        }
        var arguments: String {
            events.compactMap { if case .toolCallArguments(let value) = $0 { value } else { nil } }.joined()
        }
        var toolCallBatches: [[OpenAIToolCall]] {
            events.compactMap { if case .toolCalls(let value) = $0 { value } else { nil } }
        }
        var completion: ServingGenerationCompletion? {
            for case .completion(let value) in events { return value }
            return nil
        }
    }

    private func runEnforced(
        toolChoice: String,
        script: [Int],
        thinking: Bool = false,
        maxTokens: Int = 64
    ) async throws -> EnforcedRun {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: script],
                recorder: ContinuousRuntimeRecorder(), advancesTokenMasks: true),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator, promptByText: ["solo": [10]], pieces: continuousBytePieces(),
            stopTokenIDs: [continuousEOSID], mailboxCapacity: .init(maxDeltas: 64, maxBytes: 65_536),
            thinksByDefault: thinking,
            jsonSchemaConstraintSupport: continuousSchemaSupport(
                thinkEndTokenID: thinking ? continuousThinkEndID : nil))
        var forced = try strictRequest(toolChoice: toolChoice)
        forced.maxCompletionTokens = maxTokens
        if thinking { forced.enableThinking = true }
        let handle = try await backend.start(forced)
        var run = EnforcedRun()
        do {
            while let event = try await handle.mailbox.next() { run.events.append(event) }
        } catch let error as ServingMailboxError {
            run.failure = error
        }
        await backend.shutdown()
        return run
    }

    /// The continuous-batch route only knows how to render a CHAT-templated prompt — a legacy
    /// `/v1/completions` request (`.rawText`) must be rejected with `completions_unsupported`
    /// BEFORE anything is rendered or decoded, never silently mistreated as an already-templated
    /// chat prompt.
    func testContinuousBatchRouteRejectsRawTextCompletionsRequestBeforeRenderingOrDecoding()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99])

        let completionRequest = try OpenAICompletionRequest.decodeStrict(
            from: Data(#"{"model":"fixture","prompt":"solo","max_tokens":2}"#.utf8))
        let rawTextRequest = completionRequest.asChatCompletionRequest()
        XCTAssertEqual(rawTextRequest.promptInput, .rawText("solo"))

        do {
            _ = try await backend.start(rawTextRequest)
            XCTFail("continuous route must reject a raw-text completions request")
        } catch let error as OpenAIServingError {
            guard case .invalidRequestWithCode(_, let param, let code) = error else {
                XCTFail("expected invalidRequestWithCode, got \(error)")
                return
            }
            XCTAssertEqual(param, "prompt")
            XCTAssertEqual(code, "completions_unsupported")
        }
        XCTAssertTrue(recorder.decodeActions.isEmpty)
        XCTAssertTrue(recorder.admissionBatchSizes.isEmpty)
        await backend.shutdown()
    }

    /// The continuous-batch route has no seam to observe a step's raw logits — a request carrying
    /// a validated `logprobsRequest` must be rejected with `logprobs_unsupported` BEFORE anything
    /// is rendered or decoded, mirroring `completions_unsupported`'s mechanics exactly (never a
    /// silently-wrong response omitting the logprobs the caller asked for).
    func testContinuousBatchRouteRejectsLogprobsRequestBeforeRenderingOrDecoding()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99])

        var logprobsRequest = request(text: "solo", maxTokens: 2)
        logprobsRequest.logprobsRequest = .chat(topLogprobs: 0)

        do {
            _ = try await backend.start(logprobsRequest)
            XCTFail("continuous route must reject a logprobs request")
        } catch let error as OpenAIServingError {
            guard case .invalidRequestWithCode(_, let param, let code) = error else {
                XCTFail("expected invalidRequestWithCode, got \(error)")
                return
            }
            XCTAssertEqual(param, "logprobs")
            XCTAssertEqual(code, "logprobs_unsupported")
        }
        XCTAssertTrue(recorder.decodeActions.isEmpty)
        XCTAssertTrue(recorder.admissionBatchSizes.isEmpty)
        await backend.shutdown()
    }

    /// `repetitionPenalty` is a MULTIPLICATIVE penalty (the vendored `RepetitionContext`:
    /// `x < 0 ? x * penalty : x / penalty`), so its neutral element is `1`, not `0`. Both
    /// HF-recommended presets for the deployed model send `repetition_penalty: 1.0` on every
    /// request -- that must reach the coordinator as NO penalties (the unpenalized graph), while
    /// `1.1` is a REAL penalty and must be carried, so this test discriminates rather than merely
    /// proving "nothing here ever rejects."
    func testContinuousBatchRouteTreatsNeutralRepetitionAsNoPenaltyAndCarriesRealOne() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99])

        var neutral = request(text: "solo", maxTokens: 2)
        neutral.repetitionPenalty = 1.0
        let handle = try await backend.start(neutral)
        XCTAssertEqual(handle.route, .continuousBatchNoSpec)
        XCTAssertNil(recorder.submissions.last?.penalties)
        _ = await handle.lease.cancel(.clientDisconnected)

        var real = request(text: "solo", maxTokens: 2)
        real.repetitionPenalty = 1.1
        let realHandle = try await backend.start(real)
        XCTAssertEqual(realHandle.route, .continuousBatchNoSpec)
        XCTAssertEqual(
            recorder.submissions.last?.penalties,
            ContinuousBatchRowPenalties(presence: nil, frequency: nil, repetition: Float(1.1)))
        _ = await realHandle.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testDynamicAdmissionTwoHeldRequestsExpireAsAtomicBatchNoSpec()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 99],
                    20: [2, 99],
                ],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: [
                "one": [10],
                "two": [20],
            ],
            pieces: [1: "a", 2: "b"],
            stopTokenIDs: [99])

        let oneStart = Task {
            try await backend.start(request(text: "one", maxTokens: 2))
        }
        let twoStart = Task {
            try await backend.start(request(text: "two", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 2
        }

        await backend.diagnosticExpireAdmissionCoalescingWindow()
        let one = try await oneStart.value
        let two = try await twoStart.value

        XCTAssertEqual(one.route, .continuousBatchNoSpec)
        XCTAssertEqual(two.route, .continuousBatchNoSpec)
        let batchIDs = await backend.diagnosticCoordinatorRequestIDs()
        XCTAssertEqual(
            batchIDs,
            [BatchRequestID(1), BatchRequestID(2)])
        try await runUntilDecode(coordinator, recorder: recorder)
        XCTAssertEqual(
            recorder.decodeActions.first,
            .batch(
                [BatchRequestID(1), BatchRequestID(2)],
                speculationAllowed: false))

        _ = await one.lease.cancel(.clientDisconnected)
        _ = await two.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testDynamicAdmissionThirdArrivalWaitsForInFlightCoordinatorAdmissionWithoutResubmittingCohort()
        async throws
    {
        let admissionGate = BlockingRuntimeResourceSnapshotGate()
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 3, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 99],
                    20: [2, 99],
                    30: [3, 99],
                ],
                recorder: recorder,
                allowsSpeculation: true,
                resourceSnapshotGate: admissionGate),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: [
                "one": [10],
                "two": [20],
                "three": [30],
            ],
            pieces: [1: "a", 2: "b", 3: "c"],
            stopTokenIDs: [99],
            maximumBatchRequests: 3)

        let oneStart = Task {
            try await backend.start(request(text: "one", maxTokens: 2))
        }
        let twoStart = Task {
            try await backend.start(request(text: "two", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 2
        }

        let coordinatorBlocker = Task {
            await coordinator.runtimeResourceSnapshot()
        }
        await waitUntil { admissionGate.hasEntered }
        let expiry = Task {
            await backend.diagnosticExpireAdmissionCoalescingWindow()
        }
        await waitUntil {
            await backend.diagnosticExecutingAdmissionRequestCount() == 2
        }

        let threeStart = Task {
            try await backend.start(request(text: "three", maxTokens: 2))
        }
        for _ in 0 ..< 100 {
            await Task.yield()
        }
        let executingWhileCoordinatorBlocked =
            await backend.diagnosticExecutingAdmissionRequestCount()
        XCTAssertEqual(
            executingWhileCoordinatorBlocked,
            2,
            "the third request must wait at the coordinator preflight instead of reentering the in-flight cohort")

        admissionGate.release()
        _ = await coordinatorBlocker.value
        await expiry.value
        let one = try await oneStart.value
        let two = try await twoStart.value
        let three = try await threeStart.value

        XCTAssertEqual(recorder.admissionBatchSizes, [2, 1])
        let coordinatorIDs =
            await backend.diagnosticCoordinatorRequestIDs()
        XCTAssertEqual(
            coordinatorIDs,
            [BatchRequestID(1), BatchRequestID(2), BatchRequestID(3)])

        _ = await one.lease.cancel(.clientDisconnected)
        _ = await two.lease.cancel(.clientDisconnected)
        _ = await three.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testDynamicAdmissionCancellationWhileCoordinatorAdmissionIsInFlight()
        async throws
    {
        let admissionGate = BlockingRuntimeResourceSnapshotGate()
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true,
                resourceSnapshotGate: admissionGate),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["in-flight": [10]],
            pieces: [1: "a"],
            stopTokenIDs: [99],
            maximumBatchRequests: 1)

        let start = Task {
            try await backend.start(
                request(text: "in-flight", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }
        let coordinatorBlocker = Task {
            await coordinator.runtimeResourceSnapshot()
        }
        await waitUntil { admissionGate.hasEntered }
        let expiry = Task {
            await backend.diagnosticExpireAdmissionCoalescingWindow()
        }
        await waitUntil {
            await backend.diagnosticExecutingAdmissionRequestCount() == 1
        }

        start.cancel()
        do {
            _ = try await start.value
            XCTFail("Expected in-flight start cancellation")
        } catch is CancellationError {
        }
        let executingBeforePhysicalRemoval =
            await backend.diagnosticExecutingAdmissionRequestCount()
        XCTAssertEqual(
            executingBeforePhysicalRemoval,
            1,
            "the reducer reservation must remain held until the coordinator can physically remove the cancelled admission")

        admissionGate.release()
        _ = await coordinatorBlocker.value
        await expiry.value
        await waitUntil {
            await backend.diagnosticExecutingAdmissionRequestCount() == 0
        }
        await waitUntil {
            await coordinator.snapshots().isEmpty
        }
        XCTAssertEqual(recorder.admissionBatchSizes, [1])
        let cancellationSlots = await coordinator.snapshots()
        XCTAssertTrue(cancellationSlots.isEmpty)
        await backend.shutdown()
    }

    func testDynamicAdmissionCancellationInFlightKeepsReplacementQueuedUntilPhysicalRemoval()
        async throws
    {
        let admissionGate = BlockingRuntimeResourceSnapshotGate()
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 99],
                    20: [2, 99],
                ],
                recorder: recorder,
                allowsSpeculation: true,
                resourceSnapshotGate: admissionGate),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: [
                "cancelled": [10],
                "replacement": [20],
            ],
            pieces: [1: "a", 2: "b"],
            stopTokenIDs: [99],
            maximumBatchRequests: 1,
            maximumQueuedRequests: 1)

        let cancelledStart = Task {
            try await backend.start(
                request(text: "cancelled", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }
        let replacementStart = Task {
            try await backend.start(
                request(text: "replacement", maxTokens: 2))
        }
        await waitUntil {
            let held =
                await backend.diagnosticPendingAdmissionRequestCount()
            let queued =
                await backend.diagnosticQueuedAdmissionRequestCount()
            return held == 1 && queued == 1
        }

        let coordinatorBlocker = Task {
            await coordinator.runtimeResourceSnapshot()
        }
        await waitUntil { admissionGate.hasEntered }
        let firstExpiry = Task {
            await backend.diagnosticExpireAdmissionCoalescingWindow()
        }
        await waitUntil {
            await backend.diagnosticExecutingAdmissionRequestCount() == 1
        }

        cancelledStart.cancel()
        do {
            _ = try await cancelledStart.value
            XCTFail("Expected in-flight start cancellation")
        } catch is CancellationError {
        }

        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 0
        }
        let queuedBeforePhysicalRemoval =
            await backend.diagnosticQueuedAdmissionRequestCount()
        XCTAssertEqual(
            queuedBeforePhysicalRemoval,
            1,
            "replacement must not enter coordinator admission before the cancelled slot is physically removed")

        admissionGate.release()
        _ = await coordinatorBlocker.value
        await firstExpiry.value
        await waitUntil {
            let queued =
                await backend.diagnosticQueuedAdmissionRequestCount()
            let held =
                await backend.diagnosticPendingAdmissionRequestCount()
            return queued == 0 && held == 1
        }
        await backend.diagnosticExpireAdmissionCoalescingWindow()
        let replacement = try await replacementStart.value
        XCTAssertEqual(replacement.route, .soloPLD)
        XCTAssertEqual(recorder.admissionBatchSizes, [1, 1])

        _ = await replacement.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testDynamicAdmissionShutdownCancelsCoordinatorAdmissionInFlight()
        async throws
    {
        let admissionGate = BlockingRuntimeResourceSnapshotGate()
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true,
                resourceSnapshotGate: admissionGate),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["in-flight": [10]],
            pieces: [1: "a"],
            stopTokenIDs: [99],
            maximumBatchRequests: 1)

        let start = Task {
            try await backend.start(
                request(text: "in-flight", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }
        let coordinatorBlocker = Task {
            await coordinator.runtimeResourceSnapshot()
        }
        await waitUntil { admissionGate.hasEntered }
        let expiry = Task {
            await backend.diagnosticExpireAdmissionCoalescingWindow()
        }
        await waitUntil {
            await backend.diagnosticExecutingAdmissionRequestCount() == 1
        }

        let shutdown = Task {
            await backend.shutdown()
        }
        do {
            _ = try await start.value
            XCTFail("Expected shutdown to release the in-flight start")
        } catch let error as ContinuousServingBackendError {
            XCTAssertEqual(error, .shuttingDown)
        }

        admissionGate.release()
        _ = await coordinatorBlocker.value
        await expiry.value
        await shutdown.value
        XCTAssertEqual(recorder.admissionBatchSizes, [1])
        let shutdownSlots = await coordinator.snapshots()
        XCTAssertTrue(shutdownSlots.isEmpty)
    }

    func testDynamicAdmissionShutdownDuringValidationDoesNotEnqueueLatePendingStart()
        async throws
    {
        let validationBlocker = BlockingRuntimeResourceSnapshotGate()
        let renderGate = BlockingRuntimeResourceSnapshotGate()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: ContinuousRuntimeRecorder(),
                resourceSnapshotGate: validationBlocker),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["validating": [10]],
            pieces: [1: "a"],
            stopTokenIDs: [99],
            maximumBatchRequests: 1,
            renderGate: renderGate)

        let coordinatorBlocker = Task {
            await coordinator.runtimeResourceSnapshot()
        }
        await waitUntil { validationBlocker.hasEntered }

        let start = Task {
            try await backend.start(
                request(text: "validating", maxTokens: 2))
        }
        await waitUntil { renderGate.hasEntered }
        renderGate.release()
        let pendingWhileValidationBlocked =
            await backend.diagnosticPendingAdmissionRequestCount()
        XCTAssertEqual(pendingWhileValidationBlocked, 0)

        let shutdown = Task {
            await backend.shutdown()
        }
        await waitUntil {
            await !backend.diagnosticAcceptingRequests()
        }
        do {
            _ = try await backend.start(
                request(text: "validating", maxTokens: 2))
            XCTFail("Expected shutdown admission rejection")
        } catch let error as ContinuousServingBackendError {
            XCTAssertEqual(error, .shuttingDown)
        }

        validationBlocker.release()
        _ = await coordinatorBlocker.value
        await shutdown.value
        for _ in 0 ..< 100 {
            await Task.yield()
        }

        let pendingAfterShutdown =
            await backend.diagnosticPendingAdmissionRequestCount()
        if pendingAfterShutdown != 0 {
            start.cancel()
        }
        do {
            _ = try await start.value
            XCTFail("Expected validation-time shutdown rejection")
        } catch let error as ContinuousServingBackendError {
            XCTAssertEqual(error, .shuttingDown)
        } catch is CancellationError {
            XCTFail("shutdown left a late pending start that required cancellation")
        }
        XCTAssertEqual(
            pendingAfterShutdown,
            0,
            "validation must not enqueue a continuation after shutdown snapshots pending admissions")
    }

    func testDynamicAdmissionArrivalAfterSoloPLDJoinsWithoutResubmittingExistingRequest()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 2, 99],
                    20: [3, 99],
                ],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: [
                "solo": [10],
                "late": [20],
            ],
            pieces: [1: "a", 2: "b", 3: "c"],
            stopTokenIDs: [99])

        let soloStart = Task {
            try await backend.start(request(text: "solo", maxTokens: 3))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }
        await backend.diagnosticExpireAdmissionCoalescingWindow()
        let solo = try await soloStart.value
        XCTAssertEqual(solo.route, .soloPLD)
        let soloConsumer = Task { try? await collect(solo.mailbox) }

        try await runUntilDecode(coordinator, recorder: recorder)
        XCTAssertEqual(
            recorder.decodeActions.first,
            .solo(BatchRequestID(1), speculationAllowed: true))

        let lateStart = Task {
            try await backend.start(request(text: "late", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticExecutingAdmissionRequestCount() == 2
        }
        let late = try await lateStart.value
        XCTAssertEqual(late.route, .continuousBatchNoSpec)
        let lateConsumer = Task { try? await collect(late.mailbox) }

        let joinedCoordinatorIDs = await backend.diagnosticCoordinatorRequestIDs()
        XCTAssertEqual(
            joinedCoordinatorIDs,
            [BatchRequestID(1), BatchRequestID(2)])
        XCTAssertEqual(
            recorder.admissionBatchSizes,
            [1, 1],
            "late join should physically submit only the new request, not resubmit the solo request")

        do {
            while recorder.decodeActions.count < 3 {
                _ = try await coordinator.runOneTick()
            }
        } catch {
            let trace = await coordinator.executionTrace()
            let snapshots = await coordinator.snapshots()
            let coordinatorIDs = await backend.diagnosticCoordinatorRequestIDs()
            XCTFail(
                "adaptive tick failed: \(error); trace=\(trace); snapshots=\(snapshots); coordinatorIDs=\(coordinatorIDs)")
            _ = await solo.lease.cancel(.clientDisconnected)
            _ = await late.lease.cancel(.clientDisconnected)
            await backend.shutdown()
            _ = await soloConsumer.value
            _ = await lateConsumer.value
            return
        }
        XCTAssertEqual(
            Array(recorder.decodeActions.prefix(3)),
            [
                .solo(BatchRequestID(1), speculationAllowed: true),
                .drainSoloPipeline(BatchRequestID(1)),
                .batch(
                    [BatchRequestID(1), BatchRequestID(2)],
                    speculationAllowed: false),
            ])

        _ = await solo.lease.cancel(.clientDisconnected)
        _ = await late.lease.cancel(.clientDisconnected)
        await backend.shutdown()
        _ = await soloConsumer.value
        _ = await lateConsumer.value
    }

    func testDynamicAdmissionFailedLateJoinReleasesReservationAndPreservesSolo()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 2),
            runtime: AdmissionFailureContinuousRuntime(
                failureOnAdmission: 2,
                error: .aggregateKVByteLimitExceeded(
                    requested: 8_192,
                    limit: 4_096)),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 16)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: [
                "solo": [10],
                "late": [20],
            ],
            pieces: [:],
            stopTokenIDs: [99])

        let soloStart = Task {
            try await backend.start(request(text: "solo", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }
        await backend.diagnosticExpireAdmissionCoalescingWindow()
        let solo = try await soloStart.value
        XCTAssertEqual(solo.route, .soloPLD)

        do {
            _ = try await backend.start(request(text: "late", maxTokens: 2))
            XCTFail("Expected the late coordinator admission to fail")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(
                error,
                .capacityExceeded(retryAfterSeconds: 2))
        }

        let executingAfterFailure =
            await backend.diagnosticExecutingAdmissionRequestCount()
        let queuedAfterFailure =
            await backend.diagnosticQueuedAdmissionRequestCount()
        let coordinatorIDsAfterFailure =
            await backend.diagnosticCoordinatorRequestIDs()
        XCTAssertEqual(executingAfterFailure, 1)
        XCTAssertEqual(queuedAfterFailure, 0)
        XCTAssertEqual(coordinatorIDsAfterFailure, [BatchRequestID(1)])

        _ = await solo.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testDynamicAdmissionCancellationWhileHeldReleasesPendingStart()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: ContinuousRuntimeRecorder(),
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["held": [10]],
            pieces: [1: "h"],
            stopTokenIDs: [99])

        let start = Task {
            try await backend.start(request(text: "held", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }

        start.cancel()

        do {
            _ = try await start.value
            XCTFail("Expected held start cancellation")
        } catch is CancellationError {
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 0
        }
        let coordinatorIDs = await backend.diagnosticCoordinatorRequestIDs()
        let snapshots = await coordinator.snapshots()
        XCTAssertEqual(coordinatorIDs, [])
        XCTAssertEqual(snapshots, [])
        await backend.shutdown()
    }

    func testDynamicAdmissionAutomaticWindowStartsWithoutDiagnosticIntervention()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: ContinuousRuntimeRecorder(),
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["automatic": [10]],
            pieces: [1: "a"],
            stopTokenIDs: [99],
            coalescing: .automatic(.milliseconds(1)))

        let handle = try await backend.start(
            request(text: "automatic", maxTokens: 2))

        XCTAssertEqual(handle.route, .soloPLD)
        let coordinatorIDs = await backend.diagnosticCoordinatorRequestIDs()
        XCTAssertEqual(
            coordinatorIDs,
            [BatchRequestID(1)])
        _ = await handle.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testDynamicAdmissionCancellingLastHeldDisarmsAutomaticWindow()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 99],
                    20: [2, 99],
                ],
                recorder: ContinuousRuntimeRecorder(),
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: [
                "cancelled": [10],
                "replacement": [20],
            ],
            pieces: [1: "a", 2: "b"],
            stopTokenIDs: [99],
            maximumBatchRequests: 1,
            coalescing: .automatic(.seconds(30)))

        let cancelledStart = Task {
            try await backend.start(
                request(text: "cancelled", maxTokens: 2))
        }
        await waitUntil {
            let held =
                await backend.diagnosticPendingAdmissionRequestCount()
            let armed =
                await backend.diagnosticAdmissionCoalescingWindowArmed()
            return held == 1 && armed
        }
        let firstWindowArmed =
            await backend.diagnosticAdmissionCoalescingWindowArmed()
        XCTAssertTrue(firstWindowArmed)

        cancelledStart.cancel()
        _ = try? await cancelledStart.value
        await waitUntil {
            let held =
                await backend.diagnosticPendingAdmissionRequestCount()
            let armed =
                await backend.diagnosticAdmissionCoalescingWindowArmed()
            return held == 0 && !armed
        }
        let windowArmedAfterCancellation =
            await backend.diagnosticAdmissionCoalescingWindowArmed()
        XCTAssertFalse(
            windowArmedAfterCancellation,
            "the cancelled cohort must not leave a stale timer for the next request")

        let replacementStart = Task {
            try await backend.start(
                request(text: "replacement", maxTokens: 2))
        }
        await waitUntil {
            let held =
                await backend.diagnosticPendingAdmissionRequestCount()
            let armed =
                await backend.diagnosticAdmissionCoalescingWindowArmed()
            return held == 1 && armed
        }
        let replacementWindowArmed =
            await backend.diagnosticAdmissionCoalescingWindowArmed()
        XCTAssertTrue(replacementWindowArmed)
        await backend.diagnosticExpireAdmissionCoalescingWindow()
        let replacement = try await replacementStart.value

        _ = await replacement.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testDynamicAdmissionShutdownReleasesEveryHeldContinuation()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: ContinuousRuntimeRecorder(),
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["held": [10]],
            pieces: [1: "h"],
            stopTokenIDs: [99])
        let start = Task {
            try await backend.start(request(text: "held", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }

        await backend.shutdown()

        do {
            _ = try await start.value
            XCTFail("Expected shutdown to release the held start")
        } catch let error as ContinuousServingBackendError {
            XCTAssertEqual(error, .shuttingDown)
        }
        let pendingCount =
            await backend.diagnosticPendingAdmissionRequestCount()
        let snapshots = await coordinator.snapshots()
        XCTAssertEqual(
            pendingCount,
            0)
        XCTAssertTrue(snapshots.isEmpty)
    }

    func testDynamicAdmissionRejectsQueueOverflowBeforeCoordinatorAdmission()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 99],
                    20: [2, 99],
                ],
                recorder: ContinuousRuntimeRecorder(),
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: [
                "held": [10],
                "overflow": [20],
            ],
            pieces: [1: "h", 2: "o"],
            stopTokenIDs: [99],
            maximumBatchRequests: 1,
            maximumQueuedRequests: 0)
        let held = Task {
            try await backend.start(request(text: "held", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }

        do {
            _ = try await backend.start(
                request(text: "overflow", maxTokens: 2))
            XCTFail("Expected the bounded admission queue to reject overflow")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(
                error,
                .queueFull(retryAfterSeconds: 2))
        }
        let snapshots = await coordinator.snapshots()
        XCTAssertTrue(snapshots.isEmpty)

        held.cancel()
        _ = try? await held.value
        await backend.shutdown()
    }

    func testModelCapabilitiesClampRequestBeforeCoordinatorSubmission()
        async throws
    {
        let capabilities = try ServingModelCapabilities(
            model: "fixture",
            nativeMaxContextTokens: 10,
            effectiveMaxContextTokens: 10,
            requestedDefaultCompletionTokens: 4,
            maximumCompletionTokens: 6,
            maximumNonStreamingCompletionTokens: 6,
            completionLimitPolicy: .clamp)
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 99]],
                recorder: ContinuousRuntimeRecorder()),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["long": [10, 11, 12, 13, 14, 15, 16, 17]],
            pieces: [1: "a", 2: "b"],
            stopTokenIDs: [99],
            modelCapabilities: capabilities)

        let handle = try await backend.start(
            request(text: "long", maxTokens: 6))

        XCTAssertEqual(handle.completionBudgetResolution?.requestedCompletionTokens, 6)
        XCTAssertEqual(handle.completionBudgetResolution?.appliedCompletionTokens, 2)
        XCTAssertEqual(handle.completionBudgetResolution?.maximumAllowedCompletionTokens, 2)
        XCTAssertEqual(handle.completionBudgetResolution?.renderedPromptTokens, 8)
        XCTAssertEqual(handle.completionBudgetResolution?.wasClamped, true)
        XCTAssertEqual(handle.completionBudgetResolution?.limitingFactor, .contextWindow)
        let snapshots = await coordinator.snapshots()
        XCTAssertEqual(snapshots.map(\.request.maxOutputTokens), [2])
        XCTAssertEqual(snapshots.map(\.request.promptTokenCount), [8])

        _ = await handle.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testModelCapabilitiesRejectBeforeDynamicAdmissionOrRuntimeValidation()
        async throws
    {
        let capabilities = try ServingModelCapabilities(
            model: "fixture",
            nativeMaxContextTokens: 10,
            effectiveMaxContextTokens: 10,
            requestedDefaultCompletionTokens: 4,
            maximumCompletionTokens: 6,
            maximumNonStreamingCompletionTokens: 6,
            completionLimitPolicy: .reject)
        let runtime = CountingValidationContinuousRuntime()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: runtime,
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["too-long": [10, 11, 12, 13, 14, 15, 16, 17]],
            pieces: [:],
            stopTokenIDs: [99],
            maximumBatchRequests: 1,
            maximumQueuedRequests: 1,
            modelCapabilities: capabilities)

        do {
            _ = try await backend.start(
                request(text: "too-long", maxTokens: 6))
            XCTFail("Expected prompt-aware completion budget rejection")
        } catch let error as OpenAIServingError {
            guard case .invalidRequestWithCode(_, let param, let code) = error else {
                XCTFail("expected invalidRequestWithCode, got \(error)")
                await backend.shutdown()
                return
            }
            XCTAssertEqual(param, "max_completion_tokens")
            XCTAssertEqual(code, "completion_limit_exceeded")
        }
        let pendingAdmissionCount =
            await backend.diagnosticPendingAdmissionRequestCount()
        XCTAssertEqual(pendingAdmissionCount, 0)
        XCTAssertEqual(runtime.validationCount, 0)
        XCTAssertEqual(runtime.admissionBatchCount, 0)
        let snapshots = await coordinator.snapshots()
        XCTAssertTrue(snapshots.isEmpty)

        await backend.shutdown()
    }

    func testModelCapabilitiesKeepIndependentResolutionsForCoalescedRequests()
        async throws
    {
        let capabilities = try ServingModelCapabilities(
            model: "fixture",
            nativeMaxContextTokens: 10,
            effectiveMaxContextTokens: 10,
            requestedDefaultCompletionTokens: 4,
            maximumCompletionTokens: 6,
            maximumNonStreamingCompletionTokens: 6,
            completionLimitPolicy: .clamp)
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 99],
                    20: [2, 99],
                ],
                recorder: recorder),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: [
                "short": [10, 11],
                "long": [20, 21, 22, 23, 24, 25, 26, 27],
            ],
            pieces: [1: "s", 2: "l"],
            stopTokenIDs: [99],
            modelCapabilities: capabilities)

        let shortStart = Task {
            try await backend.start(request(text: "short", maxTokens: 5))
        }
        let longStart = Task {
            try await backend.start(request(text: "long", maxTokens: 5))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 2
        }

        await backend.diagnosticExpireAdmissionCoalescingWindow()
        let short = try await shortStart.value
        let long = try await longStart.value

        XCTAssertEqual(short.completionBudgetResolution?.renderedPromptTokens, 2)
        XCTAssertEqual(short.completionBudgetResolution?.appliedCompletionTokens, 5)
        XCTAssertEqual(short.completionBudgetResolution?.wasClamped, false)
        XCTAssertEqual(long.completionBudgetResolution?.renderedPromptTokens, 8)
        XCTAssertEqual(long.completionBudgetResolution?.appliedCompletionTokens, 2)
        XCTAssertEqual(long.completionBudgetResolution?.wasClamped, true)
        let snapshots = await coordinator.snapshots()
        XCTAssertEqual(
            snapshots.sorted { $0.request.promptTokenCount < $1.request.promptTokenCount }
                .map(\.request.maxOutputTokens),
            [5, 2])

        _ = await short.lease.cancel(.clientDisconnected)
        _ = await long.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testDefaultConfigurationStillStartsImmediateBatchNoSpec()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 2),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["default": [10]],
            pieces: [1: "d"],
            stopTokenIDs: [99])

        let handle = try await backend.start(
            request(text: "default", maxTokens: 2))

        XCTAssertEqual(handle.route, .continuousBatchNoSpec)
        let coordinatorIDs = await backend.diagnosticCoordinatorRequestIDs()
        XCTAssertEqual(
            coordinatorIDs,
            [BatchRequestID(1)])
        try await runUntilDecode(coordinator, recorder: recorder)
        XCTAssertEqual(
            recorder.decodeActions.first,
            .solo(BatchRequestID(1), speculationAllowed: false))

        _ = await handle.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testContinuousRoutePublishesExactTextUsageAndResolvedStopSet() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 99, 3]],
                recorder: recorder),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["hello": [10, 11]],
            pieces: [1: "hel", 2: "lo", 3: "hidden"],
            stopTokenIDs: [2_048, 99])

        let handle = try await backend.start(
            request(text: "hello", maxTokens: 8))
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(handle.route, .continuousBatchNoSpec)
        XCTAssertEqual(
            events,
            [
                .text("hel"),
                .text("lo"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .stop,
                        usage: OpenAIChatUsage(
                            promptTokens: 2,
                            completionTokens: 2))),
            ])
        XCTAssertTrue(recorder.decodeActions.allSatisfy(\.speculationDisabled))
        await waitUntil {
            await backend.snapshot().activeRequests == 0
        }
        let finalSnapshot = await backend.snapshot()
        XCTAssertEqual(finalSnapshot.activeRequests, 0)
        XCTAssertEqual(finalSnapshot.coordinatorSlots, 0)
        XCTAssertEqual(finalSnapshot.reservedKVBytes, 0)
        XCTAssertEqual(finalSnapshot.maxReservedKVBytes, 4_096)
    }

    // E3 (`ignore_eos` predeclaration), through the serving adapter: the request flag reaches the
    // coordinator submission, so the model stop token (99) is generated like any token and the
    // request ends by budget with finish_reason `length`. The same script without the flag is
    // `testContinuousRoutePublishesExactTextUsageAndResolvedStopSet` (stops at 99, finish `stop`).
    func testContinuousRouteIgnoreEOSGeneratesPastTheStopTokenAndEndsByLength() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 99, 3, 4, 5]],
                recorder: recorder),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["hello": [10, 11]],
            pieces: [1: "a", 2: "b", 99: "</s>", 3: "c", 4: "d", 5: "e"],
            stopTokenIDs: [2_048, 99],
            mailboxCapacity: .init(maxDeltas: 16, maxBytes: 4_096))

        var ignoring = request(text: "hello", maxTokens: 5)
        ignoring.ignoreEOS = true
        let handle = try await backend.start(ignoring)
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(
            events,
            [
                .text("a"), .text("b"), .text("</s>"), .text("c"), .text("d"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .length,
                        usage: OpenAIChatUsage(promptTokens: 2, completionTokens: 5))),
            ])
        await waitUntil {
            await backend.snapshot().activeRequests == 0
        }
    }

    func testSnapshotReportsActiveRequestResourcesMLXMemoryAndPromptFreeJSON()
        async throws
    {
        let prompt = "fixture-secret-prompt"
        let output = "fixture-secret-output"
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 3, 4, 5, 6, 99]],
                recorder: recorder),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: [prompt: [10]],
            pieces: [1: output, 2: "!", 3: "?"],
            stopTokenIDs: [99],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 64))

        let handle = try await backend.start(
            request(text: prompt, maxTokens: 8))
        await waitUntil {
            let mailbox = await handle.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1
                && mailbox.waitingProducers == 1
        }

        let activeSnapshot = await backend.snapshot()
        XCTAssertEqual(activeSnapshot.activeRequests, 1)
        XCTAssertEqual(activeSnapshot.coordinatorSlots, 1)
        XCTAssertEqual(activeSnapshot.reservedKVBytes, 1_024)
        XCTAssertEqual(activeSnapshot.maxReservedKVBytes, 4_096)
        XCTAssertGreaterThanOrEqual(activeSnapshot.mlxActiveBytes, 0)
        XCTAssertGreaterThanOrEqual(activeSnapshot.mlxCacheBytes, 0)
        XCTAssertGreaterThanOrEqual(activeSnapshot.mlxPeakBytes, 0)

        let encoded = try JSONEncoder().encode(activeSnapshot)
        let json = String(decoding: encoded, as: UTF8.self)
        XCTAssertFalse(json.contains(prompt))
        XCTAssertFalse(json.contains(output))

        let cancelled = await handle.lease.cancel(.clientDisconnected)
        XCTAssertTrue(cancelled)
        await assertMailboxCancelled(
            handle.mailbox,
            reason: .clientDisconnected)
        await backend.shutdown()

        let finalSnapshot = await backend.snapshot()
        XCTAssertEqual(finalSnapshot.activeRequests, 0)
        XCTAssertEqual(finalSnapshot.coordinatorSlots, 0)
        XCTAssertEqual(finalSnapshot.reservedKVBytes, 0)
    }

    func testRequestStopSplitAcrossTokenChunksCancelsCoordinatorNormally() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 2),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 3, 4, 99]],
                recorder: ContinuousRuntimeRecorder()),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["stop": [10]],
            pieces: [
                1: "hello<",
                2: "stop",
                3: ">hidden",
                4: "tail",
            ],
            stopTokenIDs: [99])

        let handle = try await backend.start(
            request(
                text: "stop",
                maxTokens: 8,
                stop: ["<stop>"]))
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(
            events,
            [
                .text("hello"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .stop,
                        usage: OpenAIChatUsage(
                            promptTokens: 1,
                            completionTokens: 3))),
            ])
        await waitUntil {
            await coordinator.snapshots().isEmpty
        }
        let finalSlots = await coordinator.snapshots()
        XCTAssertTrue(finalSlots.isEmpty)
    }

    func testRequestStopAtOutputBudgetReportsStopInsteadOfLength() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 2),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 3, 4, 5, 6, 99]],
                recorder: ContinuousRuntimeRecorder()),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["stop-at-budget": [10]],
            pieces: [
                1: "a",
                2: "b<",
                3: "stop>",
            ],
            stopTokenIDs: [99])

        let handle = try await backend.start(
            request(
                text: "stop-at-budget",
                maxTokens: 3,
                stop: ["<stop>"]))
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(
            events,
            [
                .text("a"),
                .text("b"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .stop,
                        usage: OpenAIChatUsage(
                            promptTokens: 1,
                            completionTokens: 3))),
            ])
        await waitUntil {
            await coordinator.snapshots().isEmpty
        }
    }

    func testDetectedStopReleasesCoordinatorBeforeBlockedPrefixPublication() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 3, 4, 5, 6, 99]],
                recorder: ContinuousRuntimeRecorder()),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["blocked-stop": [10]],
            pieces: [
                1: "buffer",
                2: "prefix<stop>",
                3: "hidden",
            ],
            stopTokenIDs: [99],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 64))

        let handle = try await backend.start(
            request(
                text: "blocked-stop",
                maxTokens: 4,
                stop: ["<stop>"]))
        await waitUntil {
            let mailbox = await handle.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1
                && mailbox.waitingProducers == 1
        }

        let slots = await coordinator.snapshots()
        let resources = await coordinator.runtimeResourceSnapshot()
        XCTAssertTrue(slots.isEmpty)
        XCTAssertEqual(resources?.reservedKVBytes, 0)

        let cancelled = await handle.lease.cancel(.clientDisconnected)
        XCTAssertTrue(cancelled)
        await assertMailboxCancelled(
            handle.mailbox,
            reason: .clientDisconnected)
    }

    // parallel_tool_calls == false stops generation at the first completed call (cycle 244). The
    // script finishes call A at token 2 and would finish call B at token 4 before EOS.
    private static let twoCallPieces: [Int: String] = [
        1: #"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}"#,
        2: "</tool_call>",
        3: "\n" + #"<tool_call>{"name":"get_weather","arguments":{"city":"Rome"}}"#,
        4: "</tool_call>",
    ]

    private func runToolScript(
        script: [Int],
        pieces: [Int: String],
        parallelToolCalls: Bool?
    ) async throws -> (
        calls: [[OpenAIToolCall]], completion: ServingGenerationCompletion?,
        slotsAfter: Int
    ) {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: script],
                recorder: ContinuousRuntimeRecorder()),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["tools": [10]],
            pieces: pieces,
            stopTokenIDs: [99])
        let handle = try await backend.start(
            request(
                text: "tools", maxTokens: 8, tools: [continuousWeatherTool],
                parallelToolCalls: parallelToolCalls))
        let events = try await collect(handle.mailbox)
        let slotsAfter = await coordinator.snapshots().count
        var calls: [[OpenAIToolCall]] = []
        var completion: ServingGenerationCompletion?
        for event in events {
            switch event {
            case .toolCalls(let batch): calls.append(batch)
            case .completion(let value): completion = value
            default: break
            }
        }
        return (calls, completion, slotsAfter)
    }

    func testParallelFalseStopsAtTheFirstCompletedCallAndReleasesTheSlot() async throws {
        let result = try await runToolScript(
            script: [1, 2, 3, 4, 99], pieces: Self.twoCallPieces, parallelToolCalls: false)

        XCTAssertEqual(result.calls.count, 1, "exactly one .toolCalls event")
        XCTAssertEqual(result.calls.first?.count, 1)
        XCTAssertTrue(result.calls.first?.first?.function.arguments.contains("Paris") == true)
        XCTAssertEqual(result.completion?.finishReason, .toolCalls)
        XCTAssertEqual(result.completion?.usage.completionTokens, 2)
        XCTAssertEqual(result.slotsAfter, 0, "coordinator slot must be released by the stop")
    }

    func testParallelAbsentOrTrueKeepsEveryCallAndRunsToEOS() async throws {
        for parallel in [nil, true] as [Bool?] {
            let result = try await runToolScript(
                script: [1, 2, 3, 4, 99], pieces: Self.twoCallPieces, parallelToolCalls: parallel)

            let flat = result.calls.flatMap { $0 }
            XCTAssertEqual(flat.count, 2, "parallel=\(String(describing: parallel))")
            XCTAssertTrue(flat.first?.function.arguments.contains("Paris") == true)
            XCTAssertTrue(flat.last?.function.arguments.contains("Rome") == true)
            XCTAssertEqual(result.completion?.finishReason, .toolCalls)
            XCTAssertEqual(result.completion?.usage.completionTokens, 4)
        }
    }

    // The tool parser sees only the bytes the client is shown as content (cycle 247): a call drafted
    // inside the reasoning block (before the first `</think>`) is reasoning text, not a call.
    private static let draftInsideThinkingPieces: [Int: String] = [
        1: "<think>draft ",
        2: #"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}"#,
        3: "</tool_call>",
        4: "</think>",
        5: #"<tool_call>{"name":"get_weather","arguments":{"city":"Rome"}}"#,
        6: "</tool_call>",
    ]

    private func runDraftInsideThinking(
        thinksByDefault: Bool, parallelToolCalls: Bool?
    ) async throws -> (
        calls: [OpenAIToolCall], completion: ServingGenerationCompletion?, text: String,
        separates: Bool
    ) {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 3, 4, 5, 6, 99]],
                recorder: ContinuousRuntimeRecorder()),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["tools": [10]],
            pieces: Self.draftInsideThinkingPieces,
            stopTokenIDs: [99],
            thinksByDefault: thinksByDefault)
        let handle = try await backend.start(
            request(
                text: "tools", maxTokens: 8, tools: [continuousWeatherTool],
                parallelToolCalls: parallelToolCalls, enableThinking: true))
        let events = try await collect(handle.mailbox)
        var calls: [OpenAIToolCall] = []
        var completion: ServingGenerationCompletion?
        var text = ""
        for event in events {
            switch event {
            case .toolCalls(let batch): calls.append(contentsOf: batch)
            case .completion(let value): completion = value
            case .text(let value): text += value
            default: break
            }
        }
        return (calls, completion, text, handle.separatesReasoning)
    }

    func testToolCallDraftedInsideReasoningIsNotParsedUnderParallelFalse() async throws {
        let result = try await runDraftInsideThinking(thinksByDefault: true, parallelToolCalls: false)

        XCTAssertTrue(result.separates)
        XCTAssertEqual(result.calls.count, 1)
        XCTAssertTrue(result.calls.first?.function.arguments.contains("Rome") == true)
        XCTAssertFalse(result.calls.contains { $0.function.arguments.contains("Paris") })
        XCTAssertEqual(result.completion?.finishReason, .toolCalls)
        XCTAssertEqual(
            result.completion?.usage.completionTokens, 6,
            "the draft must not stop generation: the whole script is generated")
        XCTAssertTrue(result.text.contains("Paris"), "the draft reaches the client as reasoning text")
        XCTAssertTrue(result.text.hasPrefix("<think>draft <tool_call>{"))
        XCTAssertTrue(result.text.contains("</think>"))
    }

    func testToolCallDraftedInsideReasoningIsNotParsedUnderParallelAbsentOrTrue() async throws {
        for parallel in [nil, true] as [Bool?] {
            let result = try await runDraftInsideThinking(
                thinksByDefault: true, parallelToolCalls: parallel)

            XCTAssertEqual(result.calls.count, 1, "parallel=\(String(describing: parallel))")
            XCTAssertTrue(result.calls.first?.function.arguments.contains("Rome") == true)
            XCTAssertEqual(result.completion?.usage.completionTokens, 6)
            XCTAssertTrue(result.text.contains("Paris"))
        }
    }

    // Control: a handle that does not separate reasoning has no gate, so today's behavior holds.
    func testWithoutReasoningSeparationStillParsesTheDraftedCall() async throws {
        let stopped = try await runDraftInsideThinking(thinksByDefault: false, parallelToolCalls: false)
        XCTAssertFalse(stopped.separates)
        XCTAssertEqual(stopped.calls.count, 1)
        XCTAssertTrue(stopped.calls.first?.function.arguments.contains("Paris") == true)
        XCTAssertEqual(stopped.completion?.usage.completionTokens, 3)

        let all = try await runDraftInsideThinking(thinksByDefault: false, parallelToolCalls: nil)
        XCTAssertEqual(all.calls.count, 2)
        XCTAssertTrue(all.calls.first?.function.arguments.contains("Paris") == true)
        XCTAssertTrue(all.calls.last?.function.arguments.contains("Rome") == true)
    }

    // End of stream with a tool call buffered AND the stop filter holding a proper prefix of a stop
    // string (cycle 250). The held tail is LATER in stream order than everything the processor has
    // buffered, so it must reach the processor (and the gate) rather than the client directly: the
    // client text, concatenated, is the visible stream in order, and the parser sees what the client sees.
    private func runEOSTail(
        pieces: [Int: String], script: [Int], stop: [String], thinksByDefault: Bool
    ) async throws -> (
        calls: [OpenAIToolCall], completion: ServingGenerationCompletion?, texts: [String]
    ) {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: script],
                recorder: ContinuousRuntimeRecorder()),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["tools": [10]],
            pieces: pieces,
            stopTokenIDs: [99],
            thinksByDefault: thinksByDefault)
        let handle = try await backend.start(
            request(
                text: "tools", maxTokens: 8, stop: stop, tools: [continuousWeatherTool],
                enableThinking: thinksByDefault ? true : nil))
        let events = try await collect(handle.mailbox)
        var calls: [OpenAIToolCall] = []
        var completion: ServingGenerationCompletion?
        var texts: [String] = []
        for event in events {
            switch event {
            case .toolCalls(let batch): calls.append(contentsOf: batch)
            case .completion(let value): completion = value
            case .text(let value): texts.append(value)
            default: break
            }
        }
        return (calls, completion, texts)
    }

    // C1
    func testEOSStopFilterTailFollowsTheProcessorResidualInStreamOrder() async throws {
        for thinks in [false, true] {
            let result = try await runEOSTail(
                pieces: [
                    1: "</think>hello ",
                    2: #"<tool_call>{"name":"get_weather","arguments":{"city":"Pa"#,
                    3: "ST",
                ],
                script: [1, 2, 3, 99], stop: ["STOP"], thinksByDefault: thinks)

            XCTAssertEqual(
                result.texts.joined(),
                #"</think>hello <tool_call>{"name":"get_weather","arguments":{"city":"PaST"#,
                "thinks=\(thinks): residual precedes the tail, concatenation is the visible stream")
            XCTAssertTrue(result.calls.isEmpty, "thinks=\(thinks)")
            XCTAssertEqual(result.completion?.finishReason, .stop, "thinks=\(thinks)")
        }
    }

    // C2: the held tail is the call's closing tag; the parser must see it and complete the call
    // instead of the tag leaking to the client as text.
    func testEOSStopFilterTailThatClosesAToolCallIsParsedNotLeaked() async throws {
        for thinks in [false, true] {
            let result = try await runEOSTail(
                pieces: [
                    1: "</think>hello ",
                    2: #"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}"#,
                    3: "</tool_call>",
                ],
                script: [1, 2, 3, 99], stop: ["</tool_call>X"], thinksByDefault: thinks)

            XCTAssertEqual(result.calls.count, 1, "thinks=\(thinks)")
            XCTAssertTrue(result.calls.first?.function.arguments.contains("Paris") == true)
            XCTAssertEqual(result.completion?.finishReason, .toolCalls, "thinks=\(thinks)")
            XCTAssertEqual(
                result.texts.joined(), "</think>hello ",
                "thinks=\(thinks): the closing tag is parser input, not client text")
        }
    }

    func testParallelFalseWithNoCallIsUnchangedAndRunsToEOS() async throws {
        let result = try await runToolScript(
            script: [1, 2, 99], pieces: [1: "hello ", 2: "world"], parallelToolCalls: false)

        XCTAssertTrue(result.calls.isEmpty)
        XCTAssertEqual(result.completion?.finishReason, .stop)
        XCTAssertEqual(result.completion?.usage.completionTokens, 2)
    }

    func testQueueExhaustionDisconnectAndReplacementRecoverOneSharedCoordinator() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 2, 99],
                    20: [3, 4, 99],
                    30: [5, 99],
                ],
                recorder: ContinuousRuntimeRecorder()),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: [
                "active": [10],
                "queued": [20],
                "replacement": [30],
            ],
            pieces: [
                1: "a",
                2: "b",
                3: "c",
                4: "d",
                5: "r",
            ],
            stopTokenIDs: [99])

        let active = try await backend.start(
            request(text: "active", maxTokens: 2))
        _ = try await coordinator.runOneTick()
        let queued = try await backend.start(
            request(text: "queued", maxTokens: 2))

        do {
            _ = try await backend.start(
                request(text: "replacement", maxTokens: 1))
            XCTFail("Expected queue-full rejection")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(error, .queueFull(retryAfterSeconds: 2))
        }

        let queuedCancelled = await queued.lease.cancel(.clientDisconnected)
        XCTAssertTrue(queuedCancelled)
        await assertMailboxCancelled(
            queued.mailbox,
            reason: .clientDisconnected)
        let activeCancelled = await active.lease.cancel(.clientDisconnected)
        XCTAssertTrue(activeCancelled)
        await assertMailboxCancelled(
            active.mailbox,
            reason: .clientDisconnected)
        await waitUntil {
            await coordinator.snapshots().isEmpty
        }

        let replacement = try await backend.start(
            request(text: "replacement", maxTokens: 1))
        let collector = Task { try await collect(replacement.mailbox) }
        while try await coordinator.runOneTick() {}
        let replacementEvents = try await collector.value
        XCTAssertEqual(
            replacementEvents,
            [
                .text("r"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .length,
                        usage: OpenAIChatUsage(
                            promptTokens: 1,
                            completionTokens: 1))),
            ])
        let finalSnapshot = await backend.snapshot()
        XCTAssertEqual(finalSnapshot.activeRequests, 0)
        XCTAssertEqual(finalSnapshot.coordinatorSlots, 0)
        XCTAssertEqual(finalSnapshot.reservedKVBytes, 0)
        XCTAssertEqual(finalSnapshot.maxReservedKVBytes, 4_096)
    }

    func testRuntimeCapacityFailuresAreTypedBeforeHTTPGenerationStarts()
        async throws
    {
        let singleCoordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: AdmissionFailureContinuousRuntime(
                failureOnAdmission: 1,
                error: .contextLimitExceeded(
                    BatchRequestID(1),
                    requested: 4_097,
                    limit: 4_096)),
            automaticDrive: false,
            publicationCapacity: 1)
        let singleBackend = makeBackend(
            coordinator: singleCoordinator,
            promptByText: ["oversized": [10]],
            pieces: [:],
            stopTokenIDs: [99])

        do {
            _ = try await singleBackend.start(
                request(text: "oversized", maxTokens: 1))
            XCTFail("Expected a typed request-size admission failure")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(error, .requestTooLarge())
        }

        let sharedCoordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 1),
            runtime: AdmissionFailureContinuousRuntime(
                failureOnAdmission: 2,
                error: .aggregateKVByteLimitExceeded(
                    requested: 8_192,
                    limit: 4_096)),
            automaticDrive: false,
            publicationCapacity: 1)
        let sharedBackend = makeBackend(
            coordinator: sharedCoordinator,
            promptByText: [
                "active": [10],
                "blocked": [20],
            ],
            pieces: [:],
            stopTokenIDs: [99])
        let active = try await sharedBackend.start(
            request(text: "active", maxTokens: 1))
        _ = try await sharedCoordinator.runOneTick()

        do {
            _ = try await sharedBackend.start(
                request(text: "blocked", maxTokens: 1))
            XCTFail("Expected a typed aggregate-capacity admission failure")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(
                error,
                .capacityExceeded(retryAfterSeconds: 2))
        }
        _ = await active.lease.cancel(.clientDisconnected)
        await sharedBackend.shutdown()

        let permanentCoordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 1),
            runtime: AdmissionFailureContinuousRuntime(
                failureOnAdmission: 2,
                error: .requestReservedKVByteLimitExceeded(
                    BatchRequestID(2),
                    requested: 8_192,
                    limit: 4_096)),
            automaticDrive: false,
            publicationCapacity: 1)
        let permanentBackend = makeBackend(
            coordinator: permanentCoordinator,
            promptByText: [
                "active": [10],
                "permanent": [20],
            ],
            pieces: [:],
            stopTokenIDs: [99])
        let permanentActive = try await permanentBackend.start(
            request(text: "active", maxTokens: 1))
        _ = try await permanentCoordinator.runOneTick()

        do {
            _ = try await permanentBackend.start(
                request(text: "permanent", maxTokens: 1))
            XCTFail("Expected permanent request oversize to remain non-retryable")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(error, .requestTooLarge())
        }
        _ = await permanentActive.lease.cancel(.clientDisconnected)
        await permanentBackend.shutdown()
    }

    func testDynamicAdmissionRejectsPermanentOversizeBeforeItOccupiesHoldOrQueue()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: AdmissionValidationFailureContinuousRuntime(
                error: .contextLimitExceeded(
                    BatchRequestID(1),
                    requested: 4_097,
                    limit: 4_096)),
            automaticDrive: false,
            publicationCapacity: 1)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["oversized": [10]],
            pieces: [:],
            stopTokenIDs: [99],
            maximumBatchRequests: 1,
            maximumQueuedRequests: 1)

        let start = Task {
            try await backend.start(
                request(text: "oversized", maxTokens: 1))
        }
        for _ in 0 ..< 100 {
            await Task.yield()
        }
        let pending =
            await backend.diagnosticPendingAdmissionRequestCount()
        XCTAssertEqual(
            pending,
            0,
            "permanently oversized work must fail before dynamic hold/queue admission")
        if pending != 0 {
            start.cancel()
        }
        do {
            _ = try await start.value
            XCTFail("Expected a typed request-size admission failure")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(error, .requestTooLarge())
        } catch is CancellationError {
            XCTFail("oversized request reached the dynamic hold instead of failing preflight")
        }
        await backend.shutdown()
    }

    func testBoundedDiagnosticsExposeCoordinatorMembershipAndTraceOnly()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 2),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 99],
                    20: [2, 99],
                ],
                recorder: ContinuousRuntimeRecorder()),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 8)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: [
                "one": [10],
                "two": [20],
            ],
            pieces: [1: "a", 2: "b"],
            stopTokenIDs: [99])

        let one = try await backend.start(
            request(text: "one", maxTokens: 1))
        let two = try await backend.start(
            request(text: "two", maxTokens: 1))

        let requestIDs = await backend.diagnosticCoordinatorRequestIDs()
        XCTAssertEqual(
            requestIDs,
            [BatchRequestID(1), BatchRequestID(2)])
        let queuedSnapshots = await backend
            .diagnosticCoordinatorSnapshots()
        XCTAssertEqual(queuedSnapshots.map(\.request.id), [
            BatchRequestID(1), BatchRequestID(2),
        ])
        _ = try await coordinator.runOneTick()
        let trace = await backend.diagnosticCoordinatorExecutionTrace()
        XCTAssertTrue(trace.contains {
            if case .operation(.prefill(let slice)) = $0 {
                return slice.id == BatchRequestID(1)
                    || slice.id == BatchRequestID(2)
            }
            return false
        })

        _ = await one.lease.cancel(.clientDisconnected)
        _ = await two.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testServingMailboxBackpressureStopsFurtherCoordinatorDecode() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 3, 4, 99]],
                recorder: recorder),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["slow": [10]],
            pieces: [1: "a", 2: "b", 3: "c", 4: "d"],
            stopTokenIDs: [99],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 8))

        let handle = try await backend.start(
            request(text: "slow", maxTokens: 4))
        await waitUntil {
            let mailbox = await handle.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1
                && mailbox.waitingProducers == 1
                && recorder.decodeActions.count >= 3
        }
        let blockedDecodeCount = recorder.decodeActions.count
        for _ in 0 ..< 100 {
            await Task.yield()
        }
        XCTAssertEqual(recorder.decodeActions.count, blockedDecodeCount)

        let cancelled = await handle.lease.cancel(.backpressureTimeout)
        XCTAssertTrue(cancelled)
        await assertMailboxCancelled(
            handle.mailbox,
            reason: .backpressureTimeout)
        await waitUntil {
            let snapshot = await backend.snapshot()
            return snapshot.activeRequests == 0
                && snapshot.coordinatorSlots == 0
                && snapshot.reservedKVBytes == 0
        }
    }

    func testShutdownCancelsEveryAcceptedRequestAndRejectsNewWork() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 2, 3, 4, 5, 6, 99],
                    20: [4, 5, 99],
                ],
                recorder: ContinuousRuntimeRecorder()),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: [
                "active": [10],
                "queued": [20],
            ],
            pieces: [1: "a", 2: "b", 3: "c", 4: "d", 5: "e"],
            stopTokenIDs: [99],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 8))

        let active = try await backend.start(
            request(text: "active", maxTokens: 3))
        let queued = try await backend.start(
            request(text: "queued", maxTokens: 2))
        await waitUntil {
            let mailbox = await active.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1
                && mailbox.waitingProducers == 1
        }

        await backend.shutdown()
        await assertMailboxCancelled(active.mailbox, reason: .shutdown)
        await assertMailboxCancelled(queued.mailbox, reason: .shutdown)
        let activeState = await active.lease.state
        let queuedState = await queued.lease.state
        XCTAssertEqual(activeState, .cancelled(.shutdown))
        XCTAssertEqual(queuedState, .cancelled(.shutdown))
        let finalSnapshot = await backend.snapshot()
        XCTAssertEqual(finalSnapshot.activeRequests, 0)
        XCTAssertEqual(finalSnapshot.coordinatorSlots, 0)
        XCTAssertEqual(finalSnapshot.reservedKVBytes, 0)
        XCTAssertEqual(finalSnapshot.maxReservedKVBytes, 4_096)

        do {
            _ = try await backend.start(
                request(text: "active", maxTokens: 1))
            XCTFail("Expected shutdown admission rejection")
        } catch let error as ContinuousServingBackendError {
            XCTAssertEqual(error, .shuttingDown)
        }
    }
}

private extension BatchDecodeAction {
    var speculationDisabled: Bool {
        switch self {
        case .drainSoloPipeline:
            true
        case .solo(_, let speculationAllowed),
            .batch(_, let speculationAllowed):
            !speculationAllowed
        }
    }
}

private final class ContinuousRuntimeRecorder: Sendable {
    private struct State: Sendable {
        var decodeActions: [BatchDecodeAction] = []
        var admissionBatchSizes: [Int] = []
        var submissions: [ContinuousBatchSubmission] = []
        var activeRequests = 0
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    var decodeActions: [BatchDecodeAction] {
        state.withLock { $0.decodeActions }
    }

    var admissionBatchSizes: [Int] {
        state.withLock { $0.admissionBatchSizes }
    }

    /// Every submission the coordinator handed to the runtime, in admission order.
    var submissions: [ContinuousBatchSubmission] {
        state.withLock { $0.submissions }
    }

    func record(_ action: BatchDecodeAction) {
        state.withLock { $0.decodeActions.append(action) }
    }

    func admitted(_ count: Int) {
        state.withLock {
            $0.admissionBatchSizes.append(count)
            $0.activeRequests += count
        }
    }

    func admitted(_ admissions: [ContinuousBatchRuntimeAdmission]) {
        state.withLock { $0.submissions.append(contentsOf: admissions.map(\.submission)) }
    }

    func removed() {
        state.withLock { state in
            state.activeRequests = max(0, state.activeRequests - 1)
        }
    }

    var resources: ContinuousBatchRuntimeResourceSnapshot {
        state.withLock { state in
            ContinuousBatchRuntimeResourceSnapshot(
                kvBytesPerToken: 64,
                reservedKVBytes: state.activeRequests * 1_024,
                maxReservedKVBytes: 4_096)
        }
    }
}

private final class BlockingRuntimeResourceSnapshotGate: Sendable {
    private struct State: Sendable {
        var hasEntered = false
        var isReleased = false
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    var hasEntered: Bool {
        state.withLock(\.hasEntered)
    }

    func wait() {
        state.withLock { $0.hasEntered = true }
        while !state.withLock(\.isReleased) {
            Thread.sleep(forTimeInterval: 0.001)
        }
    }

    func release() {
        state.withLock { $0.isReleased = true }
    }
}

private final class FixtureContinuousRuntime: ContinuousBatchRuntime {
    private struct Slot {
        var processedTokens = 0
        var ready = false
        var soloPipelineState: BatchSoloPipelineState = .canonical
        var script: [Int]
        var cursor = 0
    }

    private let scriptsByPromptHead: [Int: [Int]]
    private let recorder: ContinuousRuntimeRecorder
    private let allowsSpeculation: Bool
    private let speculativeSoloState: BatchSoloPipelineState
    private let resourceSnapshotGate: BlockingRuntimeResourceSnapshotGate?
    /// When `true` the fixture plays the real runtime's side of the `ContinuousBatchRowTokenMask`
    /// contract: after each emitted token it calls `advance(token:)` and then asks for the next
    /// bitset, BEFORE the backend's `publish` sees the token. The fixture never applies the bitset (its
    /// script is fixed), so a test chooses a script that either follows or violates the constraint.
    private let advancesTokenMasks: Bool
    private var admittedIDs: Set<BatchRequestID> = []
    private var promptHeadByID: [BatchRequestID: Int] = [:]
    private var slots: [BatchRequestID: Slot] = [:]
    private var tokenMasks: [BatchRequestID: any ContinuousBatchRowTokenMask] = [:]

    init(
        scriptsByPromptHead: [Int: [Int]],
        recorder: ContinuousRuntimeRecorder,
        allowsSpeculation: Bool = false,
        speculativeSoloState: BatchSoloPipelineState = .speculative,
        resourceSnapshotGate: BlockingRuntimeResourceSnapshotGate? = nil,
        advancesTokenMasks: Bool = false
    ) {
        self.scriptsByPromptHead = scriptsByPromptHead
        self.recorder = recorder
        self.allowsSpeculation = allowsSpeculation
        self.speculativeSoloState = speculativeSoloState
        self.resourceSnapshotGate = resourceSnapshotGate
        self.advancesTokenMasks = advancesTokenMasks
    }

    func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {
        admittedIDs.formUnion(admissions.map(\.id))
        for admission in admissions {
            tokenMasks[admission.id] = admission.submission.tokenMask
        }
        recorder.admitted(admissions.count)
        recorder.admitted(admissions)
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        resourceSnapshotGate?.wait()
        return recorder.resources
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {
        var slot: Slot
        if let existing = slots[work.id] {
            slot = existing
        } else {
            guard work.startToken == 0,
                let head = work.tokens.first,
                let script = scriptsByPromptHead[head]
            else {
                throw FixtureContinuousRuntimeError.invalidPrefill
            }
            slot = Slot(script: script)
            promptHeadByID[work.id] = head
        }
        guard work.startToken == slot.processedTokens else {
            throw FixtureContinuousRuntimeError.invalidPrefill
        }
        slot.processedTokens += work.tokens.count
        slot.ready = work.isFinal
        slots[work.id] = slot
    }

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        recorder.record(action)
        let ids: [BatchRequestID]
        let stateAfter: BatchSoloPipelineState
        let outputlessSpeculativeDrain: Bool
        switch action {
        case .solo(let id, let speculationAllowed):
            guard !speculationAllowed || allowsSpeculation else {
                throw FixtureContinuousRuntimeError.speculation
            }
            ids = [id]
            stateAfter = speculationAllowed
                ? speculativeSoloState
                : .pipelinedLookahead
            outputlessSpeculativeDrain = false
        case .drainSoloPipeline(let id):
            guard slots[id]?.soloPipelineState.requiresDrain == true else {
                throw FixtureContinuousRuntimeError.invalidDrain
            }
            ids = [id]
            stateAfter = .canonical
            outputlessSpeculativeDrain =
                slots[id]?.soloPipelineState == .speculative
        case .batch(let batchIDs, let speculationAllowed):
            guard !speculationAllowed || allowsSpeculation else {
                throw FixtureContinuousRuntimeError.speculation
            }
            guard batchIDs.allSatisfy({
                slots[$0]?.soloPipelineState == .canonical
            }) else {
                throw FixtureContinuousRuntimeError.invalidBatch
            }
            ids = batchIDs
            stateAfter = .canonical
            outputlessSpeculativeDrain = false
        }

        return try ids.map { id in
            guard var slot = slots[id], slot.ready else {
                throw FixtureContinuousRuntimeError.decodeBeforeReady
            }
            if outputlessSpeculativeDrain {
                slot.soloPipelineState = .canonical
                slots[id] = slot
                return ContinuousBatchRuntimeDecodeResult(
                    id: id,
                    tokens: [],
                    finished: false,
                    soloPipelineState: .canonical)
            }
            let tokens: [Int]
            let finished: Bool
            if slot.cursor < slot.script.count {
                tokens = [slot.script[slot.cursor]]
                slot.cursor += 1
                finished = false
                if advancesTokenMasks, let mask = tokenMasks[id] {
                    mask.advance(token: tokens[0])
                    _ = mask.allowedTokenBitset()
                }
            } else {
                tokens = []
                finished = true
            }
            slot.soloPipelineState = stateAfter
            slots[id] = slot
            return ContinuousBatchRuntimeDecodeResult(
                id: id,
                tokens: tokens,
                finished: finished,
                soloPipelineState: stateAfter)
        }
    }

    func remove(_ id: BatchRequestID) {
        slots[id] = nil
        if admittedIDs.remove(id) != nil {
            recorder.removed()
        }
        promptHeadByID[id] = nil
        tokenMasks[id] = nil
    }
}

private final class AdmissionFailureContinuousRuntime:
    ContinuousBatchRuntime
{
    private let failureOnAdmission: Int
    private let error: DenseContinuousBatchRuntimeError
    private var admissionCount = 0

    init(
        failureOnAdmission: Int,
        error: DenseContinuousBatchRuntimeError
    ) {
        self.failureOnAdmission = failureOnAdmission
        self.error = error
    }

    func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {
        admissionCount += 1
        if admissionCount == failureOnAdmission {
            throw error
        }
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: 64,
            reservedKVBytes: 0,
            maxReservedKVBytes: 4_096)
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {}

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        []
    }

    func remove(_ id: BatchRequestID) {}
}

private final class AdmissionValidationFailureContinuousRuntime:
    ContinuousBatchRuntime
{
    private let error: DenseContinuousBatchRuntimeError

    init(error: DenseContinuousBatchRuntimeError) {
        self.error = error
    }

    func decodeCohort(
        for admission: ContinuousBatchRuntimeAdmission
    ) throws -> BatchDecodeCohort {
        throw error
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: 64,
            reservedKVBytes: 0,
            maxReservedKVBytes: 4_096)
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {}

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        []
    }

    func remove(_ id: BatchRequestID) {}
}

private final class CountingValidationContinuousRuntime:
    ContinuousBatchRuntime, @unchecked Sendable
{
    private struct State: Sendable {
        var validationCount = 0
        var admissionBatchCount = 0
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    var validationCount: Int {
        state.withLock(\.validationCount)
    }

    var admissionBatchCount: Int {
        state.withLock(\.admissionBatchCount)
    }

    func decodeCohort(
        for admission: ContinuousBatchRuntimeAdmission
    ) throws -> BatchDecodeCohort {
        state.withLock { $0.validationCount += 1 }
        return .unrestricted
    }

    func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {
        state.withLock { $0.admissionBatchCount += 1 }
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: 64,
            reservedKVBytes: 0,
            maxReservedKVBytes: 4_096)
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {}

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        []
    }

    func remove(_ id: BatchRequestID) {}
}

private enum FixtureContinuousRuntimeError: Error {
    case invalidPrefill
    case decodeBeforeReady
    case invalidDrain
    case invalidBatch
    case speculation
}

private struct FixtureContinuousTextCodec: ScalarServingTextCodec {
    let promptByText: [String: [Int]]
    let pieces: [Int: String]
    let renderGate: BlockingRuntimeResourceSnapshotGate?

    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?,
        addGenerationPrompt: Bool?
    ) throws -> [Int] {
        renderGate?.wait()
        guard let text = messages.last?.text,
            let prompt = promptByText[text]
        else {
            throw FixtureContinuousRuntimeError.invalidPrefill
        }
        return prompt
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        FixtureContinuousDetokenizer(pieces: pieces)
    }
}

private struct FixtureContinuousDetokenizer: ScalarServingDetokenizer {
    let pieces: [Int: String]
    private var pending: String?

    init(pieces: [Int: String]) {
        self.pieces = pieces
    }

    mutating func append(token: Int) {
        pending = pieces[token]
    }

    mutating func next() -> String? {
        defer { pending = nil }
        return pending
    }
}

private func configuration(
    active: Int,
    queued: Int
) throws -> ContinuousBatchConfiguration {
    try ContinuousBatchConfiguration(
        maxActiveSlots: active,
        maxPrefillSlots: active,
        prefillChunkSize: 8,
        maxQueuedRequests: queued)
}

private func makeBackend(
    coordinator: ContinuousBatchCoordinator,
    promptByText: [String: [Int]],
    pieces: [Int: String],
    stopTokenIDs: Set<Int>,
    mailboxCapacity: BoundedDeltaMailbox.Capacity = .init(
        maxDeltas: 2,
        maxBytes: 4_096),
    modelCapabilities: ServingModelCapabilities? = nil,
    thinksByDefault: Bool = false,
    logitBiasVocabularySize: Int? = nil,
    jsonObjectConstraintSupport: ScalarServingJSONObjectConstraintSupport? = nil,
    jsonSchemaConstraintSupport: ScalarServingJSONSchemaConstraintSupport? = nil
) -> ContinuousServingBackend {
    ContinuousServingBackend(
        launchedModel: "fixture",
        coordinator: coordinator,
        codec: FixtureContinuousTextCodec(
            promptByText: promptByText,
            pieces: pieces,
            renderGate: nil),
        stopTokenIDs: stopTokenIDs,
        modelStopStrings: [],
        configuration: ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 8,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: mailboxCapacity,
            thinksByDefault: thinksByDefault,
            modelCapabilities: modelCapabilities,
            logitBiasVocabularySize: logitBiasVocabularySize,
            jsonObjectConstraintSupport: jsonObjectConstraintSupport,
            jsonSchemaConstraintSupport: jsonSchemaConstraintSupport))
}

private func makeDynamicBackend(
    coordinator: ContinuousBatchCoordinator,
    promptByText: [String: [Int]],
    pieces: [Int: String],
    stopTokenIDs: Set<Int>,
    maximumBatchRequests: Int = 2,
    maximumQueuedRequests: Int = 4,
    coalescing: ContinuousServingAdmissionCoalescing = .manualDiagnostic,
    mailboxCapacity: BoundedDeltaMailbox.Capacity = .init(
        maxDeltas: 2,
        maxBytes: 4_096),
    renderGate: BlockingRuntimeResourceSnapshotGate? = nil,
    modelCapabilities: ServingModelCapabilities? = nil,
    logitBiasVocabularySize: Int? = nil,
    jsonObjectConstraintSupport: ScalarServingJSONObjectConstraintSupport? = nil,
    jsonSchemaConstraintSupport: ScalarServingJSONSchemaConstraintSupport? = nil
) -> ContinuousServingBackend {
    ContinuousServingBackend(
        launchedModel: "fixture",
        coordinator: coordinator,
        codec: FixtureContinuousTextCodec(
            promptByText: promptByText,
            pieces: pieces,
            renderGate: renderGate),
        stopTokenIDs: stopTokenIDs,
        modelStopStrings: [],
        configuration: ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 8,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: mailboxCapacity,
            admission: .dynamic(
                configuration: ServingAdmissionConfiguration(
                    soloPLDQualified: true,
                    maximumBatchRequests: maximumBatchRequests,
                    maximumQueuedRequests: maximumQueuedRequests),
                coalescing: coalescing),
            modelCapabilities: modelCapabilities,
            logitBiasVocabularySize: logitBiasVocabularySize,
            jsonObjectConstraintSupport: jsonObjectConstraintSupport,
            jsonSchemaConstraintSupport: jsonSchemaConstraintSupport))
}

private let continuousWeatherTool = OpenAIToolSpec(
    name: "get_weather",
    description: "Look up the weather for a city",
    parameters: .object([
        "type": .string("object"),
        "properties": .object([
            "city": .object(["type": .string("string")])
        ]),
    ]),
    raw: .object([
        "type": .string("function"),
        "function": .object([
            "name": .string("get_weather"),
            "description": .string("Look up the weather for a city"),
            "parameters": .object([
                "type": .string("object"),
                "properties": .object([
                    "city": .object(["type": .string("string")])
                ]),
            ]),
        ]),
    ]))

private func request(
    text: String,
    maxTokens: Int,
    stop: [String] = [],
    tools: [OpenAIToolSpec] = [],
    parallelToolCalls: Bool? = nil,
    enableThinking: Bool? = nil
) -> OpenAIChatCompletionRequest {
    OpenAIChatCompletionRequest(
        model: "fixture",
        messages: [
            OpenAIChatMessage(role: .user, text: text),
        ],
        maxCompletionTokens: maxTokens,
        temperature: 0,
        choiceCount: 1,
        stream: true,
        stop: stop,
        tools: tools,
        toolChoice: tools.isEmpty ? .none : .auto,
        parallelToolCalls: parallelToolCalls,
        enableThinking: enableThinking)
}

private func collect(
    _ mailbox: BoundedDeltaMailbox
) async throws -> [ServingResponseDelta] {
    var events: [ServingResponseDelta] = []
    while let event = try await mailbox.next() {
        events.append(event)
    }
    return events
}

private func assertMailboxCancelled(
    _ mailbox: BoundedDeltaMailbox,
    reason: ServingCancellationReason,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        _ = try await mailbox.next()
        XCTFail("Expected mailbox cancellation", file: file, line: line)
    } catch let error as ServingMailboxError {
        XCTAssertEqual(
            error,
            .cancelled(reason),
            file: file,
            line: line)
    } catch {
        XCTFail(
            "Unexpected mailbox error: \(error)",
            file: file,
            line: line)
    }
}

private func runUntilDecode(
    _ coordinator: ContinuousBatchCoordinator,
    recorder: ContinuousRuntimeRecorder,
    file: StaticString = #filePath,
    line: UInt = #line
) async throws {
    for _ in 0 ..< 100 {
        if !recorder.decodeActions.isEmpty {
            return
        }
        let advanced = try await coordinator.runOneTick()
        if !advanced {
            break
        }
    }
    XCTFail("Coordinator did not reach decode", file: file, line: line)
}

private func waitUntil(
    attempts: Int = 10_000,
    _ predicate: () async -> Bool
) async {
    for _ in 0 ..< attempts {
        if await predicate() {
            return
        }
        await Task.yield()
    }
    XCTFail("Condition was not reached")
}


// MARK: - Constraint-support fixtures (file-private; shared by the tests above)

private let continuousEOSID = 256
private let continuousThinkEndID = 257

/// Byte vocabulary: ids 0...255 raw bytes, 256 EOS, 257 `</think>` (banned for JSON). Built the way
/// production does: an object support, then the schema support derived from it.
private func continuousObjectSupport(
    thinkEndTokenID: Int? = nil
) -> ScalarServingJSONObjectConstraintSupport {
    var classifications: [TokenByteClassification] = (0...255).map { .bytes([UInt8($0)]) }
    classifications.append(.eos)
    classifications.append(.banned)
    return ScalarServingJSONObjectConstraintSupport(
        classifications: classifications, thinkEndTokenID: thinkEndTokenID)
}

private func continuousSchemaSupport(
    thinkEndTokenID: Int? = nil
) -> ScalarServingJSONSchemaConstraintSupport {
    loadScalarServingJSONSchemaConstraintSupport(
        sharing: continuousObjectSupport(thinkEndTokenID: thinkEndTokenID))!
}

private func booleanSchemaFormat() throws -> JSONSchemaResponseFormat {
    let raw = try JSONSchemaSubsetCompiler.parseOrdered(Data(#"{"type":"boolean"}"#.utf8))
    return try JSONSchemaSubsetCompiler.compile(name: "flag", schema: raw, strict: true)
}

private func continuousByteIDs(_ text: String) -> [Int] { text.utf8.map { Int($0) } }

private func continuousBytePieces() -> [Int: String] {
    var pieces: [Int: String] = [:]
    for id in 0..<128 { pieces[id] = String(UnicodeScalar(UInt8(id))) }
    pieces[continuousThinkEndID] = "</think>"
    return pieces
}

private func continuousBit(_ id: Int, in bitset: [UInt64]) -> Bool {
    let word = id >> 6
    guard word < bitset.count else { return false }
    return bitset[word] & (1 << UInt64(id & 63)) != 0
}
