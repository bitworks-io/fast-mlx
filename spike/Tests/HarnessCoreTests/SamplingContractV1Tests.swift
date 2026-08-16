import XCTest

@testable import HarnessCore

final class SamplingContractV1Tests: XCTestCase {
    func testContractConstantsStoredFieldsAndConformancesAreExact() throws {
        XCTAssertEqual(
            SamplingContractV1.contractVersionTag,
            "fast-mlx-sampling-contract-v1")
        XCTAssertEqual(
            SamplingContractV1.randomWordPreimagePrefix,
            "fast-mlx-sampling-rng-v1\n")
        XCTAssertEqual(SamplingContractV1.tokenSelectionDomain, 0)
        XCTAssertEqual(SamplingContractV1.maximumVocabularyCount, 262_144)
        XCTAssertEqual(
            SamplingContractV1.uniformDenominator,
            4_503_599_627_370_496.0)

        let seed = SamplingSeedV1.callerSupplied(42)
        let greedy = SamplingPolicyV1.greedy()
        let sampled = try SamplingPolicyV1.sampled(seed: seed)
        XCTAssertEqual(sampled.temperature.bitPattern, 1.0.bitPattern)
        XCTAssertEqual(sampled.topP.bitPattern, 1.0.bitPattern)
        XCTAssertEqual(sampled.seed, seed)
        let address = try SamplingDrawAddressV1(
            seed: seed,
            committedSampleOrdinal: 0)
        let selection: SamplingSelectionV1 = try SamplingRank1OracleV1.selectToken(
            logits: [0, 0],
            policy: sampled)
        let failure = SamplingContractFailure.nonFiniteTemperature(.nan)
        let kind = SamplingContractFailure.NonFiniteKind.negativeInfinity

        assertEquatableAndSendable(seed)
        assertEquatableAndSendable(greedy)
        assertEquatableAndSendable(sampled)
        assertEquatableAndSendable(address)
        assertEquatableAndSendable(selection)
        assertFailureConforms(failure)
        assertEquatableAndSendable(kind)

        XCTAssertEqual(storedLabels(seed), ["resolvedSeedBitPattern"])
        XCTAssertEqual(
            storedLabels(sampled),
            ["contractVersionTag", "temperature", "topP", "seed"])
        XCTAssertEqual(
            storedLabels(address),
            [
                "contractVersionTag",
                "seedBitPattern",
                "domain",
                "committedSampleOrdinal",
            ])
        XCTAssertEqual(
            storedLabels(selection),
            ["tokenID", "drawAddress", "randomWord"])
    }

    func testCallerSuppliedSeedBitPatternsAreExact() {
        let fixtures: [(Int64, UInt64)] = [
            (Int64.min, 0x8000000000000000),
            (-1, 0xffffffffffffffff),
            (0, 0x0000000000000000),
            (1, 0x0000000000000001),
            (Int64.max, 0x7fffffffffffffff),
        ]

        for fixture in fixtures {
            let seed = SamplingSeedV1.callerSupplied(fixture.0)
            XCTAssertEqual(seed.resolvedSeedBitPattern, fixture.1)
        }
    }

    func testPolicyFactoriesCanonicalizeAndRejectInReviewedOrder() throws {
        let seed = SamplingSeedV1.callerSupplied(7)
        let greedy = SamplingPolicyV1.greedy()

        XCTAssertEqual(
            greedy.contractVersionTag,
            SamplingContractV1.contractVersionTag)
        XCTAssertEqual(greedy.temperature.bitPattern, 0x0000000000000000)
        XCTAssertEqual(greedy.topP, 1)
        XCTAssertNil(greedy.seed)

        assertThrowsSamplingFailure(
            try SamplingPolicyV1.sampled(
                temperature: .nan,
                topP: 1.0.nextUp,
                seed: seed),
            .nonFiniteTemperature(.nan))
        assertThrowsSamplingFailure(
            try SamplingPolicyV1.sampled(
                temperature: .infinity,
                topP: 1,
                seed: seed),
            .nonFiniteTemperature(.positiveInfinity))
        assertThrowsSamplingFailure(
            try SamplingPolicyV1.sampled(
                temperature: -.infinity,
                topP: .nan,
                seed: seed),
            .nonFiniteTemperature(.negativeInfinity))
        assertThrowsSamplingFailure(
            try SamplingPolicyV1.sampled(
                temperature: -Double.leastNonzeroMagnitude,
                topP: .nan,
                seed: seed),
            .temperatureOutOfRange(-Double.leastNonzeroMagnitude))
        assertThrowsSamplingFailure(
            try SamplingPolicyV1.sampled(
                temperature: 1,
                topP: .nan,
                seed: seed),
            .nonFiniteTopP(.nan))
        assertThrowsSamplingFailure(
            try SamplingPolicyV1.sampled(
                temperature: 1,
                topP: .infinity,
                seed: seed),
            .nonFiniteTopP(.positiveInfinity))
        assertThrowsSamplingFailure(
            try SamplingPolicyV1.sampled(
                temperature: 1,
                topP: -.infinity,
                seed: seed),
            .nonFiniteTopP(.negativeInfinity))
        assertThrowsSamplingFailure(
            try SamplingPolicyV1.sampled(
                temperature: 1,
                topP: -Double.leastNonzeroMagnitude,
                seed: seed),
            .topPOutOfRange(-Double.leastNonzeroMagnitude))
        assertThrowsSamplingFailure(
            try SamplingPolicyV1.sampled(
                temperature: 0,
                topP: 1.0.nextUp,
                seed: seed),
            .topPOutOfRange(1.0.nextUp))
        assertThrowsSamplingFailure(
            try SamplingPolicyV1.sampled(
                temperature: -0.0,
                topP: 1,
                seed: seed),
            .zeroTemperatureSamplingConflict)

        let negativeZeroTopP = try SamplingPolicyV1.sampled(
            temperature: 1,
            topP: -0.0,
            seed: seed)
        XCTAssertEqual(negativeZeroTopP.temperature, 1)
        XCTAssertEqual(negativeZeroTopP.topP.bitPattern, 0x0000000000000000)
        XCTAssertEqual(negativeZeroTopP.seed, seed)
        XCTAssertEqual(
            negativeZeroTopP.contractVersionTag,
            SamplingContractV1.contractVersionTag)

        let leastPositiveTemperature = try SamplingPolicyV1.sampled(
            temperature: Double.leastNonzeroMagnitude,
            topP: 0,
            seed: seed)
        XCTAssertEqual(
            leastPositiveTemperature.temperature,
            Double.leastNonzeroMagnitude)
        XCTAssertEqual(leastPositiveTemperature.topP, 0)

        let upperBoundary = try SamplingPolicyV1.sampled(
            temperature: 2,
            topP: 1,
            seed: seed)
        XCTAssertEqual(upperBoundary.temperature, 2)
        XCTAssertEqual(upperBoundary.topP, 1)

        assertThrowsSamplingFailure(
            try SamplingPolicyV1.sampled(
                temperature: 2.0.nextUp,
                topP: 1,
                seed: seed),
            .temperatureOutOfRange(2.0.nextUp))
    }

    func testDrawAddressUsesExactBigEndianSHA256VectorsAndMidpointUniforms()
        throws
    {
        let vectors: [DrawVector] = [
            DrawVector(
                seed: Int64.min,
                ordinal: 0,
                word: 0x833d0a8edd86fcdf,
                uniformBits: 0x3fe067a151dbb0df),
            DrawVector(
                seed: -1,
                ordinal: 0,
                word: 0xcbbfeebee316ff5d,
                uniformBits: 0x3fe977fdd7dc62df),
            DrawVector(
                seed: 0,
                ordinal: 0,
                word: 0x87d589ecbc029492,
                uniformBits: 0x3fe0fab13d978053),
            DrawVector(
                seed: 1,
                ordinal: 0,
                word: 0x62841adaa918f0cf,
                uniformBits: 0x3fd8a106b6aa463e),
            DrawVector(
                seed: Int64.max,
                ordinal: 0,
                word: 0xef908668462b457f,
                uniformBits: 0x3fedf210cd08c569),
            DrawVector(
                seed: 0,
                ordinal: 1,
                word: 0x7fad612659fa1d08,
                uniformBits: 0x3fdfeb5849967e86),
            DrawVector(
                seed: 0,
                ordinal: UInt64.max - 1,
                word: 0x58f1d183d20d7453,
                uniformBits: 0x3fd63c7460f4835e),
            DrawVector(
                seed: 0,
                ordinal: UInt64.max,
                word: 0xc627fc7fdd040747,
                uniformBits: 0x3fe8c4ff8ffba081),
        ]

        for vector in vectors {
            let seed = SamplingSeedV1.callerSupplied(vector.seed)
            let address = try SamplingDrawAddressV1(
                seed: seed,
                committedSampleOrdinal: vector.ordinal)

            XCTAssertEqual(
                address.contractVersionTag,
                SamplingContractV1.contractVersionTag)
            XCTAssertEqual(
                address.seedBitPattern,
                seed.resolvedSeedBitPattern)
            XCTAssertEqual(
                address.domain,
                SamplingContractV1.tokenSelectionDomain)
            XCTAssertEqual(address.committedSampleOrdinal, vector.ordinal)
            XCTAssertEqual(address.randomWord(), vector.word)
            XCTAssertEqual(
                address.uniformOpenInterval().bitPattern,
                vector.uniformBits)
            XCTAssertGreaterThan(address.uniformOpenInterval(), 0)
            XCTAssertLessThan(address.uniformOpenInterval(), 1)
            XCTAssertEqual(
                Double(uniformNumerator(from: vector.word))
                    / 9_007_199_254_740_992.0,
                address.uniformOpenInterval())
        }
    }

    func testRetryAdvanceMaximumOrdinalAndReservedDomainAreExact() throws {
        let seed = SamplingSeedV1.callerSupplied(0)
        let address = try SamplingDrawAddressV1(
            seed: seed,
            committedSampleOrdinal: 41)
        let advanced = try address.afterCommittedSelection()

        XCTAssertEqual(address.committedSampleOrdinal, 41)
        XCTAssertEqual(advanced.committedSampleOrdinal, 42)
        XCTAssertEqual(advanced.contractVersionTag, address.contractVersionTag)
        XCTAssertEqual(advanced.seedBitPattern, address.seedBitPattern)
        XCTAssertEqual(advanced.domain, address.domain)
        XCTAssertNotEqual(advanced.randomWord(), address.randomWord())

        assertThrowsSamplingFailure(
            try SamplingDrawAddressV1(
                seed: seed,
                domain: 1,
                committedSampleOrdinal: UInt64.max),
            .unsupportedDrawDomain(1))

        let maximum = try SamplingDrawAddressV1(
            seed: seed,
            committedSampleOrdinal: UInt64.max)
        let beforeMaximum = try SamplingDrawAddressV1(
            seed: seed,
            committedSampleOrdinal: UInt64.max - 1)
        let advancedToMaximum = try beforeMaximum.afterCommittedSelection()

        XCTAssertEqual(beforeMaximum.committedSampleOrdinal, UInt64.max - 1)
        XCTAssertEqual(beforeMaximum.randomWord(), 0x58f1d183d20d7453)
        XCTAssertEqual(advancedToMaximum, maximum)
        XCTAssertEqual(advancedToMaximum.randomWord(), 0xc627fc7fdd040747)
        XCTAssertEqual(beforeMaximum.randomWord(), 0x58f1d183d20d7453)
        XCTAssertEqual(maximum.randomWord(), 0xc627fc7fdd040747)
        XCTAssertEqual(maximum.randomWord(), maximum.randomWord())
        XCTAssertEqual(maximum.uniformOpenInterval().bitPattern, 0x3fe8c4ff8ffba081)

        let policy = try SamplingPolicyV1.sampled(seed: seed)
        let firstSelection = try SamplingRank1OracleV1.selectToken(
            logits: [0, 0],
            policy: policy,
            committedSampleOrdinal: UInt64.max)
        let secondSelection = try SamplingRank1OracleV1.selectToken(
            logits: [0, 0],
            policy: policy,
            committedSampleOrdinal: UInt64.max)

        XCTAssertEqual(firstSelection, secondSelection)
        XCTAssertEqual(firstSelection.tokenID, 1)
        XCTAssertEqual(firstSelection.drawAddress, .some(maximum))
        XCTAssertEqual(firstSelection.randomWord, .some(maximum.randomWord()))

        assertThrowsSamplingFailure(
            try maximum.afterCommittedSelection(),
            .committedSampleOrdinalOverflow)
        XCTAssertEqual(maximum.randomWord(), 0xc627fc7fdd040747)
        XCTAssertEqual(beforeMaximum.randomWord(), 0x58f1d183d20d7453)
    }

    func testCompensatedSumUsesFrozenSequentialNeumaierOrder() {
        let empty = SamplingRank1OracleV1.compensatedFiniteSum([])
        XCTAssertEqual(empty, 0)
        XCTAssertEqual(empty.bitPattern, 0x0000000000000000)

        let discriminator = [
            Double(sign: .plus, exponent: -54, significand: 1),
            1.0,
            Double(sign: .plus, exponent: -53, significand: 1),
        ]
        XCTAssertEqual(
            SamplingRank1OracleV1.compensatedFiniteSum(discriminator)
                .bitPattern,
            0x3ff0000000000001)

        XCTAssertEqual(
            SamplingRank1OracleV1.compensatedFiniteSum([0.25, 0.5, 0.25]),
            1)
        XCTAssertEqual(
            SamplingRank1OracleV1.compensatedFiniteSum([1, 0, 0, 0]),
            1)
        XCTAssertEqual(
            SamplingRank1OracleV1.compensatedFiniteSum([0.5, 0.25, 0.25]),
            1)
    }

    func testGreedyValidatesFiniteLogitsBreaksTiesAndConsumesNoDraw() throws {
        let greedy = SamplingPolicyV1.greedy()

        assertThrowsSamplingFailure(
            try SamplingRank1OracleV1.selectToken(
                logits: [10, .nan, 11],
                policy: greedy,
                committedSampleOrdinal: UInt64.max),
            .nonFiniteLogit(tokenID: 1, kind: .nan))

        let selection = try SamplingRank1OracleV1.selectToken(
            logits: [-5, 7, 7, 6],
            policy: greedy,
            committedSampleOrdinal: UInt64.max)

        XCTAssertEqual(selection.tokenID, 1)
        XCTAssertNil(selection.drawAddress)
        XCTAssertNil(selection.randomWord)
    }

    func testSampledArithmeticOrderingTopPAndCategoricalTraversalAreExact()
        throws
    {
        try assertSampledFixture(
            logits: [0, 1, 1, 0],
            temperature: 1,
            topP: 0,
            seed: 0,
            ordinal: 7,
            expectedToken: 1,
            expectedWord: 0x4c5c42358a420440)
        try assertSampledFixture(
            logits: [
                1.3862943611198906,
                0.6931471805599453,
                0,
            ],
            temperature: 2,
            topP: 0.5,
            seed: 3,
            ordinal: 0,
            expectedToken: 1,
            expectedWord: 0xb53da7f5cdee271f)
        try assertSampledFixture(
            logits: [
                Double.greatestFiniteMagnitude,
                Double.greatestFiniteMagnitude,
            ],
            temperature: 0.5,
            topP: 1,
            seed: 1,
            ordinal: 0,
            expectedToken: 0,
            expectedWord: 0x62841adaa918f0cf)
        try assertSampledFixture(
            logits: [
                Double.greatestFiniteMagnitude,
                -Double.greatestFiniteMagnitude,
            ],
            temperature: 0.5,
            topP: 1,
            seed: -1,
            ordinal: 0,
            expectedToken: 0,
            expectedWord: 0xcbbfeebee316ff5d)
        try assertSampledFixture(
            logits: [0, 0, 0, 0],
            temperature: 1,
            topP: 0.5,
            seed: -1,
            ordinal: 0,
            expectedToken: 1,
            expectedWord: 0xcbbfeebee316ff5d)
        try assertSampledFixture(
            logits: [0, 0, 0, 0],
            temperature: 1,
            topP: 0.5,
            seed: 1,
            ordinal: 0,
            expectedToken: 0,
            expectedWord: 0x62841adaa918f0cf)
        try assertSampledFixture(
            logits: [0, -10, 1],
            temperature: 1,
            topP: 0.9,
            seed: 42,
            ordinal: 0,
            expectedToken: 0,
            expectedWord: 0x39783ce23f267f72)

        let h1Logits = [
            Double(bitPattern: 0xbff3d5bf42474569),
            0.0,
        ]
        let h1Weight = exp(h1Logits[0])
        let h1Total = SamplingRank1OracleV1.compensatedFiniteSum([h1Weight, 1])
        let h1Address = try SamplingDrawAddressV1(
            seed: SamplingSeedV1.callerSupplied(42),
            committedSampleOrdinal: 0)
        XCTAssertEqual(h1Logits[0].bitPattern, 0xbff3d5bf42474569)
        XCTAssertEqual(h1Weight.bitPattern, 0x3fd286c490ee6d20)
        XCTAssertEqual(h1Total.bitPattern, 0x3ff4a1b1243b9b48)
        XCTAssertEqual(h1Address.uniformOpenInterval().bitPattern, 0x3fccbc1e711f933c)
        XCTAssertEqual(
            (h1Address.uniformOpenInterval() * h1Total).bitPattern,
            h1Weight.bitPattern)
        try assertSampledFixture(
            logits: h1Logits,
            temperature: 1,
            topP: 1,
            seed: 42,
            ordinal: 0,
            expectedToken: 1,
            expectedWord: 0x39783ce23f267f72)

        let topPZero = try SamplingRank1OracleV1.selectToken(
            logits: [3, 2, 1],
            policy: try SamplingPolicyV1.sampled(
                temperature: 1,
                topP: 0,
                seed: SamplingSeedV1.callerSupplied(99)))
        let topPOne = try SamplingRank1OracleV1.selectToken(
            logits: [3, 2, 1],
            policy: try SamplingPolicyV1.sampled(
                temperature: 1,
                topP: 1,
                seed: SamplingSeedV1.callerSupplied(99)))
        XCTAssertNotNil(topPZero.drawAddress)
        XCTAssertNotNil(topPZero.randomWord)
        XCTAssertNotNil(topPOne.drawAddress)
        XCTAssertNotNil(topPOne.randomWord)
    }

    func testInverseCDFUsesHalfOpenBinsAndStrictThreshold() throws {
        let fixtures: [BoundaryFixture] = [
            BoundaryFixture(
                seed: 26_142,
                word: 0x3ffecfb17fbabbd5,
                numerator: 2_251_636_440_299_351,
                expectedToken: 0),
            BoundaryFixture(
                seed: 31_878,
                word: 0x400014b29f132487,
                numerator: 2_251_810_925_699_685,
                expectedToken: 1),
            BoundaryFixture(
                seed: 11_568,
                word: 0x7fffe88f44218e1b,
                numerator: 4_503_587_042_919_473,
                expectedToken: 1),
            BoundaryFixture(
                seed: 8_289,
                word: 0x80009a15c1cd02af,
                numerator: 4_503_682_351_118_753,
                expectedToken: 2),
            BoundaryFixture(
                seed: 49_568,
                word: 0xbfff1fe8ad6ed03e,
                numerator: 6_755_279_133_060_571,
                expectedToken: 2),
            BoundaryFixture(
                seed: 25_319,
                word: 0xc0003095bdf55cb5,
                numerator: 6_755_425_524_891_307,
                expectedToken: 3),
        ]

        for fixture in fixtures {
            let seed = SamplingSeedV1.callerSupplied(fixture.seed)
            let address = try SamplingDrawAddressV1(
                seed: seed,
                committedSampleOrdinal: 0)
            let selection = try SamplingRank1OracleV1.selectToken(
                logits: [0, 0, 0, 0],
                policy: try SamplingPolicyV1.sampled(seed: seed))

            XCTAssertEqual(address.randomWord(), fixture.word)
            XCTAssertEqual(
                uniformNumerator(from: address.randomWord()),
                fixture.numerator)
            XCTAssertEqual(selection.tokenID, fixture.expectedToken)
            XCTAssertEqual(selection.drawAddress, .some(address))
            XCTAssertEqual(selection.randomWord, .some(fixture.word))
        }
    }

    func testOracleRefusalOrderVocabularyBoundaryAndNonFiniteKindsAreExact()
        throws
    {
        let greedy = SamplingPolicyV1.greedy()
        let sampled = try SamplingPolicyV1.sampled(
            seed: SamplingSeedV1.callerSupplied(0))

        assertThrowsSamplingFailure(
            try SamplingRank1OracleV1.selectToken(
                logits: [],
                policy: greedy),
            .emptyLogits)
        assertThrowsSamplingFailure(
            try SamplingRank1OracleV1.selectToken(
                logits: [],
                policy: sampled),
            .emptyLogits)

        let greedySingleton = try SamplingRank1OracleV1.selectToken(
            logits: [7],
            policy: greedy,
            committedSampleOrdinal: UInt64.max)
        let sampledSingleton = try SamplingRank1OracleV1.selectToken(
            logits: [7],
            policy: sampled)
        XCTAssertEqual(greedySingleton.tokenID, 0)
        XCTAssertNil(greedySingleton.drawAddress)
        XCTAssertNil(greedySingleton.randomWord)
        XCTAssertEqual(sampledSingleton.tokenID, 0)
        XCTAssertNotNil(sampledSingleton.drawAddress)
        XCTAssertNotNil(sampledSingleton.randomWord)

        var oversized = Array(
            repeating: 0.0,
            count: SamplingContractV1.maximumVocabularyCount + 1)
        oversized[0] = .nan
        assertThrowsSamplingFailure(
            try SamplingRank1OracleV1.selectToken(
                logits: oversized,
                policy: sampled),
            .vocabularyCountExceedsMaximum(
                actual: SamplingContractV1.maximumVocabularyCount + 1,
                maximum: SamplingContractV1.maximumVocabularyCount))

        let maximumVocabulary = Array(
            repeating: 0.0,
            count: SamplingContractV1.maximumVocabularyCount)
        let maximumSelection = try SamplingRank1OracleV1.selectToken(
            logits: maximumVocabulary,
            policy: greedy)
        XCTAssertEqual(maximumSelection.tokenID, 0)
        XCTAssertNil(maximumSelection.drawAddress)
        XCTAssertNil(maximumSelection.randomWord)

        assertThrowsSamplingFailure(
            try SamplingRank1OracleV1.selectToken(
                logits: [.nan, .infinity, -.infinity],
                policy: sampled),
            .nonFiniteLogit(tokenID: 0, kind: .nan))
        assertThrowsSamplingFailure(
            try SamplingRank1OracleV1.selectToken(
                logits: [0, .infinity, .nan, -.infinity],
                policy: sampled),
            .nonFiniteLogit(tokenID: 1, kind: .positiveInfinity))
        assertThrowsSamplingFailure(
            try SamplingRank1OracleV1.selectToken(
                logits: [0, -.infinity, .infinity, .nan],
                policy: sampled),
            .nonFiniteLogit(tokenID: 1, kind: .negativeInfinity))
        assertThrowsSamplingFailure(
            try SamplingRank1OracleV1.selectToken(
                logits: [1, -.infinity, 2],
                policy: greedy),
            .nonFiniteLogit(tokenID: 1, kind: .negativeInfinity))
    }

    func testDefensiveInvariantFailuresRemainTypedWithoutFaultInjection()
        throws
    {
        XCTAssertEqual(
            SamplingContractFailure.invalidFullProbabilityMass,
            .invalidFullProbabilityMass)
        XCTAssertEqual(
            SamplingContractFailure.invalidRetainedProbabilityMass,
            .invalidRetainedProbabilityMass)
        XCTAssertEqual(
            SamplingContractFailure.invalidSelectedTokenIndex(-1),
            .invalidSelectedTokenIndex(-1))
        XCTAssertNotEqual(
            SamplingContractFailure.invalidSelectedTokenIndex(-1),
            .invalidSelectedTokenIndex(0))

        let seed = SamplingSeedV1.callerSupplied(1)
        let sampled = try SamplingPolicyV1.sampled(
            temperature: 0.5,
            topP: 1,
            seed: seed)
        let maximumSubtraction = try SamplingRank1OracleV1.selectToken(
            logits: [
                Double.greatestFiniteMagnitude,
                Double.greatestFiniteMagnitude,
            ],
            policy: sampled)
        let finiteUnderflowToZeroMass = try SamplingRank1OracleV1.selectToken(
            logits: [
                Double.greatestFiniteMagnitude,
                -Double.greatestFiniteMagnitude,
            ],
            policy: sampled)

        XCTAssertEqual(maximumSubtraction.tokenID, 0)
        XCTAssertEqual(finiteUnderflowToZeroMass.tokenID, 0)
        XCTAssertEqual(
            maximumSubtraction.randomWord,
            finiteUnderflowToZeroMass.randomWord)
    }

    func testCallerMutationCannotChangeRetainedSelection() throws {
        var logits = [0.0, 1.0, 1.0, 0.0]
        let seed = SamplingSeedV1.callerSupplied(0)
        let policy = try SamplingPolicyV1.sampled(
            temperature: 1,
            topP: 0,
            seed: seed)
        let selection = try SamplingRank1OracleV1.selectToken(
            logits: logits,
            policy: policy,
            committedSampleOrdinal: 7)
        let retainedAddress = try XCTUnwrap(selection.drawAddress)
        let retainedWord = try XCTUnwrap(selection.randomWord)

        for index in logits.indices {
            logits[index] = Double(index) - 100
        }

        XCTAssertEqual(selection.tokenID, 1)
        XCTAssertEqual(selection.drawAddress, .some(retainedAddress))
        XCTAssertEqual(selection.randomWord, .some(retainedWord))
        XCTAssertEqual(
            storedLabels(selection),
            ["tokenID", "drawAddress", "randomWord"])

        let greedy = try SamplingRank1OracleV1.selectToken(
            logits: logits,
            policy: SamplingPolicyV1.greedy())
        XCTAssertNil(greedy.drawAddress)
        XCTAssertNil(greedy.randomWord)
    }

    func testDeterministicStatisticalCohortsMatchExactCountsAndBounds()
        throws
    {
        try assertCohort(
            logits: [0, 0, 0, 0],
            temperature: 1,
            topP: 1,
            exactObservedCounts: [16_302, 16_415, 16_392, 16_427],
            expectedCounts: [16_384, 16_384, 16_384, 16_384],
            bounds: [668, 668, 668, 668])
        try assertCohort(
            logits: [0, 1.0986122886681098],
            temperature: 1,
            topP: 1,
            exactObservedCounts: [16_302, 49_234],
            expectedCounts: [16_384, 49_152],
            bounds: [668, 668])
        try assertCohort(
            logits: [0, 0, 0, 0],
            temperature: 1,
            topP: 0.5,
            exactObservedCounts: [32_717, 32_819, 0, 0],
            expectedCounts: [32_768, 32_768, 0, 0],
            bounds: [770, 770, 0, 0])
    }

    private struct DrawVector {
        let seed: Int64
        let ordinal: UInt64
        let word: UInt64
        let uniformBits: UInt64
    }

    private struct BoundaryFixture {
        let seed: Int64
        let word: UInt64
        let numerator: UInt64
        let expectedToken: Int
    }

    private func assertEquatableAndSendable<T: Equatable & Sendable>(
        _ value: T,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(value, value, file: file, line: line)
    }

    private func assertFailureConforms<T: Error & Equatable & Sendable>(
        _ value: T,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(value, value, file: file, line: line)
    }

    private func assertThrowsSamplingFailure<T>(
        _ expression: @autoclosure () throws -> T,
        _ expected: SamplingContractFailure,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(try expression(), file: file, line: line) {
            error in
            XCTAssertEqual(
                error as? SamplingContractFailure,
                expected,
                file: file,
                line: line)
        }
    }

    private func assertSampledFixture(
        logits: [Double],
        temperature: Double,
        topP: Double,
        seed seedValue: Int64,
        ordinal: UInt64,
        expectedToken: Int,
        expectedWord: UInt64,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let seed = SamplingSeedV1.callerSupplied(seedValue)
        let policy = try SamplingPolicyV1.sampled(
            temperature: temperature,
            topP: topP,
            seed: seed)
        let address = try SamplingDrawAddressV1(
            seed: seed,
            committedSampleOrdinal: ordinal)
        let selection = try SamplingRank1OracleV1.selectToken(
            logits: logits,
            policy: policy,
            committedSampleOrdinal: ordinal)

        XCTAssertEqual(selection.tokenID, expectedToken, file: file, line: line)
        XCTAssertEqual(selection.drawAddress, .some(address), file: file, line: line)
        XCTAssertEqual(selection.randomWord, .some(expectedWord), file: file, line: line)
        XCTAssertEqual(address.randomWord(), expectedWord, file: file, line: line)
    }

    private func assertCohort(
        logits: [Double],
        temperature: Double,
        topP: Double,
        exactObservedCounts: [Int],
        expectedCounts: [Int],
        bounds: [Int],
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        var counts = Array(repeating: 0, count: logits.count)

        for seedValue in 0..<65_536 {
            let seed = SamplingSeedV1.callerSupplied(Int64(seedValue))
            let policy = try SamplingPolicyV1.sampled(
                temperature: temperature,
                topP: topP,
                seed: seed)
            let selection = try SamplingRank1OracleV1.selectToken(
                logits: logits,
                policy: policy)

            XCTAssertNotNil(selection.drawAddress, file: file, line: line)
            XCTAssertNotNil(selection.randomWord, file: file, line: line)
            counts[selection.tokenID] += 1
        }

        XCTAssertEqual(counts, exactObservedCounts, file: file, line: line)
        XCTAssertEqual(expectedCounts.count, counts.count, file: file, line: line)
        XCTAssertEqual(bounds.count, counts.count, file: file, line: line)

        for index in counts.indices {
            XCTAssertLessThanOrEqual(
                abs(counts[index] - expectedCounts[index]),
                bounds[index],
                file: file,
                line: line)
        }
    }

    private func storedLabels<T>(_ value: T) -> [String] {
        Mirror(reflecting: value).children.map { child in
            child.label ?? ""
        }
    }

    private func uniformNumerator(from word: UInt64) -> UInt64 {
        2 * (word >> 12) + 1
    }
}
