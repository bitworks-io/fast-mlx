import Testing

@testable import HarnessCore

/// The chunked teacher-forced scoring plan: pure position bookkeeping for scoring a forced
/// continuation by CHUNKED PREFILL (multi-token forwards) instead of thousands of single-token
/// steps. Row for forced position i is the model's output at input index promptCount-1+i over
/// the full input = prompt + forced.dropLast().
struct ForcedScoringPlanTests {

    @Test func singleChunkCoversAllPositions() {
        let plan = forcedScoringPlan(promptCount: 3, forcedCount: 5, wantedPositions: nil, chunkSize: 100)
        #expect(plan.chunks.count == 1)
        #expect(plan.chunks[0].inputRange == 0..<7) // 3 + 5 - 1
        // rows: forced position i at input index 2+i -> local index 2+i in this chunk
        #expect(plan.chunks[0].rows == (0..<5).map { RowSelection(localIndex: 2 + $0, position: $0) })
    }

    @Test func chunkBoundariesPartitionInputExactly() {
        let plan = forcedScoringPlan(promptCount: 3, forcedCount: 5, wantedPositions: nil, chunkSize: 4)
        #expect(plan.chunks.map(\.inputRange) == [0..<4, 4..<7])
        // full coverage, no overlap, in order
        let all = plan.chunks.flatMap { Array($0.inputRange) }
        #expect(all == Array(0..<7))
    }

    @Test func rowsMapCorrectlyAcrossChunkBoundary() {
        // promptCount=3: forced position i lives at input index 2+i.
        let plan = forcedScoringPlan(promptCount: 3, forcedCount: 5, wantedPositions: nil, chunkSize: 4)
        // chunk 0 = input [0,4): positions 0 (idx 2), 1 (idx 3)
        #expect(plan.chunks[0].rows == [
            RowSelection(localIndex: 2, position: 0),
            RowSelection(localIndex: 3, position: 1),
        ])
        // chunk 1 = input [4,7): positions 2 (idx 4), 3 (idx 5), 4 (idx 6)
        #expect(plan.chunks[1].rows == [
            RowSelection(localIndex: 0, position: 2),
            RowSelection(localIndex: 1, position: 3),
            RowSelection(localIndex: 2, position: 4),
        ])
    }

    @Test func everyPositionAppearsExactlyOnce() {
        let plan = forcedScoringPlan(promptCount: 7, forcedCount: 23, wantedPositions: nil, chunkSize: 5)
        let positions = plan.chunks.flatMap { $0.rows.map(\.position) }
        #expect(positions == Array(0..<23)) // ascending, complete, no duplicates
    }

    @Test func sampledPositionsAreHonoredInOrder() {
        let wanted = [0, 7, 8, 21]
        let plan = forcedScoringPlan(promptCount: 4, forcedCount: 22, wantedPositions: wanted, chunkSize: 6)
        let positions = plan.chunks.flatMap { $0.rows.map(\.position) }
        #expect(positions == wanted)
        // each selected row's local index must point at the input index promptCount-1+position
        for chunk in plan.chunks {
            for row in chunk.rows {
                #expect(chunk.inputRange.lowerBound + row.localIndex == 4 - 1 + row.position)
            }
        }
    }

    @Test func promptLongerThanChunkYieldsRowlessLeadingChunks() {
        // 10-token prompt, chunk 4: first two chunks are pure context, no scored rows.
        let plan = forcedScoringPlan(promptCount: 10, forcedCount: 3, wantedPositions: nil, chunkSize: 4)
        #expect(plan.chunks.map(\.inputRange) == [0..<4, 4..<8, 8..<12])
        #expect(plan.chunks[0].rows.isEmpty)
        #expect(plan.chunks[1].rows.isEmpty)
        // forced position i at input index 9+i: positions 0,1,2 at indices 9,10,11
        #expect(plan.chunks[2].rows == [
            RowSelection(localIndex: 1, position: 0),
            RowSelection(localIndex: 2, position: 1),
            RowSelection(localIndex: 3, position: 2),
        ])
    }

    @Test func singleForcedTokenScoresPromptOnly() {
        // forcedCount=1: input = prompt only; the one row is at input index promptCount-1.
        let plan = forcedScoringPlan(promptCount: 5, forcedCount: 1, wantedPositions: nil, chunkSize: 512)
        #expect(plan.chunks.count == 1)
        #expect(plan.chunks[0].inputRange == 0..<5)
        #expect(plan.chunks[0].rows == [RowSelection(localIndex: 4, position: 0)])
    }
}
