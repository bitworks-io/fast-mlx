import HarnessCore

@main
struct Harness {
    static func main() async {
        print("harness ready: \(HarnessCore.ready)")
    }
}
