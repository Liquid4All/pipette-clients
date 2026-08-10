import SwiftUI

extension View {
    /// Warns that Low Power Mode throttles CPU/GPU clocks and skews benchmark
    /// results. `onRunAnyway` proceeds with the run; cancelling lets the user go
    /// disable it first. Shared by the run-start flows in `NewJobView` (new jobs)
    /// and `JobDetailView` (resume/retry/rerun).
    func lowPowerModeWarning(isPresented: Binding<Bool>, onRunAnyway: @escaping () -> Void) -> some View {
        alert("Low Power Mode is on", isPresented: isPresented) {
            Button("Run Anyway", role: .destructive, action: onRunAnyway)
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Low Power Mode throttles the CPU and GPU, which can skew benchmark results. Disable it in Settings → Battery for accurate numbers.")
        }
    }
}
