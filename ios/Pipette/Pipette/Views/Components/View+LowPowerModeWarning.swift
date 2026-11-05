import SwiftUI

extension View {
    /// Warns that Low Power Mode skews benchmark results before a run starts.
    /// `alsoNote` is another warning for the same run, appended so one alert asks once.
    func lowPowerModeWarning(isPresented: Binding<Bool>, alsoNote: String? = nil,
                             onRunAnyway: @escaping () -> Void) -> some View {
        alert("Low Power Mode is on", isPresented: isPresented) {
            Button("Run Anyway", role: .destructive, action: onRunAnyway)
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Low Power Mode throttles the CPU and GPU, which can skew benchmark results. Disable it in Settings → Battery for accurate numbers."
                + (alsoNote.map { "\n\n" + $0 } ?? ""))
        }
    }
}
