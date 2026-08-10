import SwiftUI

/// Colors for `JobLiveActivityView` so the same live-activity block reads in both
/// the system-styled running page and the dark Pocket Mode card.
struct JobActivityPalette {
    var primaryText: Color
    var secondaryText: Color
    /// Emphasis color for the cooling caption — a cool tint that ties to the
    /// card's cooling wash.
    var accent: Color
}

/// The "what is the benchmark doing right now" block shared by the running job
/// page and Pocket Mode — the original text-only treatment: the current cell on
/// one line and the fine-grained per-rep progress on the next. No icon, no box.
///
/// Constant height: the cell label reserves two lines (`reservesSpace`) so a one-
/// vs two-line model name doesn't move the layout, and the progress line reserves
/// one. While the thermal gate is cooling, that same second line shows the
/// cooldown timer (elapsed vs. the deadline it's allowed to cool); the ambient
/// "we're cooling" cue is the card background wash, which costs no space.
struct JobLiveActivityView: View {
    let currentCellLabel: String
    /// The fine-grained per-rep progress ("Measurement 3/5", "Loading…") shown
    /// when not cooling.
    let progressText: String
    /// Non-nil while the thermal gate is cooling; drives the live cooldown line.
    let cooling: JobCoolingState?
    let palette: JobActivityPalette

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(currentCellLabel)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(palette.primaryText)
                .lineLimit(2, reservesSpace: true)
                .frame(maxWidth: .infinity, alignment: .leading)
            secondLine
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder private var secondLine: some View {
        if let cooling {
            // A TimelineView ticks *only this label* once a second, system-scheduled
            // — no manual timer and no whole-view invalidation. Emphasized (accent +
            // semibold) so cooling reads as the active state.
            TimelineView(.periodic(from: cooling.since, by: 1)) { context in
                line(cooling.caption(at: context.date), emphasized: true)
            }
        } else {
            line(progressText, emphasized: false)
        }
    }

    // Fixed size across states (only weight/color change) so the reserved line
    // never jumps as it swaps between cooling and normal progress.
    private func line(_ text: String, emphasized: Bool) -> some View {
        Text(text.isEmpty ? " " : text)
            .font(.system(size: 14, weight: emphasized ? .semibold : .regular))
            .foregroundStyle(emphasized ? palette.accent : palette.secondaryText)
            .lineLimit(1, reservesSpace: true)
            .monospacedDigit()
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// What the "Device temperature" indicator shows. While cooling it reports the
/// current reading *and* the setpoint (`43°C → 36°C`) and colors by temp-vs-target
/// — orange until the device drops to the setpoint — rather than the coarse OS
/// thermal state, which can read "nominal"/green even while the gate holds the
/// device above its target.
struct DeviceThermalDisplay {
    let text: String
    let iconName: String
    let color: Color

    init(temperatureC: Double?, cooling: JobCoolingState?, state: ProcessInfo.ThermalState) {
        if let cooling, let temp = temperatureC {
            text = "\(Int(temp.rounded()))°C → \(Int(cooling.targetC.rounded()))°C"
            iconName = "thermometer.high"
            color = temp > cooling.targetC ? .orange : .green
        } else if let temp = temperatureC {
            text = "\(Int(temp.rounded()))°C"
            iconName = state.iconName
            color = state.indicatorColor
        } else {
            text = state.shortLabel.capitalized
            iconName = state.iconName
            color = state.indicatorColor
        }
    }
}

extension ProcessInfo.ThermalState {
    /// SF Symbol matching the current thermal pressure, for a device-temperature
    /// indicator.
    var iconName: String {
        switch self {
        case .nominal:  return "checkmark.circle.fill"
        case .fair:     return "thermometer.medium"
        case .serious:  return "thermometer.high"
        case .critical: return "exclamationmark.triangle.fill"
        @unknown default: return "thermometer"
        }
    }

    /// Traffic-light tint for the indicator icon.
    var indicatorColor: Color {
        switch self {
        case .nominal:  return .green
        case .fair:     return .yellow
        case .serious:  return .orange
        case .critical: return .red
        @unknown default: return .secondary
        }
    }
}
