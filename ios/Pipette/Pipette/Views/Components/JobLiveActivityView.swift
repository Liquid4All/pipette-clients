import SwiftUI

/// Colors for `JobLiveActivityView`, so the block reads on both the system-styled
/// running page and the dark Pocket Mode card.
struct JobActivityPalette {
    var primaryText: Color
    var secondaryText: Color
    /// Emphasis color for the cooling caption.
    var accent: Color
}

/// The current cell and its per-rep progress, shared by the running job page and
/// Pocket Mode. While the thermal gate cools, the second line shows the cooldown
/// timer instead.
///
/// Constant height: both lines reserve their space so a one- vs two-line model
/// name or a state swap never moves the layout.
struct JobLiveActivityView: View {
    let currentCellLabel: String
    /// Per-rep progress ("Measurement 3/5"), shown when not cooling.
    let progressText: String
    /// Non-nil while the thermal gate is cooling.
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
            // Ticks only this label each second, without invalidating the whole view.
            TimelineView(.periodic(from: cooling.since, by: 1)) { context in
                line(cooling.caption(at: context.date), emphasized: true)
            }
        } else {
            line(progressText, emphasized: false)
        }
    }

    // Only weight and color change across states, so the reserved line never jumps.
    private func line(_ text: String, emphasized: Bool) -> some View {
        Text(text.isEmpty ? " " : text)
            .font(.system(size: 14, weight: emphasized ? .semibold : .regular))
            .foregroundStyle(emphasized ? palette.accent : palette.secondaryText)
            .lineLimit(1, reservesSpace: true)
            .monospacedDigit()
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// What the "Device temperature" indicator shows. While cooling it shows reading and
/// setpoint (`43°C → 36°C`) colored by temp vs target, not by the OS thermal state,
/// which can read nominal while the gate still holds the device above its target.
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
            text = formatPressureWord(state).capitalized
            iconName = state.iconName
            color = state.indicatorColor
        }
    }
}

extension ProcessInfo.ThermalState {
    /// SF Symbol for the device-temperature indicator.
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
