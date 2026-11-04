import Combine
import SwiftUI
import UIKit

/// Main tab navigation (Jobs, Models, Settings) with a custom pill tab bar.
struct MainTabView: View {
    @Binding var isRegistered: Bool
    @Environment(DeepLinkRouter.self) private var deepLinkRouter
    @State private var selectedTab: MainTab = .jobs
    @State private var hidesPillTabBar = false
    @State private var pillTabBarHeight: CGFloat = 0
    @State private var keyboardIsVisible = false

    var body: some View {
        TabView(selection: $selectedTab) {
            JobsView(selectedTab: $selectedTab)
                .tag(MainTab.jobs)
                .toolbar(.hidden, for: .tabBar)
                .environment(\.pillTabBarReservedHeight, reservedPillTabBarHeight)
            ModelsView()
                .tag(MainTab.models)
                .toolbar(.hidden, for: .tabBar)
                .environment(\.pillTabBarReservedHeight, reservedPillTabBarHeight)
            SettingsView(isRegistered: $isRegistered)
                .tag(MainTab.settings)
                .toolbar(.hidden, for: .tabBar)
                .environment(\.pillTabBarReservedHeight, reservedPillTabBarHeight)
        }
        .overlay(alignment: .bottom) {
            ZStack {
                if !hidesPillTabBar && !keyboardIsVisible {
                    PillTabBar(selected: $selectedTab)
                        .padding(.horizontal, 20)
                        .padding(.top, 8)
                        .padding(.bottom, 4)
                        .frame(maxWidth: .infinity)
                        .readPillTabBarHeight()
                        .transition(.offset(y: 100).combined(with: .opacity))
                }
            }
            .animation(.snappy(duration: 0.2, extraBounce: 0.1), value: [hidesPillTabBar, keyboardIsVisible])
        }
        .onPreferenceChange(PillTabBarHiddenPreferenceKey.self) { hidden in
            hidesPillTabBar = hidden
        }
        // A deep link asks for a tab by setting `requestedTab`; consume and clear
        // it so a later link to the same tab still fires.
        .onChange(of: deepLinkRouter.requestedTab) { _, tab in
            guard let tab else { return }
            selectedTab = tab
            deepLinkRouter.requestedTab = nil
        }
        .onReceive(NotificationCenter.default.publisher(
            for: UIResponder.keyboardWillShowNotification
        )) { _ in
            withAnimation(.easeOut(duration: 0.25)) { keyboardIsVisible = true }
        }
        .onReceive(NotificationCenter.default.publisher(
            for: UIResponder.keyboardWillHideNotification
        )) { _ in
            withAnimation(.easeOut(duration: 0.25)) { keyboardIsVisible = false }
        }
        .onPreferenceChange(PillTabBarHeightPreferenceKey.self) { height in
            pillTabBarHeight = height
        }
    }

    private var reservedPillTabBarHeight: CGFloat {
        (hidesPillTabBar || keyboardIsVisible) ? 0 : pillTabBarHeight
    }
}

private struct PillTabBarHeightPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

private extension View {
    func readPillTabBarHeight() -> some View {
        background {
            GeometryReader { proxy in
                Color.clear.preference(
                    key: PillTabBarHeightPreferenceKey.self,
                    value: proxy.size.height
                )
            }
        }
    }
}

private struct PillTabBarReservedHeightKey: EnvironmentKey {
    static let defaultValue: CGFloat = 0
}

extension EnvironmentValues {
    var pillTabBarReservedHeight: CGFloat {
        get { self[PillTabBarReservedHeightKey.self] }
        set { self[PillTabBarReservedHeightKey.self] = newValue }
    }
}

struct PillTabBarHiddenPreferenceKey: PreferenceKey {
    static var defaultValue = false

    static func reduce(value: inout Bool, nextValue: () -> Bool) {
        value = value || nextValue()
    }
}

enum MainTab: Hashable {
    case jobs
    case models
    case settings

    var iconName: String {
        switch self {
        case .jobs:     return "list.bullet.clipboard"
        case .models:   return "cube"
        case .settings: return "gearshape"
        }
    }

    var accessibilityLabel: String {
        switch self {
        case .jobs:     return "Jobs"
        case .models:   return "Models"
        case .settings: return "Settings"
        }
    }
}

/// Pill-shaped bottom tab bar with one black-filled active item.
struct PillTabBar: View {
    @Binding var selected: MainTab

    private let tabs: [MainTab] = [.jobs, .models, .settings]

    @Namespace private var selection

    var body: some View {
        HStack(spacing: 8) {
            ForEach(tabs, id: \.self) { tab in
                Button {
                    withAnimation(.snappy(duration: 0.2, extraBounce: 0.1)) {
                        selected = tab
                    }
                } label: {
                    ZStack {
                        if selected == tab {
                            Capsule()
                                .fill(Color.primary)
                                .matchedGeometryEffect(id: "selection", in: selection)
                        }
                        Image(systemName: tab.iconName)
                            .font(.system(size: 20, weight: .regular))
                            .foregroundStyle(
                                selected == tab
                                    ? Color(.systemBackground)
                                    : Color.secondary
                            )
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 52)
                    .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(tab.accessibilityLabel)
                .accessibilityAddTraits(selected == tab ? .isSelected : [])
            }
        }
        .padding(6)
        .background(
            Capsule()
                .fill(.clear)
                .glassEffect(.regular.interactive(), in: .capsule)
        )
    }
}
