import CoreMotion
import Foundation

/// Public, no-private-API SoC-temperature estimator: infers temperature from the
/// MEMS IMU **bias drift** via a per-device 6-axis linear model calibrated once
/// against `soc_temp`. Used by the readiness gate as the public fallback when the
/// private die-temperature sensor isn't available (Release builds).
///
/// Accuracy ~±1.5 °C with heavy averaging; the device must be **stationary** and in
/// the **same orientation** it was calibrated in (the model is anchored to that
/// unit's gravity baseline + per-chip bias — it does NOT transfer across devices).
/// Best used for a `< 40 °C`-style gate with margin, not a tight near-floor setpoint.
nonisolated enum IMUThermometer {
    private static let betaKey = "imuThermoBeta_v1"

    private static let motion: CMMotionManager = {
        let m = CMMotionManager()
        m.gyroUpdateInterval = 0.005
        m.accelerometerUpdateInterval = 0.005
        return m
    }()

    /// Persisted per-device fit `[intercept, gx, gy, gz, ax, ay, az]`, or nil.
    static var coefficients: [Double]? {
        get { UserDefaults.standard.array(forKey: betaKey) as? [Double] }
        set { UserDefaults.standard.set(newValue, forKey: betaKey) }
    }

    static var isCalibrated: Bool { (coefficients?.count ?? 0) == 7 }

    private static func ensureUpdates() {
        if motion.isGyroAvailable, !motion.isGyroActive { motion.startGyroUpdates() }
        if motion.isAccelerometerAvailable, !motion.isAccelerometerActive {
            motion.startAccelerometerUpdates()
        }
    }

    /// Heavily-averaged 6-axis reading `[gx,gy,gz,ax,ay,az]` (`k` reads @≈200 Hz) —
    /// the averaging is what turns the noisy MEMS bias into a ~±1.5 °C signal.
    static func averagedIMU(_ k: Int = 100) -> [Double] {
        ensureUpdates()
        var s = [Double](repeating: 0, count: 6)
        for _ in 0 ..< k {
            if let g = motion.gyroData?.rotationRate { s[0] += g.x; s[1] += g.y; s[2] += g.z }
            if let a = motion.accelerometerData?.acceleration { s[3] += a.x; s[4] += a.y; s[5] += a.z }
            Thread.sleep(forTimeInterval: 0.005)
        }
        return s.map { $0 / Double(k) }
    }

    /// Estimated SoC temperature (°C) from the IMU, or nil if not yet calibrated.
    static func estimate() -> Double? {
        guard let b = coefficients, b.count == 7 else { return nil }
        let imu = averagedIMU()
        return b[0] + zip(b.dropFirst(), imu).map(*).reduce(0, +)
    }

    /// Fit + persist the model from `(imu, temp)` pairs. Returns training RMSE (°C),
    /// or nil if the system is singular / under-determined.
    @discardableResult
    static func calibrate(_ samples: [(imu: [Double], temp: Double)]) -> Double? {
        let xrows = samples.map { [1.0] + $0.imu }
        let y = samples.map(\.temp)
        guard let beta = olsSolve(xrows, y) else { return nil }
        coefficients = beta
        let ss = zip(xrows, y).reduce(0.0) { acc, pair in
            let est = zip(beta, pair.0).map(*).reduce(0, +)
            return acc + (est - pair.1) * (est - pair.1)
        }
        return (ss / Double(y.count)).squareRoot()
    }

    /// Ordinary least squares for `[intercept + 6 axes]` via Gauss-Jordan on the
    /// 7×7 normal equations.
    private static func olsSolve(_ xrows: [[Double]], _ y: [Double]) -> [Double]? {
        let m = 7, n = xrows.count
        guard n >= m else { return nil }
        var a = [[Double]](repeating: [Double](repeating: 0, count: m + 1), count: m)
        for i in 0 ..< n {
            for r in 0 ..< m {
                for c in 0 ..< m { a[r][c] += xrows[i][r] * xrows[i][c] }
                a[r][m] += xrows[i][r] * y[i]
            }
        }
        for c in 0 ..< m {
            var p = c
            for r in (c + 1) ..< m where abs(a[r][c]) > abs(a[p][c]) { p = r }
            if abs(a[p][c]) < 1e-12 { return nil }
            a.swapAt(c, p)
            let piv = a[c][c]
            for k in 0 ... m { a[c][k] /= piv }
            for r in 0 ..< m where r != c {
                let f = a[r][c]
                for k in 0 ... m { a[r][k] -= f * a[c][k] }
            }
        }
        return (0 ..< m).map { a[$0][m] }
    }
}
