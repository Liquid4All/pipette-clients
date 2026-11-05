import Foundation

/// `sync`: pull the catalog, submit what is pending, then refresh the scores of submitted
/// evals, as `pipette sync` does. One line per phase, so the output has the same shape
/// whether or not a phase found work. The pull is public, so an unregistered device runs
/// it and is told the authenticated phases were skipped.
enum SyncCommands {
    /// The catalog pull, injectable so a test drives the command without a network.
    typealias CatalogPuller = (ServerUrl) async throws -> Int

    /// A failed catalog pull ends the command. Per-job submission errors and per-result
    /// refresh failures are reported and do not fail it, as in `pipette sync`.
    static func run(jobId: String?, serverUrl requested: String? = nil,
                    ws: PipetteWorkspace,
                    pullCatalog: CatalogPuller? = nil,
                    uploader: ResultUploader = .shared,
                    getJob: @escaping JobFetcher = fetchJobFromServer) async -> Bool {
        let registration = ws.registration.get()
        // The signing half, not just the record: a registration whose key this device
        // cannot read signs nothing.
        let session = try? ws.registration.mgmtSession()

        // Before resolving the server: for `job=` with no identity, the missing
        // registration is the blocker worth reporting.
        if session == nil, jobId != nil {
            HeadlessRunner.log("sync ERROR submitting needs a registration: "
                + "run `headlessrun auth register organization=… contact-email=…` "
                + "(or `headlessrun auth import bundle=…`) first")
            return false
        }

        guard let serverUrl = resolveServerUrl(registered: registration?.serverUrl,
                                               requested: requested)
        else { return false }

        let pull = pullCatalog ?? { server in
            try await BenchmarkSyncCoordinator.shared.sync(serverUrl: server, ws: ws)
        }
        do {
            let pulled = try await pull(serverUrl)
            HeadlessRunner.log("sync pulled benchmarks=\(pulled)")
        } catch {
            HeadlessRunner.log("sync ERROR benchmark pull failed: \(error.localizedDescription)")
            return false
        }

        guard let session else {
            // Names both phases: a reader of this line alone must know the catalog is all
            // it got.
            HeadlessRunner.log("sync skipped submit and scores: no signing identity "
                + "(run `headlessrun auth register …`, or `headlessrun auth import "
                + "bundle=…` to restore one, to submit results)")
            return true
        }

        // The CLI narrows by result id; iOS by job, the unit its store and submission
        // record share.
        let outcomes: [JobId: JobDrainOutcome]
        if let jobId {
            let id = JobId(jobId)
            outcomes = [id: await uploader.drainJob(jobId: id)]
        } else {
            outcomes = await uploader.drainAll()
        }
        let submitted = outcomes.values.reduce(0) { $0 + $1.submitted }
        let errors = outcomes.values.flatMap(\.errors)
        HeadlessRunner.log("sync submitted results=\(submitted) jobs=\(outcomes.count) "
            + "errors=\(errors.count)")
        for message in errors { HeadlessRunner.log("sync error \(message)") }

        await refreshScores(jobId: jobId.map(JobId.init), server: serverUrl,
                            auth: session.auth, ws: ws, getJob: getJob)
        return true
    }

    /// Pull the score of every submitted eval that has none yet. A refresh that fails is
    /// logged and counted as neither: the result stays synced and the next `sync` asks again.
    private static func refreshScores(jobId: JobId?, server: ServerUrl, auth: ClientIdentity,
                                      ws: PipetteWorkspace, getJob: @escaping JobFetcher) async {
        let results = ws.results
        let candidates: [CellId]
        if let jobId {
            candidates = (ws.loadJobManifest(jobId: jobId)?.cells ?? [])
                .map(\.cellId)
                .filter { results.location(of: $0) == .remoteSynced }
        } else {
            candidates = results.listIds(.remoteSynced)
        }

        var evalIds: [CellId] = []
        var alreadyScored = 0
        for id in candidates {
            switch scoreRefreshState(store: results, id: id) {
            case .needsRefresh: evalIds.append(id)
            case .alreadyScored: alreadyScored += 1
            case .notEval: break
            }
        }

        var scored = 0
        var pending = 0
        for id in evalIds {
            do {
                if try await refreshScoredJob(store: results, server: server, auth: auth,
                                              id: id, getJob: getJob) {
                    scored += 1
                } else {
                    pending += 1
                }
            } catch {
                if (error as? PipetteMgmtClient.Error)?.httpStatusCode == 404 {
                    AppLog.results.info("result \(id.value): not found on server, skipping")
                } else {
                    AppLog.results.info("result \(id.value): refresh failed: \(error)")
                }
            }
        }
        HeadlessRunner.log("sync scored results=\(scored + alreadyScored) "
            + "already_scored=\(alreadyScored) pending=\(pending)")
    }

    /// `nil` means the command has already said why it cannot answer.
    ///
    /// The registration wins because its key only signs for that server. A `server-url=`
    /// that disagrees is an error, not a silent no-op: whoever typed it wants `auth reset`.
    ///
    /// With neither, a refusal rather than the production collector: only `auth register`
    /// defaults a server, because it records what it picked.
    static func resolveServerUrl(registered: ServerUrl?, requested: String?) -> ServerUrl? {
        switch (registered, requested) {
        case let (registered?, requested?):
            // ``ServerUrl`` normalizes on construction, so another spelling of the
            // registered server is not a disagreement.
            guard let requestedServer = try? ServerUrl(requested),
                  requestedServer == registered
            else {
                HeadlessRunner.log("sync ERROR server-url=\(requested) is not the server this "
                    + "device is registered against (\(registered.value)); the signing key only "
                    + "signs for that one. Use `auth reset force=1` to move the device")
                return nil
            }
            return registered
        case let (registered?, _):
            return registered
        case let (nil, requested?):
            guard let server = try? ServerUrl(requested) else {
                HeadlessRunner.log("sync ERROR server-url=\(requested) is not an http(s) URL")
                return nil
            }
            return server
        case (nil, nil):
            HeadlessRunner.log("sync ERROR nothing names a server: this device holds no "
                + "registration, so pass `server-url=<url>` to pull the catalog, or "
                + "register with `headlessrun auth register organization=… contact-email=…`")
            return nil
        }
    }
}
