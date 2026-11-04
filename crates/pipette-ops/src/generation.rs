//! What one prompt's generation was, whatever the server speaks
//! ([`generations`]).
//!
//! An engine talks to its server in that server's format, stamps each thing it
//! reads as it arrives, and hands the lot here. Which arrival is the first
//! token, the last and the end is then decided once, for every engine.

use std::time::Instant;

use anyhow::Context;

use pipette_plan_types::PositiveCount;

/// Something an engine read from its server for one prompt, and when.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// Which prompt, in a request that sends several.
    pub prompt: usize,
    pub at: Instant,
    pub what: Arrived,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Arrived {
    /// An arrival that counts as a token.
    Token,
    Finish(Finish),
    Counts(Counts),
    /// Times the server measured itself.
    ServerClock(ServerClock),
    /// Nothing more comes for this prompt.
    End,
}

/// Why a generation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finish {
    /// At the requested number of tokens.
    Limit,
    /// On an end-of-sequence token or a stop string.
    Stop,
    /// In the server's words, when it names neither.
    Other(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

impl Counts {
    /// As the measurement entries take them.
    pub fn positive(self) -> anyhow::Result<(PositiveCount, PositiveCount)> {
        Ok((
            PositiveCount::try_from(self.prompt_tokens).context("invalid prompt_tokens")?,
            PositiveCount::try_from(self.completion_tokens).context("invalid completion_tokens")?,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ServerClock {
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

/// Which clock times a generation's prefill and decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Clock {
    /// This side's stamps: the prefill up to the first token, the decode from
    /// there to the last.
    Client,
    /// The server's own timers ([`ServerClock`]).
    Server,
}

/// One prompt's generation. The `_ms` times are the client's, from the send.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Generation {
    pub first_token_ms: f64,
    pub last_token_ms: f64,
    /// When the server said why it stopped, or the prompt's end if it never did.
    pub end_ms: f64,
    pub finish: Option<Finish>,
    pub counts: Option<Counts>,
    pub server_clock: Option<ServerClock>,
}

impl Generation {
    /// The counts of a generation the server says ran to its limit, which a
    /// timing cell needs before it can trust them.
    pub fn counts_at_limit(&self) -> anyhow::Result<Counts> {
        let stopped = match &self.finish {
            Some(Finish::Limit) => {
                return self.counts.context("the server reported no token counts");
            }
            Some(Finish::Stop) => "on an end-of-sequence token or a stop string".to_owned(),
            Some(Finish::Other(reason)) => format!("with {reason:?}"),
            None => "without saying why".to_owned(),
        };
        anyhow::bail!(
            "the server did not stop at the requested generation limit: it stopped {stopped}"
        )
    }

    pub fn phases_ms(&self, clock: Clock) -> anyhow::Result<(f64, f64)> {
        match clock {
            Clock::Client => Ok((
                self.first_token_ms,
                self.last_token_ms - self.first_token_ms,
            )),
            Clock::Server => {
                let clock = self
                    .server_clock
                    .context("the server reported no timings of its own")?;
                Ok((clock.prefill_ms, clock.decode_ms))
            }
        }
    }
}

/// One generation per prompt, in prompt order, from what arrived for each.
///
/// The first and last token are the first and last [`Arrived::Token`], and the
/// end is the prompt's [`Arrived::Finish`], or its [`Arrived::End`] if none
/// came, all measured from `sent`: what follows a finish, such as a usage
/// trailer, is bookkeeping. A later finish, count or server clock replaces an
/// earlier one. A prompt the server never answered, or never ended, is refused,
/// and so is anything after an end.
pub fn generations(
    prompts: usize,
    sent: Instant,
    observed: &[Observation],
) -> anyhow::Result<Vec<Generation>> {
    let mut seen: Vec<Seen> = (0..prompts).map(|_| Seen::default()).collect();
    add_all(&mut seen, observed)?;
    seen.into_iter()
        .enumerate()
        .map(|(prompt, seen)| seen.into_generation(prompt, sent))
        .collect()
}

/// [`generations`] of a request that sends one prompt.
pub fn generation(sent: Instant, observed: &[Observation]) -> anyhow::Result<Generation> {
    let mut seen = [Seen::default()];
    add_all(&mut seen, observed)?;
    let [seen] = seen;
    seen.into_generation(0, sent)
}

/// What arrived for one prompt so far.
#[derive(Default)]
struct Seen {
    tokens: Option<(Instant, Instant)>,
    end: Option<Instant>,
    finish: Option<(Instant, Finish)>,
    counts: Option<Counts>,
    server_clock: Option<ServerClock>,
}

impl Seen {
    fn add(&mut self, prompt: usize, at: Instant, what: &Arrived) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.end.is_none(),
            "the server sent more for prompt {prompt} after its end"
        );
        match what {
            Arrived::Token => {
                self.tokens = Some((self.tokens.map_or(at, |(first, _)| first), at));
            }
            Arrived::Finish(finish) => self.finish = Some((at, finish.clone())),
            Arrived::Counts(counts) => self.counts = Some(*counts),
            Arrived::ServerClock(clock) => self.server_clock = Some(*clock),
            Arrived::End => self.end = Some(at),
        }
        Ok(())
    }

    fn into_generation(self, prompt: usize, sent: Instant) -> anyhow::Result<Generation> {
        let (first, last) = self
            .tokens
            .with_context(|| format!("prompt {prompt} ended without a token"))?;
        let end = self
            .end
            .with_context(|| format!("prompt {prompt} never ended"))?;
        let (finished, finish) = self.finish.unzip();
        let ms = |at: Instant| at.saturating_duration_since(sent).as_secs_f64() * 1000.0;
        Ok(Generation {
            first_token_ms: ms(first),
            last_token_ms: ms(last),
            end_ms: ms(finished.unwrap_or(end)),
            finish,
            counts: self.counts,
            server_clock: self.server_clock,
        })
    }
}

fn add_all(seen: &mut [Seen], observed: &[Observation]) -> anyhow::Result<()> {
    let prompts = seen.len();
    observed
        .iter()
        .try_for_each(|Observation { prompt, at, what }| {
            seen.get_mut(*prompt)
                .with_context(|| {
                    format!("the server answered for prompt {prompt} of a {prompts}-prompt request")
                })?
                .add(*prompt, *at, what)
        })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rstest::rstest;

    use super::*;

    fn at(start: Instant, ms: u64) -> Instant {
        start + Duration::from_millis(ms)
    }

    fn arrival(start: Instant, prompt: usize, ms: u64, what: Arrived) -> Observation {
        Observation {
            prompt,
            at: at(start, ms),
            what,
        }
    }

    const COUNTS: Counts = Counts {
        prompt_tokens: 16,
        completion_tokens: 2,
    };

    #[test]
    fn each_prompt_is_read_from_its_own_arrivals() -> anyhow::Result<()> {
        let start = Instant::now();
        let observed = [
            arrival(start, 0, 5, Arrived::Token),
            arrival(start, 1, 6, Arrived::Token),
            arrival(start, 0, 7, Arrived::Token),
            arrival(start, 1, 8, Arrived::Finish(Finish::Limit)),
            arrival(start, 1, 8, Arrived::Counts(COUNTS)),
            arrival(start, 1, 8, Arrived::End),
            arrival(start, 0, 9, Arrived::Finish(Finish::Stop)),
            arrival(start, 0, 10, Arrived::End),
        ];
        let got = generations(2, start, &observed)?;
        assert_eq!(
            got,
            [
                Generation {
                    first_token_ms: 5.0,
                    last_token_ms: 7.0,
                    end_ms: 9.0,
                    finish: Some(Finish::Stop),
                    counts: None,
                    server_clock: None,
                },
                Generation {
                    first_token_ms: 6.0,
                    last_token_ms: 6.0,
                    end_ms: 8.0,
                    finish: Some(Finish::Limit),
                    counts: Some(COUNTS),
                    server_clock: None,
                },
            ]
        );
        Ok(())
    }

    /// The usage trailer and the stream's end after a finish are not timed, a
    /// later finish replaces an earlier one, time included, and only a prompt
    /// that never says why it stopped ends at its end.
    #[rstest]
    #[case::finish(vec![(20, Finish::Limit)], Some(Finish::Limit), 20.0)]
    #[case::later_finish(
        vec![(15, Finish::Stop), (20, Finish::Limit)],
        Some(Finish::Limit),
        20.0
    )]
    #[case::no_finish(vec![], None, 30.0)]
    fn the_end_is_the_finish(
        #[case] finishes: Vec<(u64, Finish)>,
        #[case] finish: Option<Finish>,
        #[case] end_ms: f64,
    ) -> anyhow::Result<()> {
        let start = Instant::now();
        let observed: Vec<_> = std::iter::once(arrival(start, 0, 10, Arrived::Token))
            .chain(
                finishes
                    .into_iter()
                    .map(|(ms, finish)| arrival(start, 0, ms, Arrived::Finish(finish))),
            )
            .chain([
                arrival(start, 0, 25, Arrived::Counts(COUNTS)),
                arrival(start, 0, 30, Arrived::End),
            ])
            .collect();
        let got = generation(start, &observed)?;
        assert_eq!((got.finish, got.end_ms), (finish, end_ms));
        Ok(())
    }

    #[test]
    fn a_whole_reply_is_one_arrival() -> anyhow::Result<()> {
        let start = Instant::now();
        let clock = ServerClock {
            prefill_ms: 1.5,
            decode_ms: 2.5,
        };
        let observed = [
            arrival(start, 0, 12, Arrived::Token),
            arrival(start, 0, 12, Arrived::ServerClock(clock)),
            arrival(start, 0, 12, Arrived::End),
        ];
        let got = generation(start, &observed)?;
        assert_eq!(
            (got.first_token_ms, got.last_token_ms, got.end_ms),
            (12.0, 12.0, 12.0)
        );
        assert_eq!(got.server_clock, Some(clock));
        Ok(())
    }

    #[rstest]
    #[case::unknown_prompt(
        vec![(2, Arrived::Token), (2, Arrived::End)],
        "prompt 2 of a 2-prompt request"
    )]
    #[case::ended_twice(
        vec![(1, Arrived::Token), (1, Arrived::End), (1, Arrived::End)],
        "prompt 1 after its end"
    )]
    #[case::token_after_the_end(
        vec![(0, Arrived::Token), (0, Arrived::End), (0, Arrived::Token)],
        "prompt 0 after its end"
    )]
    #[case::no_token(
        vec![(0, Arrived::End), (1, Arrived::Token), (1, Arrived::End)],
        "prompt 0 ended without a token"
    )]
    #[case::no_end(
        vec![(0, Arrived::Token), (0, Arrived::End), (1, Arrived::Token)],
        "prompt 1 never ended"
    )]
    fn a_prompt_left_unanswered_is_refused(
        #[case] arrivals: Vec<(usize, Arrived)>,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let start = Instant::now();
        let observed: Vec<_> = arrivals
            .into_iter()
            .map(|(prompt, what)| arrival(start, prompt, 1, what))
            .collect();
        let err = generations(2, start, &observed)
            .err()
            .context("expected a refusal")?;
        assert!(format!("{err:#}").contains(expected), "{err:#}");
        Ok(())
    }

    #[rstest]
    #[case::at_the_limit(Some(Finish::Limit), Some(COUNTS), None)]
    #[case::stopped(
        Some(Finish::Stop),
        Some(COUNTS),
        Some("it stopped on an end-of-sequence")
    )]
    #[case::other(
        Some(Finish::Other("abort".into())),
        Some(COUNTS),
        Some("it stopped with \"abort\"")
    )]
    #[case::unsaid(None, Some(COUNTS), Some("it stopped without saying why"))]
    #[case::no_counts(Some(Finish::Limit), None, Some("no token counts"))]
    fn only_a_generation_at_its_limit_gives_counts(
        #[case] finish: Option<Finish>,
        #[case] counts: Option<Counts>,
        #[case] refusal: Option<&str>,
    ) -> anyhow::Result<()> {
        let generation = Generation {
            finish,
            counts,
            ..Generation::default()
        };
        match (generation.counts_at_limit(), refusal) {
            (Ok(got), None) => assert_eq!(got, COUNTS),
            (Err(err), Some(refusal)) => {
                assert!(format!("{err:#}").contains(refusal), "{err:#}");
            }
            (got, refusal) => anyhow::bail!("expected {refusal:?}, got {got:?}"),
        }
        Ok(())
    }

    #[rstest]
    #[case::counts(16, 2, Ok((16, 2)))]
    #[case::no_prompt(0, 2, Err("invalid prompt_tokens"))]
    #[case::past_i32(16, 1 << 31, Err("invalid completion_tokens"))]
    fn counts_become_the_entries_counts(
        #[case] prompt_tokens: u32,
        #[case] completion_tokens: u32,
        #[case] expected: Result<(i32, i32), &str>,
    ) -> anyhow::Result<()> {
        let counts = Counts {
            prompt_tokens,
            completion_tokens,
        };
        match (counts.positive(), expected) {
            (Ok((prompt, completion)), Ok(want)) => {
                assert_eq!((prompt.into_inner(), completion.into_inner()), want);
            }
            (Err(err), Err(message)) => {
                assert!(format!("{err:#}").contains(message), "{err:#}");
            }
            (got, expected) => anyhow::bail!("expected {expected:?}, got {got:?}"),
        }
        Ok(())
    }

    #[rstest]
    #[case::client(Clock::Client, Some((5.0, 20.0)), Ok((5.0, 15.0)))]
    #[case::server(Clock::Server, Some((3.0, 12.0)), Ok((3.0, 12.0)))]
    #[case::server_missing(Clock::Server, None, Err("no timings of its own"))]
    fn phases_are_read_on_the_clock_asked_for(
        #[case] clock: Clock,
        #[case] server: Option<(f64, f64)>,
        #[case] expected: Result<(f64, f64), &str>,
    ) -> anyhow::Result<()> {
        let generation = Generation {
            first_token_ms: 5.0,
            last_token_ms: 20.0,
            server_clock: server.map(|(prefill_ms, decode_ms)| ServerClock {
                prefill_ms,
                decode_ms,
            }),
            ..Generation::default()
        };
        match (generation.phases_ms(clock), expected) {
            (Ok(got), Ok(want)) => assert_eq!(got, want),
            (Err(err), Err(message)) => {
                assert!(format!("{err:#}").contains(message), "{err:#}");
            }
            (got, expected) => anyhow::bail!("expected {expected:?}, got {got:?}"),
        }
        Ok(())
    }
}
