//! Read-only access to the releases bucket: presigned SigV4 GETs, sent through
//! `pipette_http::send`.
//!
//! A signed URL is a credential until it expires, so it never leaves this file.
//! Every failure is rendered here, without the URL reqwest would print and
//! without the response body, which S3 fills with the signature it was sent.

use std::collections::BTreeSet;
use std::io::{self, Read};
use std::time::Duration;

use anyhow::Context;
use hmac::{Hmac, KeyInit, Mac};
use reqwest::StatusCode;
use reqwest::blocking::Response;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use time::OffsetDateTime;

use pipette_http::HttpClient;

use crate::config::ReleasesBucket;
use crate::secret;

const USER_AGENT: &str = "pipette-ios-proxy";

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Bounds the wait for a response's headers and each read of its body, never a
/// whole download.
const READ_TIMEOUT: Duration = Duration::from_secs(60);

/// How long a signed URL is honoured; also how far the host clock may lag S3's.
const SIGNED_FOR: Duration = Duration::from_secs(15 * 60);

const ALGORITHM: &str = "AWS4-HMAC-SHA256";

/// A page holds at most 1000 keys of at most 1 KiB, escaped; past this it is
/// not a listing.
const LISTING_PAGE_MAX_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub(crate) enum S3Error {
    #[error("no response: {0}")]
    Unreachable(String),
    /// `reason` is S3's `Code: Message`, when the body carried one.
    #[error("HTTP {status}{}", .reason.as_ref().map(|reason| format!(": {reason}")).unwrap_or_default())]
    Refused {
        status: StatusCode,
        reason: Option<String>,
    },
    #[error("the response broke off: {0}")]
    Interrupted(String),
    /// Without the parser's message: it quotes the body, where a server that is
    /// not S3 may echo the signed query.
    #[error("the answer is not an S3 listing")]
    NotAListing,
    #[error("a truncated listing names no NextContinuationToken")]
    TruncatedWithoutToken,
    #[error("the listing named a continuation token it had already named")]
    RepeatedToken,
    /// `hmac` hashes a key longer than a block, so no key is refused in practice.
    #[error("the signing key was refused: {0}")]
    Signing(#[from] hmac::digest::InvalidLength),
}

impl S3Error {
    /// A 4xx: the bucket refuses this request as sent, so asking again gets the
    /// same answer.
    pub(crate) fn is_client_refusal(&self) -> bool {
        matches!(self, Self::Refused { status, .. } if status.is_client_error())
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub(crate) struct Object {
    pub(crate) key: String,
    #[serde(rename = "Size")]
    pub(crate) bytes: u64,
    #[serde(with = "time::serde::rfc3339")]
    pub(crate) last_modified: OffsetDateTime,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct ListBucketResult {
    /// Required, so a document that is not a listing does not read as an empty one.
    is_truncated: bool,
    #[serde(default)]
    contents: Vec<Object>,
    next_continuation_token: Option<String>,
}

/// The two elements of S3's `<Error>` body that carry no part of the request.
#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Refusal {
    code: String,
    message: Option<String>,
}

pub(crate) fn http_client() -> anyhow::Result<HttpClient> {
    HttpClient::builder(USER_AGENT)
        .connect_timeout(CONNECT_TIMEOUT)
        .request_timeout(READ_TIMEOUT)
        .build()
        .context("building the HTTPS client for the releases bucket")
}

/// Every object in the bucket: `ListObjectsV2` with no prefix and no delimiter,
/// followed to its last page.
pub(crate) fn list_objects(
    http: &HttpClient,
    releases: &ReleasesBucket,
) -> Result<Vec<Object>, S3Error> {
    let mut objects = Vec::new();
    let mut token: Option<String> = None;
    // Every token named so far: a server that hands one back again would
    // otherwise be followed forever.
    let mut named = BTreeSet::new();
    loop {
        let query: Vec<(&str, &str)> = [("list-type", "2")]
            .into_iter()
            .chain(token.as_deref().map(|token| ("continuation-token", token)))
            .collect();
        let body = get_bytes(http, releases, "", &query, LISTING_PAGE_MAX_BYTES)?;
        let page: ListBucketResult =
            quick_xml::de::from_reader(body.as_slice()).map_err(|_| S3Error::NotAListing)?;
        objects.extend(page.contents);
        let next = page.next_continuation_token.filter(|next| !next.is_empty());
        match (page.is_truncated, next) {
            (false, _) => return Ok(objects),
            (true, None) => return Err(S3Error::TruncatedWithoutToken),
            (true, Some(next)) if !named.insert(next.clone()) => {
                return Err(S3Error::RepeatedToken);
            }
            (true, Some(next)) => token = Some(next),
        }
    }
}

/// The body of `key`, cut at `limit` bytes.
pub(crate) fn get_bytes(
    http: &HttpClient,
    releases: &ReleasesBucket,
    key: &str,
    query: &[(&str, &str)],
    limit: u64,
) -> Result<Vec<u8>, S3Error> {
    get(http, releases, key, query, |body| {
        let mut bytes = Vec::new();
        body.take(limit)
            .read_to_end(&mut bytes)
            .map_err(|error| S3Error::Interrupted(error.to_string()))?;
        Ok(bytes)
    })
}

/// GET `key` (`""` is the bucket itself) and hand the accepted response's body
/// to `read`. A failed read of the body ends as [`S3Error::Interrupted`],
/// whatever `read` returns; any other error `read` returns is its own.
pub(crate) fn get<T, E: From<S3Error>>(
    http: &HttpClient,
    releases: &ReleasesBucket,
    key: &str,
    query: &[(&str, &str)],
    read: impl FnOnce(&mut dyn Read) -> Result<T, E>,
) -> Result<T, E> {
    let signed = presign(releases, key, query, OffsetDateTime::now_utc(), SIGNED_FOR)?;
    // No redirects: the signature covers the host, and a redirect would hand the
    // signed query to wherever it points.
    let sent = pipette_http::send(http.client_no_redirects().get(&signed.url));
    // Belt and braces: reqwest prints the URL of an error that carries one, and
    // nothing below should carry the others.
    let secrets = [
        signed.url,
        signed.signature,
        releases.secret_access_key.expose().to_owned(),
    ];
    let scrub = |text: String| secret::scrub(text, &secrets);
    let response = sent.map_err(|error| match error {
        pipette_http::Error::Request(error)
        | pipette_http::Error::TimedOut { source: error, .. } => S3Error::Unreachable(scrub(
            format!("{:#}", anyhow::Error::new(error.without_url())),
        )),
        pipette_http::Error::Status { status, body, .. } => S3Error::Refused {
            status,
            reason: refusal_reason(&body).map(scrub),
        },
        other @ (pipette_http::Error::CryptoProviderInstall
        | pipette_http::Error::ClientBuild(_)
        | pipette_http::Error::Json(_)) => S3Error::Unreachable(scrub(other.to_string())),
    })?;
    let mut body = Body {
        response,
        secrets: &secrets,
        broke_off: None,
    };
    let outcome = read(&mut body);
    match body.broke_off {
        Some(reason) => Err(S3Error::Interrupted(reason).into()),
        None => outcome,
    }
}

/// The response body. It keeps why a read of it failed, so [`get`] can tell a
/// broken response from a failure of what the bytes were written to.
struct Body<'a> {
    response: Response,
    secrets: &'a [String],
    broke_off: Option<String>,
}

impl Read for Body<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.response.read(buf).map_err(|error| {
            let reason = secret::scrub(error.to_string(), self.secrets);
            // Every reader retries `Interrupted`, so it ends nothing.
            if error.kind() != io::ErrorKind::Interrupted {
                self.broke_off = Some(reason.clone());
            }
            io::Error::new(error.kind(), reason)
        })
    }
}

/// S3's `Code: Message`. The rest of the body echoes the request, signature
/// included, and is dropped.
fn refusal_reason(body: &str) -> Option<String> {
    let refusal: Refusal = quick_xml::de::from_str(body).ok()?;
    Some(match refusal.message {
        Some(message) => format!("{}: {message}", refusal.code),
        None => refusal.code,
    })
}

/// A presigned URL, and its signature for scrubbing what a failure prints.
struct Presigned {
    url: String,
    signature: String,
}

/// AWS Signature Version 4 in query parameters, for a GET of `key` with `query`
/// beside the signature. Only `host` is signed and the payload is not.
fn presign(
    releases: &ReleasesBucket,
    key: &str,
    query: &[(&str, &str)],
    at: OffsetDateTime,
    valid_for: Duration,
) -> Result<Presigned, S3Error> {
    let at = at.to_offset(time::UtcOffset::UTC);
    let date = format!("{:04}{:02}{:02}", at.year(), u8::from(at.month()), at.day());
    let timestamp = format!(
        "{date}T{:02}{:02}{:02}Z",
        at.hour(),
        at.minute(),
        at.second()
    );
    let region = releases.region.as_str();
    let scope = format!("{date}/{region}/s3/aws4_request");
    let credential = format!("{}/{scope}", releases.access_key_id);
    let expires = valid_for.as_secs().to_string();

    // Sorted by encoded name, then value.
    let mut parameters: Vec<[String; 2]> = query
        .iter()
        .chain(&[
            ("X-Amz-Algorithm", ALGORITHM),
            ("X-Amz-Credential", credential.as_str()),
            ("X-Amz-Date", timestamp.as_str()),
            ("X-Amz-Expires", expires.as_str()),
            ("X-Amz-SignedHeaders", "host"),
        ])
        .map(|(name, value)| [uri_encode(name, false), uri_encode(value, false)])
        .collect();
    parameters.sort();
    let canonical_query = parameters
        .iter()
        .map(|pair| pair.join("="))
        .collect::<Vec<_>>()
        .join("&");

    let encoded_key = uri_encode(key, true);
    let canonical_request = format!(
        "GET\n{}{encoded_key}\n{canonical_query}\nhost:{}\n\nhost\nUNSIGNED-PAYLOAD",
        releases.url.path(),
        releases.url.authority(),
    );
    let string_to_sign = format!(
        "{ALGORITHM}\n{timestamp}\n{scope}\n{}",
        hex::encode(Sha256::digest(canonical_request))
    );
    let signing_key = [date.as_str(), region, "s3", "aws4_request"]
        .iter()
        .try_fold(
            format!("AWS4{}", releases.secret_access_key.expose()).into_bytes(),
            |key, part| hmac_sha256(&key, part),
        )?;
    let signature = hex::encode(hmac_sha256(&signing_key, &string_to_sign)?);

    Ok(Presigned {
        url: format!(
            "{}{encoded_key}?{canonical_query}&X-Amz-Signature={signature}",
            releases.url
        ),
        signature,
    })
}

fn hmac_sha256(key: &[u8], message: &str) -> Result<Vec<u8>, S3Error> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key)?;
    mac.update(message.as_bytes());
    Ok(mac.finalize().into_bytes().to_vec())
}

/// AWS's `UriEncode`: every byte but an unreserved character becomes `%XX`, and
/// `/` survives only in an object key.
fn uri_encode(raw: &str, keep_slash: bool) -> String {
    raw.bytes()
        .fold(String::with_capacity(raw.len()), |mut encoded, byte| {
            match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                    encoded.push(char::from(byte));
                }
                b'/' if keep_slash => encoded.push('/'),
                _ => encoded.push_str(&format!("%{byte:02X}")),
            }
            encoded
        })
}

#[cfg(test)]
mod tests {
    use httpmock::prelude::*;
    use rstest::rstest;
    use time::macros::datetime;
    use url::Url;

    use super::*;
    use crate::releases::testing::{FakeBucket, Published, SECRET};
    use crate::secret::Secret;

    /// The request AWS documents under "Authenticating Requests: Using Query
    /// Parameters (AWS Signature Version 4)", with the signature it publishes.
    #[test]
    fn a_presigned_get_matches_the_example_aws_publishes() -> anyhow::Result<()> {
        let releases = ReleasesBucket {
            url: Url::parse("https://examplebucket.s3.amazonaws.com/")?,
            region: "us-east-1".to_owned(),
            access_key_id: "AKIAIOSFODNN7EXAMPLE".to_owned(),
            secret_access_key: Secret::from("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_owned()),
        };
        let signed = presign(
            &releases,
            "test.txt",
            &[],
            datetime!(2013-05-24 0:00 UTC),
            Duration::from_secs(86400),
        )?;
        assert_eq!(
            signed.signature,
            "aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404"
        );
        assert_eq!(
            signed.url,
            "https://examplebucket.s3.amazonaws.com/test.txt\
             ?X-Amz-Algorithm=AWS4-HMAC-SHA256\
             &X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20130524%2Fus-east-1%2Fs3%2Faws4_request\
             &X-Amz-Date=20130524T000000Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host\
             &X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404"
        );
        Ok(())
    }

    /// The two requests this proxy sends, path-style behind an endpoint with a
    /// port, as botocore 1.43.107 presigns them (`generate_presigned_url` for the
    /// archive, `S3SigV4QueryAuth` on the same URL for the listing page) at
    /// 2026-10-02T12:00:00Z for 900 s.
    #[rstest]
    #[case::archive(
        "branches/feature x/pipette-ios-feature-x.1+b.xcarchive.zip",
        &[],
        "http://127.0.0.1:9000/builds/branches/feature%20x/\
         pipette-ios-feature-x.1%2Bb.xcarchive.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256\
         &X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20261002%2Fus-east-1%2Fs3%2Faws4_request\
         &X-Amz-Date=20261002T120000Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host\
         &X-Amz-Signature=534fa13d43f7c5816ed04606652eeef51e91184ea6c28d85e98d2722cd0437da"
    )]
    #[case::listing_page(
        "",
        &[("list-type", "2"), ("continuation-token", "page/2+of=3")],
        "http://127.0.0.1:9000/builds/?X-Amz-Algorithm=AWS4-HMAC-SHA256\
         &X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20261002%2Fus-east-1%2Fs3%2Faws4_request\
         &X-Amz-Date=20261002T120000Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host\
         &continuation-token=page%2F2%2Bof%3D3&list-type=2\
         &X-Amz-Signature=ac3fdc8346b0c63a2123662220495e82d7c73014e7fac9c7a35f11e91a424fd1"
    )]
    fn a_path_style_request_is_signed_as_botocore_signs_it(
        #[case] key: &str,
        #[case] query: &[(&str, &str)],
        #[case] url: &str,
    ) -> anyhow::Result<()> {
        let signed = presign(
            &FakeBucket::config_at("http://127.0.0.1:9000")?,
            key,
            query,
            datetime!(2026-10-02 12:00 UTC),
            SIGNED_FOR,
        )?;
        assert_eq!(signed.url, url);
        assert!(url.ends_with(&format!("&X-Amz-Signature={}", signed.signature)));
        Ok(())
    }

    /// AWS's `UriEncode`: `/` kept only in a key, every other reserved byte and
    /// all of UTF-8 escaped in upper-case hex.
    #[test]
    fn keys_and_values_are_encoded_the_way_s3_canonicalises_them() -> anyhow::Result<()> {
        assert_eq!(
            uri_encode("branches/x y/a+b=c&d~é.zip", true),
            "branches/x%20y/a%2Bb%3Dc%26d~%C3%A9.zip"
        );
        assert_eq!(uri_encode("1/2+3==", false), "1%2F2%2B3%3D%3D");
        Ok(())
    }

    #[test]
    fn a_listing_is_followed_to_its_last_page() -> anyhow::Result<()> {
        let bucket = FakeBucket::start();
        let published: Vec<Published> = (0..5)
            .map(|n| Published::new(&format!("tag/file-{n}.bin"), b"body", n + 1))
            .collect();
        let pages = bucket.publish(&published, 2);

        let objects = list_objects(&http_client()?, &bucket.config()?)?;

        let keys: Vec<&str> = objects.iter().map(|object| object.key.as_str()).collect();
        assert_eq!(
            keys,
            [
                "tag/file-0.bin",
                "tag/file-1.bin",
                "tag/file-2.bin",
                "tag/file-3.bin",
                "tag/file-4.bin"
            ]
        );
        assert_eq!(objects.first().map(|object| object.bytes), Some(4));
        assert_eq!(
            objects.first().map(|object| object.last_modified),
            Some(datetime!(2026-10-01 12:00 UTC))
        );
        assert_eq!(pages.len(), 3);
        pages.iter().for_each(|page| page.assert_calls(1));
        Ok(())
    }

    #[test]
    fn an_empty_bucket_lists_nothing() -> anyhow::Result<()> {
        let bucket = FakeBucket::start();
        bucket.publish(&[], 1000);
        assert_eq!(list_objects(&http_client()?, &bucket.config()?)?, []);
        Ok(())
    }

    /// A truncated page must name a token no page named before, or the listing
    /// would never end.
    #[rstest]
    #[case::no_token("", 1, "a truncated listing names no NextContinuationToken")]
    #[case::empty_token(
        "<NextContinuationToken></NextContinuationToken>",
        1,
        "a truncated listing names no NextContinuationToken"
    )]
    #[case::same_token(
        "<NextContinuationToken>abc</NextContinuationToken>",
        2,
        "the listing named a continuation token it had already named"
    )]
    fn a_listing_that_does_not_advance_is_refused(
        #[case] token: &str,
        #[case] pages: usize,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let bucket = FakeBucket::start();
        let answered = bucket.server.mock(|when, then| {
            when.method(GET).path("/builds/");
            then.status(200).body(format!(
                "<ListBucketResult><IsTruncated>true</IsTruncated>{token}</ListBucketResult>"
            ));
        });

        let error = list_objects(&http_client()?, &bucket.config()?)
            .err()
            .context("the listing never ends")?;

        assert_eq!(error.to_string(), expected);
        answered.assert_calls(pages);
        Ok(())
    }

    /// What a refused or unanswered request may say: never the secret, the
    /// access key, or any part of the signed query.
    fn assert_names_no_credential(message: &str) {
        [SECRET, FakeBucket::ACCESS_KEY_ID, "X-Amz-"]
            .into_iter()
            .for_each(|needle| {
                assert!(!message.contains(needle), "prints {needle:?}: {message}");
            });
    }

    /// Every answer to a listing that is not one, from S3 or from something
    /// else at its address. A body may echo the request: `{query}` (escaped in
    /// `{xml_query}`), `{signature}`, `{access_key}` and `{secret}`.
    #[rstest]
    #[case::bad_signature(
        403,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Error><Code>SignatureDoesNotMatch</Code>\
         <Message>The request was refused &amp; nothing was read.</Message>\
         <AWSAccessKeyId>{access_key}</AWSAccessKeyId>\
         <SignatureProvided>{signature}</SignatureProvided>\
         <CanonicalRequest>GET\n/builds/\n{xml_query}</CanonicalRequest>\
         <RequestId>4442587FB7D0A2F9</RequestId></Error>",
        "HTTP 403 Forbidden: SignatureDoesNotMatch: The request was refused & nothing was read."
    )]
    #[case::wrong_region(
        301,
        "<Error><Code>PermanentRedirect</Code><Message>Use the bucket's endpoint.</Message>\
         <Endpoint>builds.s3.eu-west-1.amazonaws.com</Endpoint></Error>",
        "HTTP 301 Moved Permanently: PermanentRedirect: Use the bucket's endpoint."
    )]
    #[case::message_echoes_credentials(
        403,
        "<Error><Code>AccessDenied</Code>\
         <Message>signature {signature}, secret {secret}</Message></Error>",
        "HTTP 403 Forbidden: AccessDenied: signature <redacted>, secret <redacted>"
    )]
    #[case::refusal_that_is_not_s3s(
        502,
        "upstream said {secret} to {query}",
        "HTTP 502 Bad Gateway"
    )]
    #[case::captive_portal(
        200,
        "<html>sign in to the network</html>",
        "the answer is not an S3 listing"
    )]
    #[case::echoed_as_json(
        200,
        "{\"args\": \"{query}\", \"signature\": \"{signature}\"}",
        "the answer is not an S3 listing"
    )]
    #[case::echoed_as_text(
        200,
        "no such route: /builds/?{query}",
        "the answer is not an S3 listing"
    )]
    #[case::echoed_inside_xml(
        200,
        "<ListBucketResult><IsTruncated>{xml_query}</IsTruncated></ListBucketResult>",
        "the answer is not an S3 listing"
    )]
    fn an_answer_that_is_no_listing_prints_its_reason_and_no_credential(
        #[case] status: u16,
        #[case] body: &'static str,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let bucket = FakeBucket::start();
        let answered = bucket.server.mock(|when, then| {
            when.method(GET).path("/builds/");
            then.respond_with(move |request| {
                let query = request.uri().query().unwrap_or_default().to_owned();
                let signature = request
                    .query_params()
                    .into_iter()
                    .find(|(name, _)| name == "X-Amz-Signature")
                    .map(|(_, value)| value)
                    .unwrap_or_default();
                HttpMockResponse::builder()
                    .status(status)
                    .body(
                        body.replace("{xml_query}", &query.replace('&', "&amp;"))
                            .replace("{query}", &query)
                            .replace("{signature}", &signature)
                            .replace("{access_key}", FakeBucket::ACCESS_KEY_ID)
                            .replace("{secret}", SECRET),
                    )
                    .build()
            });
        });

        let error = list_objects(&http_client()?, &bucket.config()?)
            .err()
            .context("the answer is no listing")?;

        let message = format!("{:#}", anyhow::Error::new(error));
        assert_eq!(message, expected);
        assert_names_no_credential(&message);
        answered.assert_calls(1);
        Ok(())
    }

    /// A body cut short of its `Content-Length`, as a dropped connection leaves it.
    #[test]
    fn a_body_that_breaks_off_names_no_url() -> anyhow::Result<()> {
        use std::io::Write;

        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let releases = FakeBucket::config_at(&format!("http://{}", listener.local_addr()?))?;
        let server = std::thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            let mut request = [0u8; 4096];
            let _ = stream.read(&mut request)?;
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\nhalf of it")
        });

        let error = get_bytes(&http_client()?, &releases, "tag/archive.zip", &[], u64::MAX)
            .err()
            .context("the body is short")?;
        server
            .join()
            .map_err(|_| anyhow::anyhow!("the server thread panicked"))??;

        assert!(matches!(error, S3Error::Interrupted(_)), "{error:?}");
        let message = error.to_string();
        assert!(
            !message.contains("archive.zip"),
            "prints the URL: {message}"
        );
        assert_names_no_credential(&message);
        Ok(())
    }

    /// A failure of what the body is written to is the reader's own, not a
    /// response that broke off.
    #[test]
    fn a_readers_own_failure_is_not_a_broken_response() -> anyhow::Result<()> {
        let bucket = FakeBucket::start();
        bucket.serve("tag/archive.zip", b"the archive");

        let error = get(
            &http_client()?,
            &bucket.config()?,
            "tag/archive.zip",
            &[],
            |body| -> anyhow::Result<u64> {
                let mut full: &mut [u8] = &mut [];
                io::copy(body, &mut full).context("writing to a full disk")
            },
        )
        .err()
        .context("nothing can be written")?;

        assert_eq!(
            format!("{error:#}"),
            "writing to a full disk: failed to write whole buffer"
        );
        Ok(())
    }

    #[test]
    fn an_unreachable_bucket_names_the_cause_and_no_credential() -> anyhow::Result<()> {
        let port = std::net::TcpListener::bind("127.0.0.1:0")?
            .local_addr()?
            .port();
        let releases = FakeBucket::config_at(&format!("http://127.0.0.1:{port}"))?;

        let error = list_objects(&http_client()?, &releases)
            .err()
            .context("nothing listens there")?;

        assert!(matches!(error, S3Error::Unreachable(_)), "{error:?}");
        let message = format!("{:#}", anyhow::Error::new(error));
        assert!(message.contains("no response: "), "{message}");
        assert!(!message.contains("/builds/"), "prints the URL: {message}");
        assert_names_no_credential(&message);
        Ok(())
    }
}
