//! The management-server HTTP surface, one free function per endpoint.
//!
//! The endpoints are split by the principal the server admits on them:
//! [`client_api`] is what a device calls to do work, signed `v1` as a
//! [`ClientIdentity`]; [`user_api`] is what an operator calls to author it,
//! signed RFC 9421 as a [`UserIdentity`]. Nothing abstracts over the two.
//! The private `call` module holds the pipeline both run through.
//!
//! The public benchmark reads admit anyone, so their identity is optional,
//! signed when the caller holds one since what the catalog shows can depend
//! on who asks. They live in [`client_api`] because a device's sync and claim
//! paths call them.

pub mod auth;
mod call;
pub mod client_api;
pub mod error;
mod httpsig;
pub mod transport;
pub mod types;
pub mod user_api;

pub use auth::{
    ClientIdentity, KeyOnlyIdentity, SIGNATURE_LEN, SignsPayload, UserIdentity, UserSigner,
    generate_keypair, key_id_for, public_key_hex_from_private,
};
pub use call::{ByteStream, ConditionalResponse, ImportProgress, RequestBody, ResumeFrom};
pub use error::{Error, Result};
pub use pipette_http::range::ContentRange;
pub use transport::{HttpMethod, HttpRequest, HttpResponse, TransportError};
pub use types::{
    AddUserKeyRequest, AddUserKeyResponse, ClientPreauthKey, ClientStatus, ContactEmail,
    ContactEmailParseError, EmptyClientPreauthKeyError, EmptyKeyDetailsError,
    EmptyUserPreauthKeyError, EntityTag, IfNoneMatch, JobStatus, KeyDetails, PrivateKey,
    PrivateKeyError, PublicKey, PublicKeyError, RegisterUserRequest, RegisterUserResponse,
    ServerUrl, ServerUrlError, Timestamp, TimestampError, UserKey, UserKeys, UserPreauthKey,
    UserProfile, UserStatus, VaultResource, parse_contact_email,
};
