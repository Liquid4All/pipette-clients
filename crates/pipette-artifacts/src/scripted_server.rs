//! A loopback HTTP server that answers each connection as a script says: it
//! drops the connection, cuts the body short, or answers in full.
//!
//! httpmock answers every request whole, so it cannot stage the transport
//! failures that a download retries.

use std::io::{self, BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

/// What the server does with one connection.
pub(crate) enum Reply {
    /// Read the request, then close the connection with no response.
    Drop,
    /// Answer `status` with `headers` and declare the whole `body`, but send
    /// only its first `sent` bytes before the connection closes.
    Cut {
        status: u16,
        headers: Vec<(&'static str, String)>,
        body: Vec<u8>,
        sent: usize,
    },
    /// Answer `200` with `headers` and a chunked `body` of no declared length.
    /// Send its first `sent` bytes as one chunk, then close the connection
    /// before the last chunk.
    CutChunked {
        headers: Vec<(&'static str, String)>,
        body: Vec<u8>,
        sent: usize,
    },
    /// Answer in full.
    Full {
        status: u16,
        headers: Vec<(&'static str, String)>,
        body: Vec<u8>,
    },
}

impl Reply {
    pub(crate) fn ok(body: &[u8]) -> Self {
        Self::Full {
            status: 200,
            headers: Vec::new(),
            body: body.to_vec(),
        }
    }

    pub(crate) fn cut(body: &[u8], sent: usize) -> Self {
        Self::Cut {
            status: 200,
            headers: Vec::new(),
            body: body.to_vec(),
            sent,
        }
    }

    pub(crate) fn status(status: u16) -> Self {
        Self::Full {
            status,
            headers: Vec::new(),
            body: Vec::new(),
        }
    }
}

/// The server, alive until the script runs out. Each request head is kept, in
/// order of arrival.
pub(crate) struct ScriptedServer {
    addr: SocketAddr,
    requests: Arc<Mutex<Vec<String>>>,
}

impl ScriptedServer {
    /// Serve one connection per entry of `script`, then stop listening.
    pub(crate) fn start(script: Vec<Reply>) -> io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&requests);
        std::thread::spawn(move || {
            script.into_iter().try_for_each(|reply| {
                let (stream, _) = listener.accept()?;
                serve(stream, reply, &log)
            })
        });
        Ok(Self { addr, requests })
    }

    pub(crate) fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }

    /// The request heads received so far.
    pub(crate) fn requests(&self) -> Vec<String> {
        self.requests
            .lock()
            .map(|requests| requests.clone())
            .unwrap_or_default()
    }
}

fn serve(stream: TcpStream, reply: Reply, log: &Mutex<Vec<String>>) -> io::Result<()> {
    let mut reader = BufReader::new(stream);
    let mut head = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 || line == "\r\n" {
            break;
        }
        head.push_str(&line);
    }
    if let Ok(mut requests) = log.lock() {
        requests.push(head);
    }
    let mut stream = reader.into_inner();
    match reply {
        Reply::Drop => Ok(()),
        Reply::Cut {
            status,
            headers,
            body,
            sent,
        } => {
            write_head(&mut stream, status, &headers, Some(body.len()))?;
            stream.write_all(body.get(..sent).unwrap_or(&body))?;
            stream.flush()
        }
        Reply::CutChunked {
            headers,
            body,
            sent,
        } => {
            write_head(&mut stream, 200, &headers, None)?;
            let chunk = body.get(..sent).unwrap_or(&body);
            stream.write_all(format!("{:x}\r\n", chunk.len()).as_bytes())?;
            stream.write_all(chunk)?;
            stream.write_all(b"\r\n")?;
            stream.flush()
        }
        Reply::Full {
            status,
            headers,
            body,
        } => {
            write_head(&mut stream, status, &headers, Some(body.len()))?;
            stream.write_all(&body)?;
            stream.flush()
        }
    }
}

fn write_head(
    stream: &mut TcpStream,
    status: u16,
    headers: &[(&'static str, String)],
    length: Option<usize>,
) -> io::Result<()> {
    let framing = length.map_or_else(
        || "transfer-encoding: chunked".to_owned(),
        |length| format!("content-length: {length}"),
    );
    let mut head = format!("HTTP/1.1 {status} Scripted\r\n{framing}\r\nconnection: close\r\n");
    headers
        .iter()
        .for_each(|(name, value)| head.push_str(&format!("{name}: {value}\r\n")));
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())
}
