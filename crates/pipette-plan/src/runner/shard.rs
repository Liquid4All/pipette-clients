//! `--shard INDEX/COUNT` — partition a plan's cells across independent
//! invocations (e.g. one per SLURM job).

use std::str::FromStr;

/// A round-robin slice of the matrix: this invocation handles the cells
/// whose position in the deterministically-ordered cell list is
/// congruent to `index` modulo `count`.
///
/// Sharding is by position in the *full* ordered matrix, computed before
/// any state filter, so a cell's owning shard never changes as other
/// cells complete — splitting work across jobs stays stable run-to-run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shard {
    pub index: usize,
    pub count: usize,
}

impl Shard {
    /// Whether the cell at ordered position `idx` belongs to this shard.
    pub fn contains(&self, idx: usize) -> bool {
        idx % self.count == self.index
    }
}

impl FromStr for Shard {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (i, n) = s
            .split_once('/')
            .ok_or_else(|| format!("expected INDEX/COUNT, got '{s}'"))?;
        let index = i
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("invalid shard index '{i}'"))?;
        let count = n
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("invalid shard count '{n}'"))?;
        if count == 0 {
            return Err("shard count must be >= 1".to_string());
        }
        if index >= count {
            return Err(format!("shard index {index} must be < count {count}"));
        }
        Ok(Shard { index, count })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_shard() -> Result<(), String> {
        assert_eq!("0/4".parse::<Shard>()?, Shard { index: 0, count: 4 });
        assert_eq!("3/4".parse::<Shard>()?, Shard { index: 3, count: 4 });
        Ok(())
    }

    #[test]
    fn rejects_malformed() {
        assert!("4".parse::<Shard>().is_err()); // no slash
        assert!("a/4".parse::<Shard>().is_err()); // non-numeric index
        assert!("0/x".parse::<Shard>().is_err()); // non-numeric count
        assert!("0/0".parse::<Shard>().is_err()); // zero count
        assert!("4/4".parse::<Shard>().is_err()); // index == count
        assert!("5/4".parse::<Shard>().is_err()); // index > count
    }

    #[test]
    fn shards_partition_every_index_exactly_once() {
        let count = 4;
        (0..100usize).for_each(|idx| {
            let owners: Vec<usize> = (0..count)
                .filter(|&index| Shard { index, count }.contains(idx))
                .collect();
            assert_eq!(owners, vec![idx % count], "idx {idx} must have one owner");
        });
    }

    #[test]
    fn count_one_owns_everything() {
        let shard = Shard { index: 0, count: 1 };
        assert!((0..50).all(|idx| shard.contains(idx)));
    }
}
