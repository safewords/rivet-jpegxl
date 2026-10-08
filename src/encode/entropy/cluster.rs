//! Histogram clustering: contexts with similar statistics share a
//! histogram, which costs less to send than a histogram each.

/// The most histograms a context map can address.
pub(super) const MAX_CLUSTERS: usize = 256;
/// An estimate of a histogram's header, per symbol it uses.
const HEADER_BITS_PER_SYMBOL: f64 = 5.0;
/// An estimate of a histogram's fixed header.
const HEADER_BITS: f64 = 16.0;

/// Bits to code `h` with its own histogram, header included.
pub(super) fn cost(h: &[u64]) -> f64 {
    let total: u64 = h.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    let mut bits = HEADER_BITS;
    for &c in h {
        if c > 0 {
            bits += c as f64 * (t / c as f64).log2() + HEADER_BITS_PER_SYMBOL;
        }
    }
    bits
}

fn merged(a: &[u64], b: &[u64]) -> Vec<u64> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| a.get(i).copied().unwrap_or(0) + b.get(i).copied().unwrap_or(0))
        .collect()
}

/// A context map grouping `histograms` (one per context) into at most
/// `max_clusters` clusters, numbered in order of first use. Empty contexts
/// join cluster 0.
pub(super) fn cluster(histograms: &[Vec<u64>], max_clusters: usize) -> Vec<u8> {
    let max_clusters = max_clusters.clamp(1, MAX_CLUSTERS);
    let used: Vec<usize> = (0..histograms.len())
        .filter(|&c| histograms[c].iter().any(|&n| n > 0))
        .collect();
    if used.is_empty() {
        return vec![0; histograms.len()];
    }

    // Seed: identical histograms are one cluster; then, while there are too
    // many for the pairwise pass, add contexts to the nearest of a
    // farthest-first set of centres.
    let mut clusters: Vec<(Vec<u64>, Vec<usize>)> = Vec::new();
    {
        let mut by_content: std::collections::HashMap<&[u64], usize> = Default::default();
        for &c in &used {
            let h = trim(&histograms[c]);
            if let Some(&k) = by_content.get(h) {
                clusters[k].1.push(c);
                let sum = merged(&clusters[k].0, h);
                clusters[k].0 = sum;
            } else {
                by_content.insert(h, clusters.len());
                clusters.push((h.to_vec(), vec![c]));
            }
        }
    }
    const PAIRWISE_LIMIT: usize = 64;
    if clusters.len() > PAIRWISE_LIMIT {
        clusters = farthest_first(clusters, PAIRWISE_LIMIT);
    }

    // Merge the pair whose merge saves the most, while any saves, or while
    // there are too many.
    let mut costs: Vec<f64> = clusters.iter().map(|(h, _)| cost(h)).collect();
    loop {
        let mut best: Option<(f64, usize, usize)> = None;
        for i in 0..clusters.len() {
            for j in i + 1..clusters.len() {
                let gain = cost(&merged(&clusters[i].0, &clusters[j].0)) - costs[i] - costs[j];
                if best.is_none_or(|(g, _, _)| gain < g) {
                    best = Some((gain, i, j));
                }
            }
        }
        let Some((gain, i, j)) = best else { break };
        if gain >= 0.0 && clusters.len() <= max_clusters {
            break;
        }
        let (hj, cj) = clusters.remove(j);
        costs.remove(j);
        clusters[i].0 = merged(&clusters[i].0, &hj);
        clusters[i].1.extend(cj);
        costs[i] = cost(&clusters[i].0);
    }

    // Number clusters by first use.
    let mut of_context = vec![usize::MAX; histograms.len()];
    for (k, (_, members)) in clusters.iter().enumerate() {
        for &c in members {
            of_context[c] = k;
        }
    }
    let mut renumber = vec![usize::MAX; clusters.len()];
    let mut next = 0;
    let mut map = vec![0u8; histograms.len()];
    for c in 0..histograms.len() {
        let k = of_context[c];
        if k == usize::MAX {
            continue;
        }
        if renumber[k] == usize::MAX {
            renumber[k] = next;
            next += 1;
        }
        map[c] = renumber[k] as u8;
    }
    map
}

fn trim(h: &[u64]) -> &[u64] {
    let end = h.iter().rposition(|&c| c > 0).map_or(0, |p| p + 1);
    &h[..end]
}

/// At most `k` clusters: centres picked farthest-first by the cost of
/// joining, every cluster then joined to its cheapest centre.
fn farthest_first(clusters: Vec<(Vec<u64>, Vec<usize>)>, k: usize) -> Vec<(Vec<u64>, Vec<usize>)> {
    let join_cost = |a: &[u64], b: &[u64]| cost(&merged(a, b)) - cost(a) - cost(b);
    // Start from the largest.
    let first = (0..clusters.len())
        .max_by_key(|&i| clusters[i].0.iter().sum::<u64>())
        .unwrap();
    let mut centres = vec![first];
    let mut nearest: Vec<f64> = clusters
        .iter()
        .map(|(h, _)| join_cost(h, &clusters[first].0))
        .collect();
    while centres.len() < k {
        let (far, &d) = nearest
            .iter()
            .enumerate()
            .filter(|(i, _)| !centres.contains(i))
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        if d <= 0.0 {
            break;
        }
        centres.push(far);
        for (i, (h, _)) in clusters.iter().enumerate() {
            nearest[i] = nearest[i].min(join_cost(h, &clusters[far].0));
        }
    }
    let mut out: Vec<(Vec<u64>, Vec<usize>)> = centres
        .iter()
        .map(|&c| (clusters[c].0.clone(), clusters[c].1.clone()))
        .collect();
    for (i, (h, members)) in clusters.into_iter().enumerate() {
        if centres.contains(&i) {
            continue;
        }
        let best = (0..out.len())
            .min_by(|&a, &b| join_cost(&h, &out[a].0).total_cmp(&join_cost(&h, &out[b].0)))
            .unwrap();
        out[best].0 = merged(&out[best].0, &h);
        out[best].1.extend(members);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similar_contexts_share_and_numbers_follow_first_use() {
        let a = vec![100, 50, 1, 0];
        let b = vec![0, 0, 3, 900];
        let histograms = vec![
            b.clone(),
            a.clone(),
            vec![],
            a.clone(),
            b,
            vec![101, 49, 1, 0],
        ];
        let map = cluster(&histograms, 256);
        assert_eq!(map[0], 0);
        assert_eq!(map[1], 1);
        assert_eq!(map[3], 1);
        assert_eq!(map[4], 0);
        assert_eq!(map[5], 1);
    }

    #[test]
    fn the_cap_is_kept() {
        let histograms: Vec<Vec<u64>> = (0..300)
            .map(|i| {
                let mut h = vec![0u64; 40];
                h[i % 40] = 1000;
                h[(i * 7) % 40] += 10;
                h
            })
            .collect();
        let map = cluster(&histograms, 8);
        assert!(map.iter().all(|&k| k < 8));
        let max = *map.iter().max().unwrap() as usize;
        for k in 0..=max {
            assert!(map.contains(&(k as u8)));
        }
    }
}
