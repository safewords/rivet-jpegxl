//! jxl-rs's parallel runner on scoped std threads: no thread pool crate.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use jxl::api::{JxlParallelRunner, JxlParallelRunnerFun};

/// Runs a batch of independent tasks over up to `threads` scoped threads,
/// each taking the next index until none are left.
pub(crate) struct ThreadRunner {
    threads: usize,
}

impl ThreadRunner {
    /// `threads` workers; 0 is one per available core.
    pub(crate) fn new(threads: usize) -> Self {
        let threads = if threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            threads
        };
        Self { threads }
    }

    /// Whether running on this one is worth more than the caller's thread.
    pub(crate) fn is_parallel(&self) -> bool {
        self.threads > 1
    }
}

impl JxlParallelRunner for ThreadRunner {
    fn run(&mut self, num: usize, fun: &JxlParallelRunnerFun<'_>) -> jxl::error::Result<()> {
        let workers = self.threads.min(num);
        if workers <= 1 {
            return (0..num).try_for_each(fun);
        }
        let next = AtomicUsize::new(0);
        let failed: Mutex<Option<jxl::error::Error>> = Mutex::new(None);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= num {
                            return;
                        }
                        if let Err(e) = fun(i) {
                            failed.lock().unwrap().get_or_insert(e);
                            // The rest of the batch is not worth running.
                            next.store(num, Ordering::Relaxed);
                            return;
                        }
                    }
                });
            }
        });
        match failed.into_inner().unwrap() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn num_threads(&self) -> usize {
        self.threads
    }
}
