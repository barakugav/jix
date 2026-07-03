use crate::arrayvec::ArrayVec;
use crate::ArrayStorage;

/// Introspection metadata for an [`ArrayStorage`], returned by
/// [`ArrayStorage::info`](crate::ArrayStorage::info).
///
/// Pairs a short human-readable name for the storage node with references to the storages it
/// depends on. Chaining these across a lazy pipeline yields a tree describing the whole operation.
pub struct ArrayStorageInfo<'a> {
    name: &'a str,
    dependencies: ArrayDependencies<'a>,
}
impl<'a> ArrayStorageInfo<'a> {
    #[inline]
    pub(crate) fn new(name: &'a str) -> Self {
        Self {
            name,
            dependencies: ArrayDependencies::Inline(ArrayVec::new()),
        }
    }
    #[inline]
    fn new_deps_inline(name: &'a str, deps: &[&'a dyn ArrayStorage]) -> Self {
        debug_assert!(deps.len() <= 4);
        let mut deps_vec = ArrayVec::new();
        for dep in deps.iter() {
            deps_vec.push(*dep);
        }
        Self {
            name,
            dependencies: ArrayDependencies::Inline(deps_vec),
        }
    }
    #[inline]
    pub(crate) fn new_deps<const N: usize>(name: &'a str, deps: [&'a dyn ArrayStorage; N]) -> Self {
        if N <= 4 {
            Self::new_deps_inline(name, &deps)
        } else {
            Self {
                name,
                dependencies: ArrayDependencies::Vec(deps.to_vec()),
            }
        }
    }

    #[inline]
    pub(crate) fn new_deps_dyn(name: &'a str, deps: Vec<&'a dyn ArrayStorage>) -> Self {
        if deps.len() <= 4 {
            Self::new_deps_inline(name, &deps)
        } else {
            Self {
                name,
                dependencies: ArrayDependencies::Vec(deps),
            }
        }
    }

    /// The short, human-readable name of this storage node (e.g. `"Compact"`, `"Slice"`).
    pub fn name(&self) -> &'a str {
        self.name
    }

    /// The storages this node reads from, in argument order. Empty for leaf storages.
    pub fn dependencies(&self) -> &[&'a dyn ArrayStorage] {
        match &self.dependencies {
            ArrayDependencies::Inline(deps) => deps.as_slice(),
            ArrayDependencies::Vec(deps) => deps.as_slice(),
        }
    }
}
pub(crate) enum ArrayDependencies<'a> {
    Inline(ArrayVec<&'a dyn ArrayStorage, 4>),
    Vec(Vec<&'a dyn ArrayStorage>),
}
