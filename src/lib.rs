#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]
#![deny(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use nalgebra::{
    ClosedAdd, ClosedDiv, ClosedMul, ClosedSub, Const, OVector, Point, Scalar, SimdRealField,
};
use num_traits::Float;
use petgraph::{
    csr::IndexType,
    stable_graph::{DefaultIx, StableGraph},
    Directed, EdgeType,
};
use rand::distributions::{uniform::SampleUniform, Distribution, Uniform};

mod forces;

#[doc(inline)]
pub use forces::*;

pub use nalgebra;
pub use petgraph;
pub use rand::distributions as rand_distributions;

/// A simple wrapper around [`petgraph`]'s [`StableGraph`] which contains your node data as well as its corresponding position.
pub type ForceGraph<T, const D: usize, N, E, Ty = Directed, Ix = DefaultIx> =
    StableGraph<(N, Point<T, D>), E, Ty, Ix>;

pub type ForceGraphNode<T, const D: usize, N> = (N, Point<T, D>);

/// A generic force trait representing an iterable transformation on a [`ForceGraph`].
///
/// T: Coordinate type ([`Field`])
/// D: Number of dimensions (usize)
/// N: Node weight type
/// E: Edge weight type
pub trait Force<T: Field, const D: usize, N, E, Ty, Ix> {
    /// Apply the force to a given graph.
    fn apply(&mut self, graph: &mut ForceGraph<T, D, N, E, Ty, Ix>);

    /// Apply the force to a given graph `n` times.
    fn apply_many(&mut self, graph: &mut ForceGraph<T, D, N, E, Ty, Ix>, n: usize) {
        for _ in 0..n {
            Self::apply(self, graph);
        }
    }
}

impl<F, T: Field, const D: usize, N, E, Ty, Ix> Force<T, D, N, E, Ty, Ix> for F
where
    F: Fn(&mut ForceGraph<T, D, N, E, Ty, Ix>),
{
    fn apply(&mut self, graph: &mut ForceGraph<T, D, N, E, Ty, Ix>) {
        self(graph);
    }
}

/// Create a [`ForceGraph`] from any graph and randomize the node positions within a uniform distribution around the origin in `-range` to `range`.
pub fn init_force_graph_uniform<
    T: Field + SampleUniform,
    const D: usize,
    N: Clone,
    E: Clone,
    Ty: EdgeType,
    Ix: IndexType,
>(
    input: impl Into<StableGraph<N, E, Ty, Ix>>,
    range: T,
) -> ForceGraph<T, D, N, E, Ty, Ix> {
    init_force_graph(input, Uniform::new(-range, range))
}

/// Create a [`ForceGraph`] from any graph and randomize the node positions within a given distribution.
pub fn init_force_graph<
    T: Field,
    const D: usize,
    N: Clone,
    E: Clone,
    Ty: EdgeType,
    Ix: IndexType,
>(
    input: impl Into<StableGraph<N, E, Ty, Ix>>,
    distribution: impl Distribution<T>,
) -> ForceGraph<T, D, N, E, Ty, Ix> {
    let mut graph = input.into().map(
        |_, node| (node.clone(), Point::default()),
        |_, edge| edge.clone(),
    );

    randomize_positions(&mut graph, distribution);

    graph
}

/// Randomize all the node positions in a [`ForceGraph`] with a given distribution.
///
/// This is helpful for generating starting positions.
pub fn randomize_positions<T: Field, const D: usize, N, E, Ty, Ix>(
    graph: &mut ForceGraph<T, D, N, E, Ty, Ix>,
    distribution: impl Distribution<T>,
) where
    Ty: EdgeType,
    Ix: IndexType,
{
    let mut rng = rand::thread_rng();

    for (_, pos) in graph.node_weights_mut() {
        *pos = Point::from(OVector::from_distribution_generic(
            Const::<D>,
            Const::<1>,
            &distribution,
            &mut rng,
        ));
    }
}

/// A composite trait for node positions that covers both [`f32`] and [`f64`] floating-point types.
pub trait Field:
    SimdRealField + Float + Scalar + ClosedMul + ClosedDiv + ClosedAdd + ClosedSub + Send + Sync
{
}
impl<T> Field for T where
    T: SimdRealField + Float + Scalar + ClosedMul + ClosedDiv + ClosedAdd + ClosedSub + Send + Sync
{
}
