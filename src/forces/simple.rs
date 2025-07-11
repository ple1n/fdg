//! Simple [`Force`]s used to transform the graph.

use nalgebra::SVector;
use petgraph::{csr::IndexType, EdgeType};

use crate::{Field, Force, ForceGraph};

/// Scale all node positions by a scalar [`Field`] about the origin.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Scale<T: Field> {
    pub factor: T,
}

impl<T: Field, const D: usize, N, E, Ty, Ix> Force<T, D, N, E, Ty, Ix> for Scale<T>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    fn apply(&mut self, graph: &mut ForceGraph<T, D, N, E, Ty, Ix>) {
        graph
            .node_weights_mut()
            .for_each(|(_, pos)| pos.coords.scale_mut(self.factor));
    }
}

/// Translate each node by a [`SVector`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Translate<T: Field, const D: usize> {
    pub translation: SVector<T, D>,
}

impl<T: Field, const D: usize> Translate<T, D> {
    /// Create a new [`Translate`] with a given translation.
    pub fn new(translation: SVector<T, D>) -> Self {
        Self { translation }
    }
}

impl<T: Field, const D: usize, N, E, Ty, Ix> Force<T, D, N, E, Ty, Ix> for Translate<T, D>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    fn apply(&mut self, graph: &mut ForceGraph<T, D, N, E, Ty, Ix>) {
        graph
            .node_weights_mut()
            .for_each(|(_, pos)| pos.coords += self.translation);
    }
}

/// Center all nodes around the origin by translating their average position to the origin.
#[derive(Default, Copy, Clone, Debug, PartialEq, Eq)]
pub struct Center;

impl<T: Field, const D: usize, N, E, Ty, Ix> Force<T, D, N, E, Ty, Ix> for Center
where
    Ty: EdgeType,
    Ix: IndexType,
{
    fn apply(&mut self, graph: &mut ForceGraph<T, D, N, E, Ty, Ix>) {
        let avg: SVector<T, D> = graph
            .node_weights()
            .map(|(_, p)| p.coords)
            .sum::<SVector<T, D>>()
            .scale(T::from(graph.node_count()).unwrap().recip());

        graph.node_weights_mut().for_each(|(_, p)| *p -= avg);
    }
}
