//! Model Surgery, Abliteration & Model Soups Engine.
//! Provides mathematically rigorous implementations for directional weight ablation,
//! LEACE concept erasure, contrastive activation steering, Model Soups (Uniform, Greedy,
//! Fisher), SLERP hyperspherical interpolation, Task Arithmetic, TIES-Merging, DARE,
//! and Frankenmerge layer grafting.

pub mod abliteration;
pub mod grafting;
pub mod soups;

pub use abliteration::*;
pub use grafting::*;
pub use soups::*;
