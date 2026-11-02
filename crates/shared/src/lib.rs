//! Core AI Designer engine: blueprint specification, analysis, advisor, code generation,
//! importers, file formats, and bundled starter templates.

pub mod advisor;
pub mod analysis;
pub mod codegen;
pub mod format;
pub mod importer;
pub mod spec;
pub mod surgery;
pub mod templates;
pub mod training;
pub mod workspace;

pub use advisor::*;
pub use analysis::*;
pub use codegen::*;
pub use format::*;
pub use importer::*;
pub use spec::*;
pub use surgery::*;
pub use templates::*;
pub use training::*;
pub use workspace::*;

pub use spodeian_tokens::ThemeMode;
