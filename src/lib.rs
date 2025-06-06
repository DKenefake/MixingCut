#![warn(clippy::all, clippy::cargo, clippy::nursery)]
#![allow(non_snake_case)]
#![allow(dead_code)]

pub mod initialize;
pub mod sdp_local_search;
pub mod sdp_project;
pub mod step_rules;

pub mod maxcut_oracle;

pub mod io_operations;

pub mod sdp_solver;
