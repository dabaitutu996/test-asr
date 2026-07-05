//! TUI 界面层：启动流程各交互屏（screens）与主循环渲染（render）。

pub(crate) mod render;
pub(crate) mod screens;

pub(crate) use render::draw;
pub(crate) use screens::{run_device_screen, run_preview_screen, run_selection_screen};
