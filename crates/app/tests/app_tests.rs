use app::{
    AiDesignerApp, AppPage, BlueprintExportFormat, TemplateApp, bi, install_bootstrap_icons,
    storage_manager::{DEDICATED_STORAGE_KEY, deserialize_app_state, load_state_multi_tier},
};
use eframe::{App, Storage};
use shared::{AiDesignSpec, BlueprintFileFormat, ThemeMode};
use std::collections::HashMap;

#[derive(Default)]
struct MockStorage {
    data: HashMap<String, String>,
}

impl Storage for MockStorage {
    fn get_string(&self, key: &str) -> Option<String> {
        self.data.get(key).cloned()
    }
    fn set_string(&mut self, key: &str, value: String) {
        self.data.insert(key.to_owned(), value);
    }
    fn remove_string(&mut self, key: &str) {
        self.data.remove(key);
    }
    fn flush(&mut self) {}
}

#[test]
fn test_designer_app_initialization() {
    let app = AiDesignerApp::default();
    assert_eq!(app.state.active_page, AppPage::Designer);
    assert!(!app.studio.spec.graph.nodes.is_empty());
    assert_eq!(app.studio.pareto_candidates.len(), 5);
}

#[test]
fn test_designer_app_save_and_load() {
    let mut storage = MockStorage::default();
    let mut app = TemplateApp::default();

    // Modify active spec and theme
    app.studio.spec.metadata.display_name = "Modified Spatiotemporal Flow".to_string();
    app.state.theme = ThemeMode::Light;

    // Save
    app.save(&mut storage);

    // Verify storage contents
    let serialized = storage
        .get_string(eframe::APP_KEY)
        .expect("App key must exist in storage");
    assert!(serialized.contains("Modified Spatiotemporal Flow"));

    // Reload from storage
    let loaded_state = load_state_multi_tier(Some(&storage)).expect("Should reload state");
    assert_eq!(loaded_state.theme, ThemeMode::Light);
    let restored_spec = loaded_state.saved_spec.expect("Saved spec must exist");
    assert_eq!(
        restored_spec.metadata.display_name,
        "Modified Spatiotemporal Flow"
    );
}

#[test]
fn test_designer_app_export_dialog() {
    let mut app = AiDesignerApp::default();
    app.open_export_dialog(BlueprintExportFormat::Yaml);

    assert_eq!(app.show_export_dialog, Some(BlueprintExportFormat::Yaml));
    assert!(app.export_text_buffer.contains("metadata:"));

    app.open_export_dialog(BlueprintExportFormat::Json);
    assert_eq!(app.show_export_dialog, Some(BlueprintExportFormat::Json));
    assert!(app.export_text_buffer.contains("\"metadata\":"));
}

#[test]
fn test_designer_app_load_multi_tier_json_and_ron() {
    let original_state = app::AppState {
        theme: ThemeMode::HighContrastDark,
        saved_spec: Some(AiDesignSpec::new("custom_test_model", "Custom Test Model")),
        ..Default::default()
    };

    // 1. JSON roundtrip
    let json_str = serde_json::to_string(&original_state).unwrap();
    let from_json = deserialize_app_state(&json_str).expect("JSON should deserialize");
    assert_eq!(from_json.theme, ThemeMode::HighContrastDark);
    assert_eq!(
        from_json.saved_spec.unwrap().metadata.model_id,
        "custom_test_model"
    );

    // 2. RON roundtrip
    let ron_str = ron::to_string(&original_state).unwrap();
    let from_ron = deserialize_app_state(&ron_str).expect("RON should deserialize");
    assert_eq!(from_ron.theme, ThemeMode::HighContrastDark);

    // 3. Load from dedicated key in storage (JSON)
    let mut storage_dedicated = MockStorage::default();
    storage_dedicated.set_string(DEDICATED_STORAGE_KEY, json_str.clone());
    let loaded_dedicated =
        load_state_multi_tier(Some(&storage_dedicated)).expect("Should load from dedicated key");
    assert_eq!(loaded_dedicated.theme, ThemeMode::HighContrastDark);
}

#[test]
fn test_studio_workflow_layer_manipulation_and_pareto() {
    let mut app = AiDesignerApp::default();

    // Pareto frontier candidates should be pre-computed on initialization
    assert_eq!(app.studio.pareto_candidates.len(), 5);
    let initial_len = app.studio.spec.graph.nodes.len();
    assert!(initial_len >= 2);

    // Select first node and move it down
    let first_id = app.studio.spec.graph.nodes[0].id.clone();
    let second_id = app.studio.spec.graph.nodes[1].id.clone();
    app.studio.select_node(Some(first_id.clone()));
    assert!(app.studio.move_selected_node(1));
    assert_eq!(app.studio.spec.graph.nodes[0].id, second_id);
    assert_eq!(app.studio.spec.graph.nodes[1].id, first_id);

    // Duplicate selected node
    let dup_id = app
        .studio
        .duplicate_selected_node()
        .expect("Should duplicate selected node");
    assert_eq!(app.studio.spec.graph.nodes.len(), initial_len + 1);
    assert_eq!(
        app.studio.selected_node_id.as_deref(),
        Some(dup_id.as_str())
    );

    // Delete duplicated node
    assert!(app.studio.delete_selected_node());
    assert_eq!(app.studio.spec.graph.nodes.len(), initial_len);
    assert!(app.studio.selected_node_id.is_none());

    // Auto-layout resets zoom and pan and positions Input, Layer, and Output cards
    app.studio.zoom_level = 1.75;
    app.studio.auto_layout_nodes();
    assert_eq!(app.studio.zoom_level, 1.0);
    assert!(
        app.studio
            .node_positions
            .keys()
            .any(|k| k.starts_with("__input__:"))
    );
    assert!(
        app.studio
            .node_positions
            .keys()
            .any(|k| k.starts_with("__output__:"))
    );
}

#[test]
fn test_bootstrap_icons_and_operator_overviews() {
    let ctx = eframe::egui::Context::default();
    install_bootstrap_icons(&ctx);
    assert!(!app::icons::BOOTSTRAP_ICONS_TTF.is_empty());
    assert_eq!(bi::LIGHTNING_CHARGE_FILL, "\u{f46c}");

    // Verify operator overview, recommended next layers, and clickable curriculum URLs
    let app = AiDesignerApp::default();
    for node in &app.studio.spec.graph.nodes {
        let ov = node.operator.overview();
        assert!(!ov.what_it_does.is_empty());
        assert!(!ov.typical_usage.is_empty());
        assert!(!ov.typically_followed_by.is_empty());
        assert!(
            node.operator
                .curriculum_url()
                .starts_with("https://github.com/")
        );
        assert!(!node.operator.recommended_next_layers().is_empty());
    }

    // Verify file extensions
    assert_eq!(BlueprintFileFormat::NativeAid.default_extension(), "aid");
    assert_eq!(BlueprintFileFormat::Yaml.default_extension(), "yaml");
}
