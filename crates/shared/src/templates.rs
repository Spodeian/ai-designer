//! Starter and industry blueprints bundled into the binary at compile time (from `templates/`).

pub struct StarterTemplate {
    pub file_name: &'static str,
    pub display_name: &'static str,
    pub yaml_content: &'static str,
}

impl StarterTemplate {
    /// Returns the primary `AI-Examples` curriculum category ID associated with this blueprint.
    pub fn category_id(&self) -> &'static str {
        match self.file_name {
            "01_spatiotemporal_dit_flow.yaml" => "01_paradigms_and_dichotomies",
            "02_bimanual_diffusion_act_policy.yaml" => "07_domain_tasks",
            "03_streaming_codec_mamba.yaml" => "17_deployment_low_level_and_capstones",
            "04_symplectic_fourier_neural_operator.yaml" => "06_smooth_and_physics_informed_ai",
            "05_equivariant_biomolecular_flow.yaml" => "07_domain_tasks",
            "06_patch_xlstm_probabilistic_forecaster.yaml" => "05_architectures",
            "07_monotonic_ple_tabpfn.yaml" => "02_classical_and_probabilistic_ml",
            "08_hybrid_jamba_mla_verifier.yaml" => "05_architectures",
            "09_autograd_custom_kernel_backward.yaml" => "00_foundations",
            "10_optimal_transport_sinkhorn_matcher.yaml" => "03_building_blocks_and_losses",
            "11_muon_wsd_accelerated_transformer.yaml" => "04_training_methods_and_optimisers",
            "12_safe_cbf_qp_diffusion_policy.yaml" => "08_learning_paradigms_rl_and_alignment",
            "13_structural_causal_neural_scm.yaml" => "09_causal_ai_and_evolutionary_search",
            "14_monosemantic_sparse_autoencoder_sae.yaml" => "10_xai_and_ai_analysis_methods",
            "15_differentially_private_dp_sgd.yaml" => "11_ai_safety_privacy_and_verification",
            "16_torch_compile_inductor_fastpath.yaml" => "12_compilation_and_performance",
            "17_bitnet_ternary_1_58_quant.yaml" => "13_precision_and_numerics",
            "18_awq_activation_aware_quantizer.yaml" => "14_quantisation",
            "19_speculative_medusa_draft_decoder.yaml" => {
                "15_compression_peft_and_speculative_inference"
            }
            "20_fsdp2_ring_attention_transformer.yaml" => "16_distributed_and_scalable",
            "21_deepseek_v3_r1_mla_moe_mtp.yaml" => "05_architectures",
            "22_flux_mmdit_rectified_flow_generator.yaml" => "07_domain_tasks",
            "23_llama3_qwen25_dense_gqa_llm.yaml" => "05_architectures",
            "24_gemma2_local_global_softcap_llm.yaml" => "05_architectures",
            "25_recurrent_depth_ttt_titans_reasoner.yaml" => {
                "08_learning_paradigms_rl_and_alignment"
            }
            "26_gaussian_splatting_plucker_3d_reconstructor.yaml" => "07_domain_tasks",
            "27_liquid_cfc_drone_flight_controller.yaml" => "06_smooth_and_physics_informed_ai",
            "28_spiking_lif_neuromorphic_event_processor.yaml" => "05_architectures",
            "29_reservoir_esn_ngrc_chaotic_attractor.yaml" => "06_smooth_and_physics_informed_ai",
            "30_whitebox_crate_deq_optnet_solver.yaml" => "06_smooth_and_physics_informed_ai",
            "31_hyperbolic_poincare_sheaf_gflownet.yaml" => "01_paradigms_and_dichotomies",
            "32_zkml_homomorphic_verifiable_classifier.yaml" => {
                "11_ai_safety_privacy_and_verification"
            }
            "33_alphafold3_pairformer_atomic_diffusion.yaml" => "07_domain_tasks",
            "34_vjepa_ijepa_latent_video_world_model.yaml" => "01_paradigms_and_dichotomies",
            "35_omnimodal_native_speech_vision_llm.yaml" => "07_domain_tasks",
            "36_openvla_pi0_vision_language_action_policy.yaml" => "07_domain_tasks",
            "37_colpali_late_interaction_document_retriever.yaml" => "07_domain_tasks",
            "38_dreamerv3_rssm_imagination_world_model.yaml" => {
                "08_learning_paradigms_rl_and_alignment"
            }
            "39_decision_transformer_dpo_preference_aligner.yaml" => {
                "08_learning_paradigms_rl_and_alignment"
            }
            "40_alphagenome_enformer_regulatory_dna_model.yaml" => "07_domain_tasks",
            "41_mattergen_equivariant_crystal_diffusion.yaml" => {
                "06_smooth_and_physics_informed_ai"
            }
            "42_instant_ngp_siren_volumetric_nerf.yaml" => "07_domain_tasks",
            "43_graphcast_spherical_icosahedral_weather.yaml" => {
                "06_smooth_and_physics_informed_ai"
            }
            "44_rwkv7_blt_byte_latent_transformer.yaml" => "05_architectures",
            "45_quantum_variational_pqc_hybrid_eigensolver.yaml" => "05_architectures",
            "46_photonic_mzi_memristive_analog_accelerator.yaml" => {
                "17_deployment_low_level_and_capstones"
            }
            "53_attention_is_not_all_you_need_gmlp_hyena.yaml" => "05_architectures",
            "54_raw_bit_byte_megabyte_mambabyte_llm.yaml" => "05_architectures",
            "55_zero_shot_voice_cloning_codec_flow_tts.yaml" => "07_domain_tasks",
            "56_tabular_gbdt_vime_self_supervised_booster.yaml" => {
                "02_classical_and_probabilistic_ml"
            }
            "57_self_supervised_dinov2_mae_vicreg_vision.yaml" => "01_paradigms_and_dichotomies",
            "58_super_smooth_lipschitz_sobolev_icnn.yaml" => "06_smooth_and_physics_informed_ai",
            "59_learnable_quantization_lsq_fracbits_qat.yaml" => "14_quantisation",
            _ => "05_architectures",
        }
    }

    /// Concise architectural summary describing the blueprint's key operators and industry role.
    pub fn summary(&self) -> &'static str {
        match self.file_name {
            "01_spatiotemporal_dit_flow.yaml" => {
                "Spatiotemporal DiT with WTConv wavelet stem, 3D-RoPE, Slot Attention, and OT-CFM flow matching."
            }
            "02_bimanual_diffusion_act_policy.yaml" => {
                "Bimanual robotic Action Chunking Transformer with SO(3) 6D rotation & CBF-QP safety barrier."
            }
            "03_streaming_codec_mamba.yaml" => {
                "Low-latency neural audio codec with periodic Snake activations, causal Mamba-2 SSD, and unrolled LISTA."
            }
            "04_symplectic_fourier_neural_operator.yaml" => {
                "Resolution-invariant Fourier Neural Operator (FNO) with curl-free vector potential and symplectic conservation."
            }
            "05_equivariant_biomolecular_flow.yaml" => {
                "AlphaFold-3 style biomolecular complex predictor: Evoformer triangle updates, SE(3) IPA, EGNN, and FAPE loss."
            }
            "06_patch_xlstm_probabilistic_forecaster.yaml" => {
                "Non-stationary time-series forecaster combining RevIN, xLSTM matrix memory (mLSTM), and Quantile CRPS/GPD."
            }
            "07_monotonic_ple_tabpfn.yaml" => {
                "Prior-data tabular classifier with Piecewise Linear Embeddings (PLE), HardConcrete L0, and monotonic risk head."
            }
            "08_hybrid_jamba_mla_verifier.yaml" => {
                "Hybrid Mamba-2 SSD + Multi-Head Latent Attention (MLA) + DeepSeekMoE process reward verifier."
            }
            "09_autograd_custom_kernel_backward.yaml" => {
                "Custom forward/backward JVP/VJP autograd verification pipeline with fused kernel parity checks."
            }
            "10_optimal_transport_sinkhorn_matcher.yaml" => {
                "Differentiable combinatorial assignment using SoftSort permutahedron projection and Entmax-1.5."
            }
            "11_muon_wsd_accelerated_transformer.yaml" => {
                "Newton-Schulz Muon 2D + AdamW 1D partitioned Transformer trained under Warmup-Stable-Decay (WSD)."
            }
            "12_safe_cbf_qp_diffusion_policy.yaml" => {
                "Embodied trajectory policy with differentiable Control Barrier Function Quadratic Program (CBF-QP)."
            }
            "13_structural_causal_neural_scm.yaml" => {
                "Identifiable Structural Causal Model (SCM) for interventional do-calculus and counterfactual inference."
            }
            "14_monosemantic_sparse_autoencoder_sae.yaml" => {
                "Mechanistic interpretability Sparse Autoencoder (SAE) with L0/Entmax gating to disentangle polysemantic superpositions."
            }
            "15_differentially_private_dp_sgd.yaml" => {
                "Differentially Private (epsilon, delta)-DP network with per-sample gradient clipping and Rényi accounting."
            }
            "16_torch_compile_inductor_fastpath.yaml" => {
                "Zero-graph-break static-shape fastpath optimized for torch.compile TorchInductor and CUDA Graphs."
            }
            "17_bitnet_ternary_1_58_quant.yaml" => {
                "Multiplier-free 1.58-bit native ternary {-1, 0, +1} BitNet b1.58 architecture with INT8 activations."
            }
            "18_awq_activation_aware_quantizer.yaml" => {
                "Activation-Aware Weight Quantization (AWQ) protecting 1% salient channels in INT4/NF4 precision."
            }
            "19_speculative_medusa_draft_decoder.yaml" => {
                "Multi-head Medusa speculative decoding draft network for parallel tree-based token verification."
            }
            "20_fsdp2_ring_attention_transformer.yaml" => {
                "Multi-node FSDP2 per-parameter sharded Transformer with Ring Attention for million-token context."
            }
            "21_deepseek_v3_r1_mla_moe_mtp.yaml" => {
                "Industry Flagship: DeepSeek-V3/R1 with Decoupled-RoPE MLA, Dense Layer-1 Warmup, 256-Expert Aux-Loss-Free MoE, MTP & FP8."
            }
            "22_flux_mmdit_rectified_flow_generator.yaml" => {
                "Industry Image Generator: Flux.1 / SD3 Dual-Stream MMDiT + Single-Stream AdaLN-Zero + Per-Head QK-Norm + Rectified Flow."
            }
            "23_llama3_qwen25_dense_gqa_llm.yaml" => {
                "Industry Dense LLM: Llama-3.3 / Qwen-2.5 with 8:1 Grouped-Query Attention (GQA), High-Theta 500K RoPE, and SwiGLU."
            }
            "24_gemma2_local_global_softcap_llm.yaml" => {
                "Industry Dense LLM: Gemma-2 / Gemma-3 interleaving Local Sliding-Window & Global GQA, Sandwich-LN, GeGLU, and Logit Soft-Cap."
            }
            "25_recurrent_depth_ttt_titans_reasoner.yaml" => {
                "System-2 Latent Reasoner: Input-anchored Recurrent Depth Loop with PonderNet KL halting, Titans Memory, TTT & DiffAttn."
            }
            "26_gaussian_splatting_plucker_3d_reconstructor.yaml" => {
                "Industry 3D Generator: Large Reconstruction Model mapping multi-view 6D Plücker ray patches to 3D Gaussian Splatting primitives."
            }
            "27_liquid_cfc_drone_flight_controller.yaml" => {
                "Rare Continuous-Time ML: Closed-form Continuous-time (CfC) Liquid Neural Network for adaptive SO(3) drone flight control."
            }
            "28_spiking_lif_neuromorphic_event_processor.yaml" => {
                "Rare Neuromorphic ML: Event-driven Spiking Leaky Integrate-and-Fire (LIF) SNN with surrogate gradients on 16MB Cortex-M55."
            }
            "29_reservoir_esn_ngrc_chaotic_attractor.yaml" => {
                "Rare Dynamical Systems ML: Echo State Network (ESN) + Next-Gen Delay-Tap Polynomial Reservoir (NG-RC) for chaotic PDEs in FP64."
            }
            "30_whitebox_crate_deq_optnet_solver.yaml" => {
                "Rare White-Box / Implicit ML: Coding Rate Reduction (CRATE) + Unrolled ADMM-Net + Deep Equilibrium (DEQ) + OptNet KKT QP."
            }
            "31_hyperbolic_poincare_sheaf_gflownet.yaml" => {
                "Rare Non-Euclidean ML: Hyperbolic Poincaré gyrovector manifold + Cellular Sheaf diffusion + Clifford algebra + GFlowNet."
            }
            "32_zkml_homomorphic_verifiable_classifier.yaml" => {
                "Rare Cryptographic ML: Zero-Knowledge (zkML) prime finite-field F_p & CKKS homomorphic encrypted inference with CROWN/IBP bounds."
            }
            "33_alphafold3_pairformer_atomic_diffusion.yaml" => {
                "Biomolecular Flagship: AlphaFold 3 48-block Pairformer (Single+Pair Triangle Attention) + AdaLN-Zero All-Atom Coordinate Diffusion."
            }
            "34_vjepa_ijepa_latent_video_world_model.yaml" => {
                "Self-Supervised World Model: V-JEPA / I-JEPA (JEV) predicting masked spatiotemporal embeddings in representation space with EMA target alignment."
            }
            "35_omnimodal_native_speech_vision_llm.yaml" => {
                "Omnimodal Duplex Model: Gemini Live / Moshi style interleaved acoustic RVQ codebooks, dynamic vision patches, and Depth-Transformer MTP."
            }
            "36_openvla_pi0_vision_language_action_policy.yaml" => {
                "Embodied VLA Foundation Model: OpenVLA / Pi0 combining SigLIP vision patches, VLM GQA backbone, and 50-Hz Flow-Matching Action Expert."
            }
            "37_colpali_late_interaction_document_retriever.yaml" => {
                "Visual Document Retrieval: ColPali / ColQwen2 OCR-free page patch encoder with 128-D multi-vector Late-Interaction MaxSim scoring."
            }
            "38_dreamerv3_rssm_imagination_world_model.yaml" => {
                "Model-Based RL World Model: DreamerV3 / MuZero RSSM with Symlog inputs, 32x32 unimix categorical latents, and Symexp two-hot value heads."
            }
            "39_decision_transformer_dpo_preference_aligner.yaml" => {
                "Offline RL & Alignment: Return-conditioned Decision Transformer with reference-anchored DPO contrastive log-ratio preference optimization."
            }
            "40_alphagenome_enformer_regulatory_dna_model.yaml" => {
                "Regulatory Genomics Foundation Model: AlphaGenome / Enformer 1-Mb DNA wavelet conv + chromatin attention predicting 5,313 epigenetic tracks."
            }
            "41_mattergen_equivariant_crystal_diffusion.yaml" => {
                "Materials Discovery: MatterGen / CDVAE periodic SO(3)xZ^3 crystal graph diffusion over fractional coordinates and unit-cell lattice vectors."
            }
            "42_instant_ngp_siren_volumetric_nerf.yaml" => {
                "Volumetric Neural Radiance Field: Instant-NGP 16-level spatial hash grid + periodic SIREN/Snake activations for 5D ray volume rendering."
            }
            "43_graphcast_spherical_icosahedral_weather.yaml" => {
                "Global Earth-System AI: GraphCast / NeuralGCM multi-mesh icosahedral spherical GNN + spectral FNO across 37 atmospheric pressure levels."
            }
            "44_rwkv7_blt_byte_latent_transformer.yaml" => {
                "Tokenizer-Free State-Tracking LLM: Byte Latent Transformer (BLT) dynamic entropy byte patcher + RWKV-7 generalized delta-rule linear RNN."
            }
            "45_quantum_variational_pqc_hybrid_eigensolver.yaml" => {
                "Quantum Neural Network: Hybrid PQC-VQE with data re-uploading Bloch angle encoding, SU(2) Pauli rotations, CNOT entanglement & Hamiltonian readout."
            }
            "46_photonic_mzi_memristive_analog_accelerator.yaml" => {
                "Optical & Analog Hardware AI: Coherent Clements 64x64 Mach-Zehnder Interferometer (MZI) unitary mesh + 8-bit ADC memristive crossbar."
            }
            "47_simplicial_hodge_laplacian_topological_net.yaml" => {
                "Topological Deep Learning: Simplicial & cellular complex Hodge-Laplacian (0/1/2-simplices) message passing with Helmholtz-Hodge curl/harmonic decomposition."
            }
            "48_continuous_normalizing_flow_ffjord_copula.yaml" => {
                "Exact-Likelihood Generative Flow: Free-form Continuous Normalizing Flow (FFJORD) with Hutchinson trace log-det Jacobian & neural vine copula splines."
            }
            "49_energy_based_langevin_predictive_coding.yaml" => {
                "Energy-Based & Non-Backprop ML: Cortical Predictive Coding Network (PCN) & Continuous EBM with Langevin MCMC relaxation and Hopfield energy basins."
            }
            "50_neurosymbolic_logic_tensor_dnc_memory.yaml" => {
                "Neurosymbolic Reasoning & Memory: Differentiable Neural Computer (DNC) addressable matrix memory + Łukasiewicz Logic Tensor Network (LTN) t-norms."
            }
            "51_qmix_multi_agent_cfr_game_coordinator.yaml" => {
                "Multi-Agent RL & Game Theory: Monotonic QMIX hypernetwork value mixer (IGM guaranteed) + TarMAC communication + Counterfactual Regret (CFR) solver."
            }
            "52_spintronic_skyrmion_organoid_active_inference.yaml" => {
                "Wetware & Spintronic AI: Magnetic skyrmion spin-torque reservoir + biological cortical organoid HD-MEA Variational Free Energy active inference."
            }
            "53_attention_is_not_all_you_need_gmlp_hyena.yaml" => {
                "Attention-Free Sequence Mixer: Rank-collapse prevention (Dong et al.) with gMLP Spatial Gating Unit (SGU), Hyena implicit long convolutions, and ConvNeXt-V2."
            }
            "54_raw_bit_byte_megabyte_mambabyte_llm.yaml" => {
                "Raw Bit & Byte Foundation LLM: MegaByte / MambaByte / bGPT hierarchical global-patch to local-byte & 8-bit-plane autoregressive transformer/SSM."
            }
            "55_zero_shot_voice_cloning_codec_flow_tts.yaml" => {
                "Zero-Shot Voice Cloning & TTS: F5-TTS / VALL-E / CosyVoice in-context speaker acoustic prompt conditioning, 8-codebook RVQ, OT-CFM DiT & Snake vocoder."
            }
            "56_tabular_gbdt_vime_self_supervised_booster.yaml" => {
                "Self-Supervised Gradient Boosting: Oblivious GBDT (CatBoost/NODE) with VIME mask/value reconstruction, Hastie marginal discrimination & KAN leaf calibration."
            }
            "57_self_supervised_dinov2_mae_vicreg_vision.yaml" => {
                "Unified Self-Supervised Vision: DINOv2 EMA Student-Teacher + Sinkhorn-Knopp prototypes, I-JEPA/MAE masked prediction, and VICReg anti-collapse."
            }
            "58_super_smooth_lipschitz_sobolev_icnn.yaml" => {
                "Super-Smooth & Certified Lipschitz Architecture: Björck orthonormalization, Cayley transform, C^∞ SoftplusBeta/GroupSort-2 activations, and Sobolev H^2 curvature penalty."
            }
            "59_learnable_quantization_lsq_fracbits_qat.yaml" => {
                "Learnable Quantization Level & Mixed-Bitwidth QAT: LSQ+ learnable step-size, PACT activation clipping, FracBits continuous bitwidth optimization, and BOPs complexity penalty."
            }
            _ => "Pre-engineered AI architecture blueprint.",
        }
    }
}

pub const TEMPLATES: &[StarterTemplate] = &[
    StarterTemplate {
        file_name: "01_spatiotemporal_dit_flow.yaml",
        display_name: "01. Spatiotemporal DiT Flow",
        yaml_content: include_str!("../../../templates/01_spatiotemporal_dit_flow.yaml"),
    },
    StarterTemplate {
        file_name: "02_bimanual_diffusion_act_policy.yaml",
        display_name: "02. Bimanual Diffusion ACT Policy",
        yaml_content: include_str!("../../../templates/02_bimanual_diffusion_act_policy.yaml"),
    },
    StarterTemplate {
        file_name: "03_streaming_codec_mamba.yaml",
        display_name: "03. Streaming Codec Mamba",
        yaml_content: include_str!("../../../templates/03_streaming_codec_mamba.yaml"),
    },
    StarterTemplate {
        file_name: "04_symplectic_fourier_neural_operator.yaml",
        display_name: "04. Symplectic Fourier Neural Operator",
        yaml_content: include_str!("../../../templates/04_symplectic_fourier_neural_operator.yaml"),
    },
    StarterTemplate {
        file_name: "05_equivariant_biomolecular_flow.yaml",
        display_name: "05. Equivariant Biomolecular Flow",
        yaml_content: include_str!("../../../templates/05_equivariant_biomolecular_flow.yaml"),
    },
    StarterTemplate {
        file_name: "06_patch_xlstm_probabilistic_forecaster.yaml",
        display_name: "06. Patch xLSTM Forecaster",
        yaml_content: include_str!(
            "../../../templates/06_patch_xlstm_probabilistic_forecaster.yaml"
        ),
    },
    StarterTemplate {
        file_name: "07_monotonic_ple_tabpfn.yaml",
        display_name: "07. Monotonic PLE TabPFN",
        yaml_content: include_str!("../../../templates/07_monotonic_ple_tabpfn.yaml"),
    },
    StarterTemplate {
        file_name: "08_hybrid_jamba_mla_verifier.yaml",
        display_name: "08. Hybrid Jamba-2 MLA Verifier",
        yaml_content: include_str!("../../../templates/08_hybrid_jamba_mla_verifier.yaml"),
    },
    StarterTemplate {
        file_name: "09_autograd_custom_kernel_backward.yaml",
        display_name: "09. Autograd Custom Kernel Backward",
        yaml_content: include_str!("../../../templates/09_autograd_custom_kernel_backward.yaml"),
    },
    StarterTemplate {
        file_name: "10_optimal_transport_sinkhorn_matcher.yaml",
        display_name: "10. Optimal Transport Sinkhorn Matcher",
        yaml_content: include_str!("../../../templates/10_optimal_transport_sinkhorn_matcher.yaml"),
    },
    StarterTemplate {
        file_name: "11_muon_wsd_accelerated_transformer.yaml",
        display_name: "11. Muon WSD Accelerated Transformer",
        yaml_content: include_str!("../../../templates/11_muon_wsd_accelerated_transformer.yaml"),
    },
    StarterTemplate {
        file_name: "12_safe_cbf_qp_diffusion_policy.yaml",
        display_name: "12. Safe CBF-QP Diffusion Policy",
        yaml_content: include_str!("../../../templates/12_safe_cbf_qp_diffusion_policy.yaml"),
    },
    StarterTemplate {
        file_name: "13_structural_causal_neural_scm.yaml",
        display_name: "13. Structural Causal Neural SCM",
        yaml_content: include_str!("../../../templates/13_structural_causal_neural_scm.yaml"),
    },
    StarterTemplate {
        file_name: "14_monosemantic_sparse_autoencoder_sae.yaml",
        display_name: "14. Monosemantic Sparse Autoencoder (SAE)",
        yaml_content: include_str!(
            "../../../templates/14_monosemantic_sparse_autoencoder_sae.yaml"
        ),
    },
    StarterTemplate {
        file_name: "15_differentially_private_dp_sgd.yaml",
        display_name: "15. Differentially Private DP-SGD",
        yaml_content: include_str!("../../../templates/15_differentially_private_dp_sgd.yaml"),
    },
    StarterTemplate {
        file_name: "16_torch_compile_inductor_fastpath.yaml",
        display_name: "16. Torch Compile Inductor Fastpath",
        yaml_content: include_str!("../../../templates/16_torch_compile_inductor_fastpath.yaml"),
    },
    StarterTemplate {
        file_name: "17_bitnet_ternary_1_58_quant.yaml",
        display_name: "17. BitNet Ternary 1.58b Quant",
        yaml_content: include_str!("../../../templates/17_bitnet_ternary_1_58_quant.yaml"),
    },
    StarterTemplate {
        file_name: "18_awq_activation_aware_quantizer.yaml",
        display_name: "18. AWQ Activation-Aware Quantizer",
        yaml_content: include_str!("../../../templates/18_awq_activation_aware_quantizer.yaml"),
    },
    StarterTemplate {
        file_name: "19_speculative_medusa_draft_decoder.yaml",
        display_name: "19. Speculative Medusa Draft Decoder",
        yaml_content: include_str!("../../../templates/19_speculative_medusa_draft_decoder.yaml"),
    },
    StarterTemplate {
        file_name: "20_fsdp2_ring_attention_transformer.yaml",
        display_name: "20. FSDP2 Ring Attention Transformer",
        yaml_content: include_str!("../../../templates/20_fsdp2_ring_attention_transformer.yaml"),
    },
    StarterTemplate {
        file_name: "21_deepseek_v3_r1_mla_moe_mtp.yaml",
        display_name: "21. DeepSeek-V3 / R1 (MLA + Aux-Loss-Free MoE + MTP)",
        yaml_content: include_str!("../../../templates/21_deepseek_v3_r1_mla_moe_mtp.yaml"),
    },
    StarterTemplate {
        file_name: "22_flux_mmdit_rectified_flow_generator.yaml",
        display_name: "22. Flux.1 / SD3 MMDiT Rectified Flow Image Generator",
        yaml_content: include_str!(
            "../../../templates/22_flux_mmdit_rectified_flow_generator.yaml"
        ),
    },
    StarterTemplate {
        file_name: "23_llama3_qwen25_dense_gqa_llm.yaml",
        display_name: "23. Llama-3.3 / Qwen-2.5 Dense GQA + SwiGLU LLM",
        yaml_content: include_str!("../../../templates/23_llama3_qwen25_dense_gqa_llm.yaml"),
    },
    StarterTemplate {
        file_name: "24_gemma2_local_global_softcap_llm.yaml",
        display_name: "24. Gemma-2 / Gemma-3 Local-Global + Soft-Cap LLM",
        yaml_content: include_str!("../../../templates/24_gemma2_local_global_softcap_llm.yaml"),
    },
    StarterTemplate {
        file_name: "25_recurrent_depth_ttt_titans_reasoner.yaml",
        display_name: "25. Recurrent-Depth Latent Reasoner (Titans + TTT + DiffAttn)",
        yaml_content: include_str!(
            "../../../templates/25_recurrent_depth_ttt_titans_reasoner.yaml"
        ),
    },
    StarterTemplate {
        file_name: "26_gaussian_splatting_plucker_3d_reconstructor.yaml",
        display_name: "26. 3D Gaussian Splatting Reconstructor (Plücker Rays + 3DGS)",
        yaml_content: include_str!(
            "../../../templates/26_gaussian_splatting_plucker_3d_reconstructor.yaml"
        ),
    },
    StarterTemplate {
        file_name: "27_liquid_cfc_drone_flight_controller.yaml",
        display_name: "27. Liquid Time-Constant (CfC) Autonomous Flight Controller",
        yaml_content: include_str!("../../../templates/27_liquid_cfc_drone_flight_controller.yaml"),
    },
    StarterTemplate {
        file_name: "28_spiking_lif_neuromorphic_event_processor.yaml",
        display_name: "28. Neuromorphic Spiking LIF Event-Camera (DVS) SNN",
        yaml_content: include_str!(
            "../../../templates/28_spiking_lif_neuromorphic_event_processor.yaml"
        ),
    },
    StarterTemplate {
        file_name: "29_reservoir_esn_ngrc_chaotic_attractor.yaml",
        display_name: "29. Echo State (ESN) & Next-Gen Reservoir (NG-RC) Chaos Forecaster",
        yaml_content: include_str!(
            "../../../templates/29_reservoir_esn_ngrc_chaotic_attractor.yaml"
        ),
    },
    StarterTemplate {
        file_name: "30_whitebox_crate_deq_optnet_solver.yaml",
        display_name: "30. White-Box CRATE + Implicit DEQ + OptNet KKT Convex Solver",
        yaml_content: include_str!("../../../templates/30_whitebox_crate_deq_optnet_solver.yaml"),
    },
    StarterTemplate {
        file_name: "31_hyperbolic_poincare_sheaf_gflownet.yaml",
        display_name: "31. Hyperbolic Poincaré + Cellular Sheaf + GFlowNet Sampler",
        yaml_content: include_str!("../../../templates/31_hyperbolic_poincare_sheaf_gflownet.yaml"),
    },
    StarterTemplate {
        file_name: "32_zkml_homomorphic_verifiable_classifier.yaml",
        display_name: "32. Cryptographic zkML Finite-Field & CKKS Homomorphic Classifier",
        yaml_content: include_str!(
            "../../../templates/32_zkml_homomorphic_verifiable_classifier.yaml"
        ),
    },
    StarterTemplate {
        file_name: "33_alphafold3_pairformer_atomic_diffusion.yaml",
        display_name: "33. AlphaFold 3 Pairformer & All-Atom Coordinate Diffusion",
        yaml_content: include_str!(
            "../../../templates/33_alphafold3_pairformer_atomic_diffusion.yaml"
        ),
    },
    StarterTemplate {
        file_name: "34_vjepa_ijepa_latent_video_world_model.yaml",
        display_name: "34. V-JEPA / I-JEPA (JEV) Joint-Embedding Video World Model",
        yaml_content: include_str!(
            "../../../templates/34_vjepa_ijepa_latent_video_world_model.yaml"
        ),
    },
    StarterTemplate {
        file_name: "35_omnimodal_native_speech_vision_llm.yaml",
        display_name: "35. Omnimodal Native Speech-Vision-Text Duplex Model (Gemini Live / Moshi)",
        yaml_content: include_str!("../../../templates/35_omnimodal_native_speech_vision_llm.yaml"),
    },
    StarterTemplate {
        file_name: "36_openvla_pi0_vision_language_action_policy.yaml",
        display_name: "36. OpenVLA / Pi0 Vision-Language-Action Flow Robotics Policy",
        yaml_content: include_str!(
            "../../../templates/36_openvla_pi0_vision_language_action_policy.yaml"
        ),
    },
    StarterTemplate {
        file_name: "37_colpali_late_interaction_document_retriever.yaml",
        display_name: "37. ColPali / ColQwen2 Late-Interaction Visual Document Retriever",
        yaml_content: include_str!(
            "../../../templates/37_colpali_late_interaction_document_retriever.yaml"
        ),
    },
    StarterTemplate {
        file_name: "38_dreamerv3_rssm_imagination_world_model.yaml",
        display_name: "38. DreamerV3 / MuZero Recurrent State-Space World Model (RSSM)",
        yaml_content: include_str!(
            "../../../templates/38_dreamerv3_rssm_imagination_world_model.yaml"
        ),
    },
    StarterTemplate {
        file_name: "39_decision_transformer_dpo_preference_aligner.yaml",
        display_name: "39. Offline Decision Transformer & DPO Preference Aligner",
        yaml_content: include_str!(
            "../../../templates/39_decision_transformer_dpo_preference_aligner.yaml"
        ),
    },
    StarterTemplate {
        file_name: "40_alphagenome_enformer_regulatory_dna_model.yaml",
        display_name: "40. AlphaGenome / Enformer Megabase Regulatory DNA Model",
        yaml_content: include_str!(
            "../../../templates/40_alphagenome_enformer_regulatory_dna_model.yaml"
        ),
    },
    StarterTemplate {
        file_name: "41_mattergen_equivariant_crystal_diffusion.yaml",
        display_name: "41. MatterGen / CDVAE Periodic Crystal & Material Generator",
        yaml_content: include_str!(
            "../../../templates/41_mattergen_equivariant_crystal_diffusion.yaml"
        ),
    },
    StarterTemplate {
        file_name: "42_instant_ngp_siren_volumetric_nerf.yaml",
        display_name: "42. Instant-NGP Hash Grid & SIREN Volumetric NeRF",
        yaml_content: include_str!("../../../templates/42_instant_ngp_siren_volumetric_nerf.yaml"),
    },
    StarterTemplate {
        file_name: "43_graphcast_spherical_icosahedral_weather.yaml",
        display_name: "43. GraphCast / NeuralGCM Spherical Multi-Mesh Weather Forecaster",
        yaml_content: include_str!(
            "../../../templates/43_graphcast_spherical_icosahedral_weather.yaml"
        ),
    },
    StarterTemplate {
        file_name: "44_rwkv7_blt_byte_latent_transformer.yaml",
        display_name: "44. Byte Latent Transformer (BLT) & RWKV-7 Delta-Rule Hybrid",
        yaml_content: include_str!("../../../templates/44_rwkv7_blt_byte_latent_transformer.yaml"),
    },
    StarterTemplate {
        file_name: "45_quantum_variational_pqc_hybrid_eigensolver.yaml",
        display_name: "45. Hybrid Quantum Neural Network (QNN) & Variational Eigensolver (PQC-VQE)",
        yaml_content: include_str!(
            "../../../templates/45_quantum_variational_pqc_hybrid_eigensolver.yaml"
        ),
    },
    StarterTemplate {
        file_name: "46_photonic_mzi_memristive_analog_accelerator.yaml",
        display_name: "46. Coherent Photonic MZI Optical Mesh & Memristive Crossbar",
        yaml_content: include_str!(
            "../../../templates/46_photonic_mzi_memristive_analog_accelerator.yaml"
        ),
    },
    StarterTemplate {
        file_name: "47_simplicial_hodge_laplacian_topological_net.yaml",
        display_name: "47. Simplicial & Cellular Complex Hodge-Laplacian Topological Network",
        yaml_content: include_str!(
            "../../../templates/47_simplicial_hodge_laplacian_topological_net.yaml"
        ),
    },
    StarterTemplate {
        file_name: "48_continuous_normalizing_flow_ffjord_copula.yaml",
        display_name: "48. Continuous Normalizing Flow (FFJORD) & Neural Vine Copula",
        yaml_content: include_str!(
            "../../../templates/48_continuous_normalizing_flow_ffjord_copula.yaml"
        ),
    },
    StarterTemplate {
        file_name: "49_energy_based_langevin_predictive_coding.yaml",
        display_name: "49. Continuous Energy-Based Model (EBM) & Predictive Coding Network (PCN)",
        yaml_content: include_str!(
            "../../../templates/49_energy_based_langevin_predictive_coding.yaml"
        ),
    },
    StarterTemplate {
        file_name: "50_neurosymbolic_logic_tensor_dnc_memory.yaml",
        display_name: "50. Neurosymbolic Logic Tensor Network (LTN) & DNC Matrix Memory",
        yaml_content: include_str!(
            "../../../templates/50_neurosymbolic_logic_tensor_dnc_memory.yaml"
        ),
    },
    StarterTemplate {
        file_name: "51_qmix_multi_agent_cfr_game_coordinator.yaml",
        display_name: "51. Multi-Agent Monotonic QMIX Coordinator & CFR Game Solver",
        yaml_content: include_str!(
            "../../../templates/51_qmix_multi_agent_cfr_game_coordinator.yaml"
        ),
    },
    StarterTemplate {
        file_name: "52_spintronic_skyrmion_organoid_active_inference.yaml",
        display_name: "52. Spintronic Skyrmion Reservoir & Bio-Organoid Active Inference",
        yaml_content: include_str!(
            "../../../templates/52_spintronic_skyrmion_organoid_active_inference.yaml"
        ),
    },
    StarterTemplate {
        file_name: "53_attention_is_not_all_you_need_gmlp_hyena.yaml",
        display_name: "53. Attention Is Not All You Need (gMLP + Hyena Rank-Preserving Mixer)",
        yaml_content: include_str!(
            "../../../templates/53_attention_is_not_all_you_need_gmlp_hyena.yaml"
        ),
    },
    StarterTemplate {
        file_name: "54_raw_bit_byte_megabyte_mambabyte_llm.yaml",
        display_name: "54. Raw Bit & Byte Multi-Scale Foundation LLM (MegaByte / MambaByte / bGPT)",
        yaml_content: include_str!(
            "../../../templates/54_raw_bit_byte_megabyte_mambabyte_llm.yaml"
        ),
    },
    StarterTemplate {
        file_name: "55_zero_shot_voice_cloning_codec_flow_tts.yaml",
        display_name: "55. Zero-Shot Voice Cloning & Neural Codec Flow-Matching TTS (F5-TTS / VALL-E)",
        yaml_content: include_str!(
            "../../../templates/55_zero_shot_voice_cloning_codec_flow_tts.yaml"
        ),
    },
    StarterTemplate {
        file_name: "56_tabular_gbdt_vime_self_supervised_booster.yaml",
        display_name: "56. Self-Supervised Gradient Boosted Decision Trees (VIME + CatBoost/NODE)",
        yaml_content: include_str!(
            "../../../templates/56_tabular_gbdt_vime_self_supervised_booster.yaml"
        ),
    },
    StarterTemplate {
        file_name: "57_self_supervised_dinov2_mae_vicreg_vision.yaml",
        display_name: "57. Unified Self-Supervised Vision Backbone (DINOv2 + MAE + VICReg + I-JEPA)",
        yaml_content: include_str!(
            "../../../templates/57_self_supervised_dinov2_mae_vicreg_vision.yaml"
        ),
    },
    StarterTemplate {
        file_name: "58_super_smooth_lipschitz_sobolev_icnn.yaml",
        display_name: "58. Super-Smooth C^∞ 1-Lipschitz Björck-Cayley & Sobolev H² Network",
        yaml_content: include_str!(
            "../../../templates/58_super_smooth_lipschitz_sobolev_icnn.yaml"
        ),
    },
    StarterTemplate {
        file_name: "59_learnable_quantization_lsq_fracbits_qat.yaml",
        display_name: "59. Learnable Quantization Level & Mixed-Bitwidth QAT (LSQ+ / FracBits / PACT)",
        yaml_content: include_str!(
            "../../../templates/59_learnable_quantization_lsq_fracbits_qat.yaml"
        ),
    },
];

pub fn get_template_by_filename(filename: &str) -> Option<&'static str> {
    TEMPLATES
        .iter()
        .find(|t| t.file_name == filename)
        .map(|t| t.yaml_content)
}
