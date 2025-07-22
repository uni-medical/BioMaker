meddle data_dir="meddle/med_dl_tasks/mco_wsi_conch" exp_name="all_pipeline-mco_os_conch" \
     goal="Predict patient survival risk (hazard score) from whole slide image (WSI) features (shape = tile_num x feat_dim)." \
     eval="Use the Concordance Index (C-Index) to evaluate risk prediction accuracy." \
     agent.step_plan.proposal=10 agent.step_plan.tuning=10 agent.step_plan.hpo=0 \
     agent.force_monai_with_prompt=false agent.use_deep_research_knowledge="wsi_tricks" log_level="DEBUG"