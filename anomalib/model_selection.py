def model_selection(model,MODEL_CONFIG):
    match model:
        case 'Padim':
            from anomalib.models import Padim
            model=Padim( backbone=MODEL_CONFIG['backbone'],n_features=MODEL_CONFIG['n_features'],layers=list(MODEL_CONFIG['Layers']))
        case 'EfficientAd':
            from anomalib.models import EfficientAd
            model=EfficientAd(
            teacher_out_channels=MODEL_CONFIG['teacher_out_channels'],
            lr=MODEL_CONFIG['lr'],
            weight_decay=MODEL_CONFIG['weight_decay'],
            padding=MODEL_CONFIG['padding'],
            pad_maps=MODEL_CONFIG['pad_maps'],
            batch_size=MODEL_CONFIG['batch_size']
        )
        case 'Stfpm':
            from anomalib.models import Stfpm
            model=Stfpm(
            backbone=MODEL_CONFIG['backbone'],
            layers=list(MODEL_CONFIG['Layers']))
        case 'Ganomaly':
            from anomalib.models import Ganomaly
            model=Ganomaly(
            batch_size=MODEL_CONFIG['batch_size'],
            n_features=MODEL_CONFIG['n_features'], 
            latent_vec_size=MODEL_CONFIG['latent_vec_size'], 
            extra_layers=MODEL_CONFIG['extra_layers'], 
            add_final_conv_layer=MODEL_CONFIG['add_final_conv_layer'], 
            wadv=MODEL_CONFIG['wadv'], 
            wcon=MODEL_CONFIG['wcon'], 
            wenc=MODEL_CONFIG['wenc'], 
            lr=MODEL_CONFIG['lr'], 
            beta1=MODEL_CONFIG['beta1'], 
            beta2=MODEL_CONFIG['beta2'])
        case 'Cfa':
            from anomalib.models import Cfa
            model=Cfa(backbone=MODEL_CONFIG['backbone'], gamma_c=MODEL_CONFIG['gamma_c'], gamma_d=MODEL_CONFIG['gamma_d'],
                       num_nearest_neighbors=MODEL_CONFIG['num_nearest_neighbors'], num_hard_negative_features=MODEL_CONFIG['num_hard_negative_features'], radius=MODEL_CONFIG['radius'])
        case 'Cflow':
            from anomalib.models import Cflow
            model=Cflow(backbone=MODEL_CONFIG['backbone'], layers=tuple(MODEL_CONFIG['layers']), fiber_batch_size=MODEL_CONFIG['fiber_batch_size'], decoder=MODEL_CONFIG['decoder'], condition_vector=MODEL_CONFIG['condition_vector'], coupling_blocks=MODEL_CONFIG['coupling_blocks'], clamp_alpha=MODEL_CONFIG['clamp_alpha'], permute_soft=MODEL_CONFIG['permute_soft'], lr=MODEL_CONFIG['lr'])
        case 'Csflow':
            #it uses efficient net, so it requires torchvision!=0.16.0
            from anomalib.models import Csflow
            model=Csflow(cross_conv_hidden_channels=MODEL_CONFIG['cross_conv_hidden_channels'], n_coupling_blocks=MODEL_CONFIG['n_coupling_blocks'], clamp=MODEL_CONFIG['clamp'], num_channels=MODEL_CONFIG['num_channels'])
        case 'Dfkde':
            from anomalib.models import Dfkde
            model=Dfkde(backbone=MODEL_CONFIG['backbone'], layers=tuple(MODEL_CONFIG['layers']),  n_pca_components=MODEL_CONFIG['n_pca_components'], max_training_points=MODEL_CONFIG['max_training_points'])
        case 'Dfm':
            from anomalib.models import Dfm
            model=Dfm(backbone=MODEL_CONFIG['backbone'], layer=MODEL_CONFIG['layer'], pooling_kernel_size=MODEL_CONFIG['pooling_kernel_size'], pca_level=MODEL_CONFIG['pca_level'], score_type=MODEL_CONFIG['score_type'])
        case 'Draem':
            from anomalib.models import Draem
            model=Draem(enable_sspcab=MODEL_CONFIG['enable_sspcab'], sspcab_lambda=MODEL_CONFIG['sspcab_lambda'], anomaly_source_path=MODEL_CONFIG['anomaly_source_path'], beta=tuple(MODEL_CONFIG['beta']))
        case 'Fastflow':
            from anomalib.models import Fastflow
            model=Fastflow(backbone=MODEL_CONFIG['backbone'], flow_steps=MODEL_CONFIG['flow_steps'], conv3x3_only=MODEL_CONFIG['conv3x3_only'], hidden_ratio=MODEL_CONFIG['hidden_ratio'])
        case 'Patchcore':
            from anomalib.models import Patchcore
            model=Patchcore(backbone=MODEL_CONFIG['backbone'], layers=list(MODEL_CONFIG['layers']), coreset_sampling_ratio=MODEL_CONFIG['coreset_sampling_ratio'], num_neighbors=MODEL_CONFIG['num_neighbors'])
        case 'ReverseDistillation':
            from anomalib.models import ReverseDistillation
            model=ReverseDistillation(backbone=MODEL_CONFIG['backbone'], layers=tuple(MODEL_CONFIG['layers']))
        case 'Rkde':
            from anomalib.models import Rkde
            model=Rkde(roi_score_threshold=MODEL_CONFIG['roi_score_threshold'], min_box_size=MODEL_CONFIG['min_box_size'], iou_threshold=MODEL_CONFIG['iou_threshold'], max_detections_per_image=MODEL_CONFIG['max_detections_per_image'], n_pca_components=MODEL_CONFIG['n_pca_components'], max_training_points=MODEL_CONFIG['max_training_points'])
        case 'Uflow':
            from anomalib.models import Uflow
            model=Uflow(backbone=MODEL_CONFIG['backbone'], flow_steps=MODEL_CONFIG['flow_steps'], affine_clamp=MODEL_CONFIG['affine_clamp'], affine_subnet_channels_ratio=MODEL_CONFIG['affine_subnet_channels_ratio'], permute_soft=MODEL_CONFIG['permute_soft'])
        case 'WinClip':
            from anomalib.models import WinClip
            model=WinClip(class_name=MODEL_CONFIG['class_name'], k_shot=MODEL_CONFIG['k_shot'], scales=list(MODEL_CONFIG['scales']), few_shot_source=MODEL_CONFIG['few_shot_source'])
        case 'Dsr':
            from anomalib.models import Dsr
            model=Dsr(latent_anomaly_strength=MODEL_CONFIG['latent_anomaly_strength'], upsampling_train_ratio=MODEL_CONFIG['upsampling_train_ratio'])
        case _:
            print('model does not exist, please check the model name') 
            return None
        
    return model           
