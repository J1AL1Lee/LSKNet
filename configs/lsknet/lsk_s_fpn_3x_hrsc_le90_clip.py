_base_ = ['./lsk_s_fpn_3x_hrsc_le90.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(
            clip_cfg=dict(
                enable=True,
                text_embed_path='resources/clip/hrsc_ship_ViT-B-32.pt',
                loss_weight=0.1))))
