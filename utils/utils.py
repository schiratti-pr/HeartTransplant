from monai.networks.nets import SegResNet, SwinUNETR, UNet


def get_model(name, spatial_dims=2, num_in_channels=1):
    if name == 'SegResNet':
        net = SegResNet(
            spatial_dims=spatial_dims,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=num_in_channels,
            out_channels=1,
            dropout_prob=0.2,
        )
    elif name == 'SwinUNETR':
        net = SwinUNETR(
            img_size=512,
            in_channels=num_in_channels,
            out_channels=1,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            spatial_dims=spatial_dims
        )
    elif name == 'UNet':
        net = UNet(
            spatial_dims=spatial_dims,
            in_channels=num_in_channels,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    if spatial_dims == 3:
        net = net.double()
    else:
        net = net

    return net
