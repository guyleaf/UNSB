import os

from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def common_options(self):
        name = "fwid_image2weather"
        dataroot = os.path.expanduser(
            "~/data/Weather/FWID_Image2Weather_unpaired/dataset"
        )
        weathers = ["foggy", "rainy", "snowy"]

        # Maximum training image size in RTX 3090 24GB
        # issue: https://github.com/taesungp/contrastive-unpaired-translation/issues/139#issuecomment-1165577005
        max_crop_size = 720
        max_load_size = round(max_crop_size * 1.1)

        return [
            Options(
                dataroot=dataroot,
                dataset_mode="unaligned_weather",
                data_domainA="clear",
                data_domainB=weather,
                name=f"{name}_SB_{weather}_fixed_load_{max_load_size}_fixed_crop_{max_crop_size}_no_seed",
                mode="sb",
                lambda_SB=1.0,
                lambda_NCE=1.0,
                display_env=f"{name}_SB_{weather}_fixed_load_{max_load_size}_fixed_crop_{max_crop_size}_no_seed",
                # max_load_size=max_load_size,
                # max_crop_size=max_crop_size,
                load_size=max_load_size,
                crop_size=max_crop_size
            )
            for weather in weathers
        ]

    def commands(self):
        return [
            "python train.py "
            + str(
                opt.set(
                    data_domainA=os.path.join("train", opt.kvs["data_domainA"]),
                    data_domainB=os.path.join("train", opt.kvs["data_domainB"]),
                )
            )
            for opt in self.common_options()
        ]

    def test_commands(self):
        return [
            "python test.py "
            + str(
                opt.remove("display_env").set(
                    data_domainA=os.path.join("val", opt.kvs["data_domainA"]),
                    data_domainB=os.path.join("val", opt.kvs["data_domainB"]),
                    num_test=1000,
                )
            )
            for opt in self.common_options()
        ]
