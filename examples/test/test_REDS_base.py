import spikezoo as sz
from spikezoo.pipeline import EnsemblePipeline, EnsemblePipelineConfig
pipeline = EnsemblePipeline(
    cfg=EnsemblePipelineConfig(
        # save folder
        version="v023",
        save_folder="results",
        exp_name="REDS_BASE",
        # save metric or not
        save_metric=True,
        metric_names=["psnr", "ssim", "lpips", "niqe", "brisque","piqe"],
        # save image or not
        save_img=True,
        img_norm=False,
    ),
    model_cfg_list=[
        sz.METHOD.BASE,sz.METHOD.TFP,sz.METHOD.TFI,sz.METHOD.SPK2IMGNET,sz.METHOD.WGSE,
        sz.METHOD.SSML,sz.METHOD.BSF,sz.METHOD.STIR,sz.METHOD.SPIKECLIP,sz.METHOD.SSIR],
    dataset_cfg=sz.DATASET.REDS_BASE, 
)
pipeline.cal_metrics()
pipeline.cal_params()